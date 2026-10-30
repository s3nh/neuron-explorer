"""Seriated sheet layouts: places each layer's neurons on its 96x96 grid so that neurons with similar weights sit
together. Embeds each neuron by its down_proj column ("writes": the direction it adds to the residual stream) or its
gate and up rows ("reads"), cosine-normalised, PCA to 50 dims, UMAP to 2D, then snaps to the grid: sort by y into
rows of 96, then by x within each row.

    uv run python explorer/build_layout.py --kind writes

Writes explorer/cache/layout_<kind>.npz: cell[L, N] (int16, row * 96 + col for neuron n of layer L).
"""
import argparse
import json
import os

import numpy as np
import umap
from huggingface_hub import snapshot_download
from safetensors import safe_open

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arch import Adapter

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--kind", default="writes", choices=["writes", "reads"])
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--neighbors", type=int, default=30)
a = ap.parse_args()

root = snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"])
index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
_cfg = json.load(open(os.path.join(root, "config.json")))
cfg = _cfg.get("text_config", _cfg)
L, N = cfg["num_hidden_layers"], cfg["intermediate_size"]
side = int(round(N ** 0.5))
assert side * side == N


def tensor(name):
    with safe_open(os.path.join(root, index[name]), "pt") as f:
        return f.get_tensor(name).float().numpy()


PRE = Adapter.weight_prefixes(index)
cells = np.zeros((L, N), np.int16)
for l in range(L):
    pre = f"{PRE}{l}.mlp."
    if a.kind == "writes":
        X = tensor(pre + "down_proj.weight").T  # [N, hidden]
    else:
        X = np.concatenate([tensor(pre + "gate_proj.weight"), tensor(pre + "up_proj.weight")], 1)  # [N, 2*hidden]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, : a.pca] * S[: a.pca]
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for r in range(side):
        band = by_y[r * side:(r + 1) * side]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cells[l, band] = r * side + np.arange(side)
    print(f"L{l} done", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", f"layout_{a.kind}.npz")
np.savez(out, cell=cells)
print("->", out)
