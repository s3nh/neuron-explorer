"""Architecture adapter: finds the decoder, MLP and attention parts of any Hugging Face causal LM, so the
explorer is not tied to Qwen3.5.

What it resolves:

- the decoder module and its layer list, final norm and lm head, across wrapper layouts
  (dense text models, VL wrappers such as Qwen3.5's `model.language_model`, GPT-NeoX, GPT-2);
- the MLP "neuron" site: SwiGLU (`gate_proj`/`up_proj`/`down_proj`, the neurons are silu(gate) * up, which is the
  down_proj input) and single-activation MLPs (`dense_h_to_4h`/`dense_4h_to_h`, `c_fc`/`c_proj`, `fc_in`/`fc_out`,
  `fc1`/`fc2`, the last also directly on the block as in OPT), and one projection holding gate and up (`gate_up_proj`);
- the attention kind of each layer: "full" softmax attention (separate q/k/v/o projections, optional q/k norms,
  optionally a gated q_proj as in Qwen3.5) or "linear" (Qwen3.5's Gated DeltaNet). Anything else reports
  `attn_kind(layer) is None`: MLP steering, reading, lenses and connections still work, the attention-head
  views are unavailable.

Loading uses `AutoModelForCausalLM`; the rotary helper is imported from the model's own transformers module
when available.
"""
import importlib
import os
import re
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def text_config(model_id):
    cfg = AutoConfig.from_pretrained(model_id)
    get = getattr(cfg, "get_text_config", None)
    if get is not None:
        try:
            return get()
        except TypeError:
            pass
    return cfg


def arch_dict(cfg):
    """What the frontend needs to draw the model: layer count and sizes, layer types, head counts. Missing
    fields (MoE sizes, linear-attention head counts on dense models) become None."""
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    inter = _first_attr(cfg, ("intermediate_size", "moe_intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size")) \
        or (4 * hidden if hidden else None)  # models that leave the MLP size out of the config use 4 * hidden
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    # only "linear_attention" differs for the views; sliding and other softmax variants count as full attention
    types = [t if t == "linear_attention" else "full_attention"
             for t in (getattr(cfg, "layer_types", None) or ["full_attention"] * (n_layers or 0))]
    heads = getattr(cfg, "num_attention_heads", None)
    return {"n_layers": n_layers,
            "hidden": hidden,
            "inter": inter,
            "layer_types": types,
            "attn_heads": heads,
            "kv_heads": getattr(cfg, "num_key_value_heads", None) or heads,
            "linear_k_heads": getattr(cfg, "linear_num_key_heads", None),
            "linear_v_heads": getattr(cfg, "linear_num_value_heads", None)}


def slug(model_id):
    """Short filesystem-safe name for per-model stats and caches, e.g. qwen3.5-4b, smollm2-135m."""
    return re.sub(r"[^a-z0-9.]+", "-", model_id.split("/")[-1].lower()).strip("-")


def cache_path(name, model_id=MODEL):
    """Cache files are model-specific: the ones shipped with the repo describe Qwen3.5-4B, so any other model reads
    and writes its own copies under a suffixed name."""
    model = slug(model_id)
    if model == "qwen3.5-4b":
        return os.path.join(CACHE, name)
    stem, ext = os.path.splitext(name)
    return os.path.join(CACHE, f"{stem}__{model}{ext}")


def _first_attr(obj, names):
    for name in names:
        if obj is not None and hasattr(obj, name):
            got = getattr(obj, name)
            if got is not None:
                return got
    return None


class Adapter:
    def __init__(self, model_id=MODEL):
        self.model_id = model_id
        self.cfg = text_config(model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = None
        self.lm = None  # the decoder: has .layers, the final norm
        self._rotary = None

    # ---------- loading ----------
    def load(self, dtype=torch.bfloat16, device=None):
        dev = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        m = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype).to(dev).eval()
        self.model = m
        self.lm = self._find_decoder(m)
        self.dev = dev
        return m

    def _find_decoder(self, m):
        """The module holding .layers (or .h) of transformer blocks, walking in through wrapper layouts."""
        for path in (("model", "language_model"), ("model",), ("language_model",), ("transformer",), ("gpt_neox",)):
            obj = m
            for step in path:
                obj = getattr(obj, step, None)
            if obj is not None and self._layer_list(obj) is not None:
                return obj
        # last resort: any submodule that owns a ModuleList of blocks
        for mod in m.modules():
            if mod is not m and self._layer_list(mod) is not None:
                return mod
        raise RuntimeError(f"could not find the decoder (layer list) inside {type(m).__name__}")

    @staticmethod
    def _layer_list(dec):
        return _first_attr(dec, ("layers", "h"))

    @property
    def layers(self):
        ls = self._layer_list(self.lm)
        if ls is None:
            raise RuntimeError("decoder has no layer list")
        return ls

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def n_neurons(self):
        return self.down_weight(self.layers[0]).shape[1]

    @property
    def final_norm(self):
        norm = _first_attr(self.lm, ("norm", "final_layernorm", "final_layer_norm", "ln_f"))
        if norm is None:
            raise RuntimeError("decoder has no final norm")
        return norm

    @property
    def lm_head(self):
        head = _first_attr(self.model, ("lm_head", "embed_out"))
        if head is None:
            raise RuntimeError("model has no lm head")
        return head

    # ---------- MLP ----------
    @staticmethod
    def _mlp(layer):
        """The block's MLP module; OPT keeps its projections directly on the block."""
        mlp = _first_attr(layer, ("mlp", "feed_forward"))
        return layer if mlp is None else mlp

    @staticmethod
    def mlp_down(layer):
        """The projection whose *input* is the neuron activations (silu(gate) * up for SwiGLU). Steering and
        reading hook this module's input; its weight column n is what neuron n writes to the residual."""
        down = _first_attr(Adapter._mlp(layer), ("down_proj", "dense_4h_to_h", "c_proj", "fc_out", "fc2"))
        if down is None:
            raise RuntimeError(f"could not find the MLP output projection in {type(layer).__name__}")
        return down

    @staticmethod
    def _matrix(proj):
        """A projection's weight as [out, in]; GPT-2's Conv1D stores it the other way round."""
        return proj.weight.T if type(proj).__name__ == "Conv1D" else proj.weight

    @staticmethod
    def down_weight(layer):
        """[hidden, neurons]: column n is what neuron n writes into the residual."""
        return Adapter._matrix(Adapter.mlp_down(layer))

    @staticmethod
    def mlp_rows(layer):
        """The projections whose weight row n reads the residual into neuron n. SwiGLU has two (gate and up;
        their activations multiply), GPT-NeoX/GPT-2 one. Connections score with the product over these rows. Each
        result's .weight is [neurons, hidden]."""
        mlp = Adapter._mlp(layer)
        gate = _first_attr(mlp, ("gate_proj",))
        up = _first_attr(mlp, ("up_proj",))
        if gate is not None and up is not None:
            return (gate, up)
        both = _first_attr(mlp, ("gate_up_proj",))
        if both is not None:
            return tuple(SimpleNamespace(weight=w) for w in both.weight.chunk(2, 0))
        single = _first_attr(mlp, ("dense_h_to_4h", "c_fc", "fc_in", "fc1"))
        if single is None:
            raise RuntimeError("could not find the MLP input projections")
        return (SimpleNamespace(weight=Adapter._matrix(single)),)

    @staticmethod
    def pre_mlp_norm(layer):
        """The norm whose output enters the MLP. Parallel blocks (Phi, Cohere, Falcon, GPT-J) share the one norm."""
        return _first_attr(layer, ("pre_feedforward_layernorm", "post_attention_layernorm", "ln_2", "final_layer_norm",
                                   "ln_mlp", "input_layernorm", "ln_1"))

    @staticmethod
    def pre_attn_norm(layer):
        return _first_attr(layer, ("input_layernorm", "ln_1", "self_attn_layer_norm"))

    @staticmethod
    def norm_eps(norm):
        return _first_attr(norm, ("eps", "variance_epsilon")) or 1e-5

    @staticmethod
    def norm_gain(norm):
        """The per-dimension factor a norm applies after dividing by the rms: (1 + weight) in Qwen3.5 and Gemma, weight
        in most others, 1 without learned weights. Measured on a zero-mean unit-rms input (the difference of +x and -x
        cancels any bias), so no model needs its own rule."""
        w = getattr(norm, "weight", None)
        if w is None:
            return torch.ones(1)
        with torch.no_grad():
            x = torch.ones_like(w)
            x[1::2] = -1
            return ((norm(x[None, None]) - norm(-x[None, None]))[0, 0] / (2 * x)).float()

    def down_col(self, layer_i, n):
        """The direction neuron n writes into the residual stream (a down-projection weight column)."""
        return self.down_weight(self.layers[layer_i])[:, n].detach().float()

    def read_rows(self, layer_i, n, scaled=True):
        """Row tensors of neuron n's input projections, optionally scaled by (1 + pre-MLP norm weight)."""
        layer = self.layers[layer_i]
        rows = [p.weight[n].float() for p in self.mlp_rows(layer)]
        if scaled:
            w = self.norm_gain(self.pre_mlp_norm(layer)).to(rows[0].device)
            rows = [r * w for r in rows]
        return rows

    # ---------- attention ----------
    def attn_kind(self, layer):
        """"full" | "linear" | None (unknown layout: head views unavailable)."""
        a = self.attn_module(layer)
        if a is None:
            return None
        if all(hasattr(a, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return "full"
        if hasattr(a, "in_proj_qkv") and hasattr(a, "out_proj"):
            return "linear"
        return None

    def attn_module(self, layer):
        return _first_attr(layer, ("self_attn", "linear_attn", "attention"))

    def full_attn_spec(self, a):
        """Head geometry of a separate-projection softmax attention. Handles GQA (nkv < nq), optional
        q/k norms, and a gated q_proj whose output doubles (Qwen3.5)."""
        hd = getattr(a, "head_dim", None)
        if hd is None:
            nq_cfg = getattr(self.cfg, "num_attention_heads", None)
            hd = a.q_proj.out_features // nq_cfg if nq_cfg else a.head_dim
        nq = a.o_proj.in_features // hd
        nkv = max(1, a.k_proj.out_features // hd)
        return {"head_dim": hd, "nq": nq, "nkv": nkv,
                "gated_q": a.q_proj.out_features == 2 * nq * hd,
                "q_norm": getattr(a, "q_norm", None), "k_norm": getattr(a, "k_norm", None),
                "scaling": getattr(a, "scaling", hd ** -0.5)}

    def n_heads(self, layer_i):
        layer = self.layers[layer_i]
        a = self.attn_module(layer)
        if self.attn_kind(layer) == "full":
            return self.full_attn_spec(a)["nq"]
        if self.attn_kind(layer) == "linear":
            return a.num_v_heads
        return 0  # attention layout the adapter doesn't know: no head views

    def head_io_weights(self, layer, g):
        """[(WV, WO)] per head: the head's value-read rows (scaled by g = 1 + pre-attention norm weight) and its
        output columns, for the weight-space read/write views. None when the attention layout is unsupported."""
        kind = self.attn_kind(layer)
        a = self.attn_module(layer)
        if kind == "full":
            spec = self.full_attn_spec(a)
            hd, nq, nkv = spec["head_dim"], spec["nq"], spec["nkv"]
            return [(a.v_proj.weight[(h // (nq // nkv)) * hd:(h // (nq // nkv) + 1) * hd].float() * g,
                     a.o_proj.weight[:, h * hd:(h + 1) * hd].float()) for h in range(nq)]
        if kind == "linear":
            hd = a.head_v_dim
            o = 2 * a.key_dim
            return [(a.in_proj_qkv.weight[o + h * hd:o + (h + 1) * hd].float() * g,
                     a.out_proj.weight[:, h * hd:(h + 1) * hd].float() * a.norm.weight.float())
                    for h in range(a.num_v_heads)]
        return None

    def head_qk_weights(self, layer, head, g):
        """(WV, WO, WQ, WK) of one head, or None when unsupported."""
        kind = self.attn_kind(layer)
        a = self.attn_module(layer)
        if kind == "full":
            spec = self.full_attn_spec(a)
            hd, nq, nkv = spec["head_dim"], spec["nq"], spec["nkv"]
            kv = head // (nq // nkv)
            WV = a.v_proj.weight[kv * hd:(kv + 1) * hd].float() * g
            WO = a.o_proj.weight[:, head * hd:(head + 1) * hd].float()
            if spec["gated_q"]:
                WQ = a.q_proj.weight.view(nq, 2 * hd, -1)[head, :hd].float() * g
            else:
                WQ = a.q_proj.weight[head * hd:(head + 1) * hd].float() * g
            WK = a.k_proj.weight[kv * hd:(kv + 1) * hd].float() * g
            return WV, WO, WQ, WK
        if kind == "linear":
            hd = a.head_v_dim
            WV = a.in_proj_qkv.weight[2 * a.key_dim + head * hd: 2 * a.key_dim + (head + 1) * hd].float() * g
            WO = a.out_proj.weight[:, head * hd:(head + 1) * hd].float() * a.norm.weight.float()
            kh, kd = head // (a.num_v_heads // a.num_k_heads), a.head_k_dim
            WQ = a.in_proj_qkv.weight[kh * kd:(kh + 1) * kd].float() * g
            WK = a.in_proj_qkv.weight[a.key_dim + kh * kd:a.key_dim + (kh + 1) * kd].float() * g
            return WV, WO, WQ, WK
        return None

    def head_qk_units(self, layer, role, g):
        """[(W, [head indices])]: one weight block per query (role "q") or key (role "k") read space, with the
        output heads that share it (grouped keys in full attention, shared q/k heads in linear attention)."""
        kind = self.attn_kind(layer)
        a = self.attn_module(layer)
        if kind == "full":
            spec = self.full_attn_spec(a)
            hd, nq, nkv = spec["head_dim"], spec["nq"], spec["nkv"]
            if role == "q":
                if spec["gated_q"]:
                    Wq = a.q_proj.weight.view(nq, 2 * hd, -1)[:, :hd]
                else:
                    Wq = a.q_proj.weight.view(nq, hd, -1)
                return [(Wq[h].float() * g, [h]) for h in range(nq)], hd
            rep = nq // nkv
            return [(a.k_proj.weight[kv * hd:(kv + 1) * hd].float() * g,
                     list(range(kv * rep, (kv + 1) * rep))) for kv in range(nkv)], hd
        if kind == "linear":
            hd = a.head_k_dim
            rep = a.num_v_heads // a.num_k_heads
            o = 0 if role == "q" else a.key_dim
            return [(a.in_proj_qkv.weight[o + kh * hd:o + (kh + 1) * hd].float() * g,
                     list(range(kh * rep, (kh + 1) * rep))) for kh in range(a.num_k_heads)], hd
        return None, 0

    def rotary(self):
        """apply_rotary_pos_emb from the model's own transformers module, or None when unavailable."""
        if self._rotary is None:
            try:
                mod = importlib.import_module(
                    f"transformers.models.{self.cfg.model_type}.modeling_{self.cfg.model_type}")
                self._rotary = getattr(mod, "apply_rotary_pos_emb", None) or False
            except (ImportError, AttributeError):
                self._rotary = False
        return self._rotary or None

    def modeling_attr(self, name):
        """Any helper (e.g. l2norm) from the model's own transformers module."""
        try:
            mod = importlib.import_module(
                f"transformers.models.{self.cfg.model_type}.modeling_{self.cfg.model_type}")
            return getattr(mod, name, None)
        except (ImportError, AttributeError):
            return None

    # ---------- chat ----------
    def messages_ids(self, messages):
        """Token ids of a chat, using the model's template when it has one. `enable_thinking` is only passed
        when the template accepts it."""
        if getattr(self.tok, "chat_template", None):
            for kw in ({"add_generation_prompt": True, "enable_thinking": False},
                       {"add_generation_prompt": True}):
                try:
                    enc = self.tok.apply_chat_template(messages, tokenize=True, **kw)
                    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)
                except (TypeError, ValueError):
                    continue
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return self.tok(text).input_ids

    def eos_ids(self):
        """Token ids that end a reply: the tokenizer's eos plus common chat end markers."""
        out = set()
        for cand in (self.tok.eos_token_id,
                     self.tok.convert_tokens_to_ids("<|im_end|>"),
                     self.tok.convert_tokens_to_ids("<|eot_id|>")):
            if isinstance(cand, int) and cand >= 0 and cand != self.tok.unk_token_id:
                out.add(cand)
        return out

    # ---------- safetensors layout (build scripts) ----------
    @staticmethod
    def weight_prefixes(weight_map):
        """The state-dict prefix of the layer stack, e.g. 'model.language_model.layers.' or 'model.layers.',
        found from any key holding a down projection."""
        for name in weight_map:
            m = re.match(r"^(.*(?:^|\.)layers)\.0\.mlp\.down_proj\.weight$", name)
            if m:
                return m.group(1) + "."
        for name in weight_map:
            m = re.match(r"^(.*(?:^|\.)layers)\.0\.", name)
            if m:
                return m.group(1) + "."
        raise RuntimeError("no layer-stack keys found in the safetensors index")
