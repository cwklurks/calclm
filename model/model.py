"""CalcLM v0: llama2.c-style decoder with BitNet b1.58 ternary linears (QAT).

The forward pass fake-quantizes exactly the tensors the integer reference (intref.py)
quantizes, so the two agree up to integer rounding:
- token and position embeddings: int8 per row, PF8 row scale;
- every linear: ternary weights, PF7 per-row scale, int8 absmax input with PF8 scale;
- q, k, v: int8 with static per-layer scales (EMA of the absmax during training);
- softmax: uint8 weights from an exp table indexed by rounded integer score differences.
RMSNorm, SiLU and the residual stream stay in float here; intref uses int24 fixed point and
tables for them.

Extra per-token inputs (v2, `feat_cats` / `feat_bag`, none by default): categorical features
and one multi-hot bag, embedded with the same int8-per-row QuantEmbedding scheme as tokens
in one table `feat_emb` and added to the residual stream with the token and position rows.
Row layout and the exact integer computation are in INTREF.md, "Extra input features".
With `feat_fold` (E3, PLAN-v3-E3.md) a single categorical feature (a per-program tag) is
instead folded into the token rows: token t under tag c reads Q(tok[t] + tag[c]), so
fold_feature(model, c) is a plain model and its export runs unchanged on the calculator.

Positions (`pos`): "learned" (v1, an int8 table added to the input), or the relative schemes
"alibi" (integer distance penalties with power-of-two slopes in the softmax, INTREF.md 12) and
"nope" (none). Relative models attend over a sliding window of max_seq_len tokens, so they run
on sequences of any length; the calculator keeps max_seq_len K/V entries in a ring.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.quant import (
    act_quant as _act_quant, attn_lut_shift, exp_lut_index, exp_lut_weight, pf_value,
    round_half_up, weight_quant,
)


QUANT_ENABLED = True  # Training-only switch; evaluation always quantizes.


def set_quant_enabled(enabled: bool) -> None:
    """Select the training path. Eval and export always use quantization."""
    global QUANT_ENABLED
    QUANT_ENABLED = bool(enabled)


def act_quant(x, training=True):
    """Training uses identity and unit scale during float warmup; eval always quantizes."""
    if training and not QUANT_ENABLED:
        return x, 1.0
    return _act_quant(x)


POSITION_SCHEMES = ("learned", "alibi", "nope")


def alibi_exponents(n_heads: int) -> list[int]:
    """ALiBi slope of head h: 2**(n_heads - 1 - h) exp-table steps per token of distance."""
    return [n_heads - 1 - h for h in range(n_heads)]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 812
    dim: int = 192
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 1
    hidden_dim: int = 512
    max_seq_len: int = 256
    norm_eps: float = 1e-6  # intref has no eps; keep the mismatch negligible
    lut_max: int = 512  # attention exp table length cap
    ema_momentum: float = 0.99
    feat_cats: tuple[int, ...] = ()  # values of each categorical input feature (one row each)
    feat_bag: int = 0  # bits of the multi-hot bag feature (one row per set bit), <= 64
    feat_shuffle: bool = False  # A2 control: feed the features of another program (data side)
    feat_fold: bool = False  # E3: one categorical feature folded into the token rows
    pos: str = "learned"  # "learned" (v1), "alibi" or "nope" (sliding window of max_seq_len)

    def __post_init__(self):
        if self.pos not in POSITION_SCHEMES:
            raise ValueError(f"pos must be one of {POSITION_SCHEMES}")
        object.__setattr__(self, "feat_cats", tuple(int(n) for n in self.feat_cats))
        if any(not 1 <= n <= 256 for n in self.feat_cats):
            raise ValueError("each categorical feature needs 1..256 values (stored as uint8)")
        if not 0 <= self.feat_bag <= 64:
            raise ValueError("the bag feature has 0..64 bits (one uint64 per token)")
        if self.feat_shuffle and not self.feat_rows:
            raise ValueError("feat_shuffle needs features")
        if self.feat_fold and (len(self.feat_cats) != 1 or self.feat_bag):
            raise ValueError("feat_fold needs exactly one categorical feature and no bag")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def feat_offsets(self) -> tuple[int, ...]:
        """First row of each categorical feature in feat_emb; bag bit b is row sum(cats) + b."""
        return tuple(sum(self.feat_cats[:i]) for i in range(len(self.feat_cats)))

    @property
    def feat_rows(self) -> int:
        return sum(self.feat_cats) + self.feat_bag

    def to_dict(self) -> dict:
        """Plain dict; feature fields only when used, so v1 configs serialize as before."""
        d = asdict(self)
        for k in ("feat_cats", "feat_bag", "feat_shuffle", "feat_fold"):
            if not d[k]:
                del d[k]
        if d["pos"] == "learned":
            del d["pos"]
        if "feat_cats" in d:
            d["feat_cats"] = list(d["feat_cats"])
        return d


class BitLinear(nn.Module):
    """y = (int8 x) @ ternary(W).T * row_scale * sx; straight-through gradients.

    With `post`, the row scale is PF7(absmean * post), i.e. the output is expressed in units
    of 1/post (used to land q/k/v directly on their static int8 grids).
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x8, sx, post=None):
        if self.training and not QUANT_ENABLED:  # latent float weights, no ternary STE
            return F.linear(x8, self.weight)
        w = self.weight
        wt, bm, be = weight_quant(w, post)
        beta = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_ste = w / beta + (wt.to(w.dtype) - w / beta).detach()
        acc = F.linear(x8, w_ste.to(x8.dtype))
        scale = (bm * torch.pow(2.0, be)).to(sx.dtype)
        return acc.to(sx.dtype) * scale * sx


class QuantEmbedding(nn.Module):
    """Embedding table fake-quantized to int8 with a PF8 scale per row."""

    def __init__(self, n: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, dim) * 0.02)

    def quantized(self):
        return quantize_rows(self.weight)

    def ste_weight(self):
        """Quantized rows with STE gradients, or latent rows during float warmup."""
        return self.weight if self.training and not QUANT_ENABLED else ste_rows(self.weight)

    def forward(self, idx):
        return F.embedding(idx, self.ste_weight())


def quantize_rows(w):
    """int8 values and PF8 scale of every row (last dim) of w."""
    with torch.no_grad():
        s = pf_value(w.abs().amax(dim=-1, keepdim=True) / 127.0, 8)
        s = torch.where(s > 0, s, torch.ones_like(s))
        w8 = torch.clamp(round_half_up(w / s), -127, 127)
    return w8, s


def ste_rows(w):
    """w on its rows' int8 grid, with straight-through gradients to w."""
    w8, s = quantize_rows(w)
    return w + (w8 * s - w).detach()


def unpack_bag(bag, n_bits: int):
    """(...,) int64 bit sets -> (..., n_bits) 0/1 floats; bit 63 is the int64 sign bit."""
    shifts = torch.arange(n_bits, device=bag.device)
    return ((bag.unsqueeze(-1) >> shifts) & 1).float()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight


def static_round(y):
    """int8 with a static scale: clamp to +-127 and round; straight-through inside the range."""
    yc = torch.clamp(y, -127, 127)
    return yc + (round_half_up(yc.detach()) - yc).detach()


def attention_core(q8, k8, v8, u, sv, n_heads: int, lut_max: int, alibi=None, window=None):
    """Causal MQA over int8 q/k/v with the uint8 exp-table softmax. Returns (o, weights).

    Autocast is disabled so scores and weighted sums are exact integers in fp32 (TF32 is
    also exact here: the int8 operands fit its significand and accumulation is fp32).
    alibi: per-head slope exponents (ALiBi); window: each query sees its last `window` keys
    (itself included). Without either, this is the v1 computation.
    """
    with torch.autocast(q8.device.type, enabled=False):
        if alibi is None and (window is None or q8.shape[1] <= window):
            return _attention_core(q8.float(), k8.float(), v8.float(), u, sv, n_heads, lut_max)
        return _windowed_attention(q8.float(), k8.float(), v8.float(), u, sv, n_heads, lut_max,
                                   alibi, window or q8.shape[1])


def _windowed_attention(q8, k8, v8, u, sv, n_heads, lut_max, alibi, window):
    """Sliding-window attention in query blocks of `window` (memory O(T * window)); with
    ALiBi, z = s - (min(dist << e_h, lut_max) << sh) before the max subtraction (INTREF.md 12).
    Returns the weights only for a single block (T <= window)."""
    b, t, _ = q8.shape
    q = q8.view(b, t, n_heads, -1).transpose(1, 2)
    k = k8.view(b, t, 1, -1).transpose(1, 2)
    v = v8.view(b, t, 1, -1).transpose(1, 2)
    sh = attn_lut_shift(u, lut_max)
    step = torch.pow(2.0, sh)
    outs, weights = [], None
    for qs in range(0, t, window):
        qe, ks = min(qs + window, t), max(0, qs - window + 1)
        s = q[:, :, qs:qe] @ k[:, :, ks:qe].transpose(-1, -2)          # exact integer scores
        i = torch.arange(qs, qe, device=s.device)[:, None]
        j = torch.arange(ks, qe, device=s.device)[None, :]
        dist = i - j
        if alibi is not None:
            ex = torch.pow(2.0, alibi.float())[:, None, None]
            s = s - torch.clamp(dist.clamp(min=0) * ex, max=lut_max) * step
        s = s.masked_fill(~((dist >= 0) & (dist < window)), float("-inf"))
        delta = s.amax(dim=-1, keepdim=True).detach() - s
        hard = exp_lut_weight(exp_lut_index(delta.detach(), sh), u * step, lut_max)
        soft = 255.0 * torch.exp(-delta * u)
        e = soft + (hard - soft).detach()
        outs.append((e @ v[:, :, ks:qe]) / e.sum(dim=-1, keepdim=True) * sv)
        if t <= window:
            weights = e
    o = torch.cat(outs, dim=2)
    return o.transpose(1, 2).reshape(b, t, -1), weights


def _attention_core(q8, k8, v8, u, sv, n_heads: int, lut_max: int):
    b, t, _ = q8.shape
    q = q8.view(b, t, n_heads, -1).transpose(1, 2).float()
    k = k8.view(b, t, 1, -1).transpose(1, 2).float()
    v = v8.view(b, t, 1, -1).transpose(1, 2).float()
    s = q @ k.transpose(-1, -2)  # exact integer scores
    causal = torch.ones(t, t, dtype=torch.bool, device=s.device).tril()
    s = s.masked_fill(~causal, float("-inf"))
    delta = s.amax(dim=-1, keepdim=True).detach() - s
    sh = attn_lut_shift(u, lut_max)
    hard = exp_lut_weight(exp_lut_index(delta.detach(), sh), u * torch.pow(2.0, sh), lut_max)
    soft = 255.0 * torch.exp(-delta * u)
    e = soft + (hard - soft).detach()
    o = (e @ v) / e.sum(dim=-1, keepdim=True) * sv
    return o.transpose(1, 2).reshape(b, t, -1), e


def _float_attention(q, k, v, n_heads: int, alibi, window, lut_max):
    """Float multi-query attention with causal/window masks and capped ALiBi distance.

    Distance penalties cap at lut_max, as on the integer path. The conversion to nats
    uses the nominal LUT span ln(510), without integer power-of-two scale rounding.
    """
    b, t, _ = q.shape
    hd = k.shape[-1]
    q = q.view(b, t, n_heads, hd).transpose(1, 2)
    k = k.view(b, t, 1, hd).transpose(1, 2)
    v = v.view(b, t, 1, hd).transpose(1, 2)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    i = torch.arange(t, device=q.device)[:, None]
    j = torch.arange(t, device=q.device)[None, :]
    dist = i - j
    if alibi is not None:
        ex = torch.pow(2.0, alibi.float())[:, None, None]
        pen = torch.clamp(dist.clamp(min=0).to(s.dtype) * ex, max=float(lut_max))
        s = s - pen * (math.log(510.0) / (lut_max - 1))
    w = window or t
    s = s.masked_fill(~((dist >= 0) & (dist < w)), float("-inf"))
    e = torch.softmax(s, dim=-1)
    o = (e @ v).transpose(1, 2).reshape(b, t, -1)
    return o, e


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hd, kv = cfg.head_dim, cfg.n_kv_heads
        if kv != 1:
            raise ValueError("intref implements multi-query attention (n_kv_heads=1) only")
        self.cfg = cfg
        self.wq = BitLinear(cfg.dim, cfg.n_heads * hd)
        self.wk = BitLinear(cfg.dim, kv * hd)
        self.wv = BitLinear(cfg.dim, kv * hd)
        self.wo = BitLinear(cfg.n_heads * hd, cfg.dim)
        self.register_buffer("amax", torch.zeros(3))  # static absmax of q, k, v
        self.register_buffer("alibi", torch.tensor(alibi_exponents(cfg.n_heads))
                             if cfg.pos == "alibi" else None, persistent=False)
        self.window = None if cfg.pos == "learned" else cfg.max_seq_len
        self.calibrating = False
        self.capture = False
        self.last_weights = None

    def scales(self):
        return self.amax / 127.0

    def forward(self, x8, sx):
        lins = (self.wq, self.wk, self.wv)
        if self.training and not QUANT_ENABLED:  # float q/k/v and real softmax
            q, k, v = (lin(x8, sx) for lin in lins)
            o, e = _float_attention(q, k, v, self.cfg.n_heads, self.alibi, self.window,
                                    self.cfg.lut_max)
            if self.capture:
                self.last_weights = e.detach()
            return self.wo(*act_quant(o, self.training))
        if self.calibrating:
            with torch.no_grad():
                self.amax.copy_(torch.stack([lin(x8, sx).abs().max() for lin in lins]))
        s = self.scales()
        ys = [lin(x8, sx, post=1.0 / s[i]) for i, lin in enumerate(lins)]
        if self.training and not self.calibrating:
            with torch.no_grad():
                batch = torch.stack([y.detach().abs().max() for y in ys]).float() * s
                m = self.cfg.ema_momentum
                self.amax.mul_(m).add_((1 - m) * batch)
        q8, k8, v8 = (static_round(y) for y in ys)
        u = s[0] * s[1] / math.sqrt(self.cfg.head_dim)
        o, e = attention_core(q8, k8, v8, u, s[2], self.cfg.n_heads, self.cfg.lut_max,
                              self.alibi, self.window)
        if self.capture:
            self.last_weights = e.detach()
        return self.wo(*act_quant(o, self.training))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = BitLinear(cfg.dim, cfg.hidden_dim)
        self.w2 = BitLinear(cfg.hidden_dim, cfg.dim)
        self.w3 = BitLinear(cfg.dim, cfg.hidden_dim)

    def forward(self, x8, sx):
        h = F.silu(self.w1(x8, sx)) * self.w3(x8, sx)
        return self.w2(*act_quant(h, self.training))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(*act_quant(self.attn_norm(x), self.training))
        return x + self.ffn(*act_quant(self.ffn_norm(x), self.training))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = QuantEmbedding(cfg.vocab_size, cfg.dim)
        if cfg.pos == "learned":
            self.pos_emb = QuantEmbedding(cfg.max_seq_len, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = BitLinear(cfg.dim, cfg.vocab_size)
        for name, p in self.named_parameters():  # llama2.c scaled init for residual outputs
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.feat_rows:  # created last: the other weights initialize as without features
            self.feat_emb = QuantEmbedding(cfg.feat_rows, cfg.dim)
            self.register_buffer("feat_offsets", torch.tensor(cfg.feat_offsets, dtype=torch.long),
                                 persistent=False)

    def feature_embedding(self, cats, bag):
        """Sum of the feature rows of every token: (B, T, dim). cats (B, T, n_cats) values,
        bag (B, T) int64 bit sets. Computed in fp32 (autocast off), like attention."""
        cfg = self.cfg
        if (cats is None) != (not cfg.feat_cats) or (bag is None) != (not cfg.feat_bag):
            raise ValueError(f"this model takes cats for {len(cfg.feat_cats)} categorical "
                             f"features and a {cfg.feat_bag}-bit bag (None when 0)")
        with torch.autocast(self.feat_emb.weight.device.type, enabled=False):
            w = self.feat_emb.ste_weight()
            n_cat = sum(cfg.feat_cats)
            h = 0.0
            if cfg.feat_cats:
                h = F.embedding(cats + self.feat_offsets, w[:n_cat]).sum(-2)
            if cfg.feat_bag:
                h = h + unpack_bag(bag, cfg.feat_bag).to(w.dtype) @ w[n_cat:]
        return h

    def folded_embedding(self, idx, cats, bag=None):
        """feat_fold: token t under tag c reads Q(tok[t] + tag[c]), the row that
        fold_feature(model, c) puts in its token table. Computed in fp32 (autocast off)."""
        if cats is None or cats.shape[-1] != 1 or bag is not None:
            raise ValueError("a folded model takes cats (B, T, 1) and no bag")
        with torch.autocast(self.feat_emb.weight.device.type, enabled=False):
            tables = self.tok_emb.weight.unsqueeze(0) + self.feat_emb.weight.unsqueeze(1)
            if QUANT_ENABLED or not self.training:
                tables = ste_rows(tables)
            return tables[cats[..., 0].long(), idx]

    def embed(self, idx, cats=None, bag=None):
        """Residual stream input: token row (folded with the tag under feat_fold) + position row
        (learned positions) + feature rows."""
        t = idx.shape[1]
        h = self.folded_embedding(idx, cats, bag) if self.cfg.feat_fold else self.tok_emb(idx)
        if self.cfg.pos == "learned":
            if t > self.cfg.max_seq_len:
                raise ValueError(f"learned positions stop at {self.cfg.max_seq_len} tokens")
            h = h + self.pos_emb(torch.arange(t, device=idx.device))
        if self.cfg.feat_rows and not self.cfg.feat_fold:
            h = h + self.feature_embedding(cats, bag)
        elif not self.cfg.feat_rows and (cats is not None or bag is not None):
            raise ValueError("this model has no feature inputs")
        return h

    def forward(self, idx, targets=None, cats=None, bag=None):
        h = self.embed(idx, cats, bag)
        for layer in self.layers:
            h = layer(h)
        logits = self.output(*act_quant(self.norm(h), self.training))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def calibrate(self, idx, cats=None, bag=None):
        """Initialize the static q/k/v scales from one batch (call before training)."""
        attns = [layer.attn for layer in self.layers]
        for a in attns:
            a.calibrating = True
        enabled = QUANT_ENABLED
        set_quant_enabled(True)
        try:
            self(idx, cats=cats, bag=bag)
        finally:
            set_quant_enabled(enabled)
            for a in attns:
                a.calibrating = False

    def ternary_param_count(self) -> int:
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, BitLinear))

    def embedding_param_count(self) -> int:
        feat = self.feat_emb.weight.numel() if self.cfg.feat_rows else 0
        pos = self.pos_emb.weight.numel() if self.cfg.pos == "learned" else 0
        return self.tok_emb.weight.numel() + pos + feat


def fold_feature(model: Transformer, value: int) -> Transformer:
    """A plain copy of a feat_fold model for one tag value: token table tok + tag[value], no
    feature table. Its forward equals the source's under that tag (PLAN-v3-E3.md)."""
    cfg = model.cfg
    if not cfg.feat_fold:
        raise ValueError("fold_feature needs a feat_fold model")
    if not 0 <= value < cfg.feat_cats[0]:
        raise ValueError(f"tag value {value} is outside 0..{cfg.feat_cats[0] - 1}")
    plain = ModelConfig(**{**cfg.to_dict(), "feat_cats": (), "feat_shuffle": False,
                           "feat_fold": False})
    with torch.random.fork_rng(devices=[]):     # the throwaway init must not move the global RNG
        out = Transformer(plain)
    out.load_state_dict({k: v for k, v in model.state_dict().items()
                         if not k.startswith("feat_emb.")})
    with torch.no_grad():
        out.tok_emb.weight.copy_(model.tok_emb.weight + model.feat_emb.weight[value])
    return out.to(model.tok_emb.weight.device).train(model.training)


def load_checkpoint(path, map_location="cpu") -> Transformer:
    """Rebuild a Transformer (eval mode) from a train.py checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = Transformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


@torch.no_grad()
def attention_weights(model: Transformer, idx, layer: int):
    """uint8 softmax weights (B, H, T, T) of one layer, for tests and inspection."""
    attn = model.layers[layer].attn
    attn.capture = True
    try:
        model(idx)
        return attn.last_weights
    finally:
        attn.capture = False
        attn.last_weights = None
