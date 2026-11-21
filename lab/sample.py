"""Fast batched sampling from a QAT checkpoint on the GPU, with intref's sampler semantics.

    uv run python -m lab.sample --ckpt runs/v1/best.pt --n 1000 --out runs/lab/x/samples.jsonl

`QATDecoder` runs the fake-quant forward of model/model.py one position at a time with a KV
cache, for a whole batch. It keeps exact integers where the QAT model has integers (int8
activations, ternary weights, int8 q/k/v, uint8 softmax weights) and floats elsewhere, so it
matches `Transformer.forward` (up to float summation order) but not intref bit for bit: about
98% top-1 agreement with model/intref.py, the QAT-vs-intref gap.

`choose` is intref's sampler (model/intref.py `choose`) on int64 tensors: logits rounded to Q8,
top-k by (logit desc, id asc) over ids other than <pad>/<bos>/<unk>, exp-table weights at
temperature T*256, xorshift32 PRNG per sample, seed 0 meaning 2463534242. Given identical
integer logits it picks identical tokens and PRNG states.

v2: a `tracker` (BatchTracker interface: reset(batch), step(tokens) -> (cats, bag) per row)
runs alongside generation so a feature model gets, at every position, the features training
computes for the same program (state after that position's token; for the A2 control, those
of another row, model.data.shuffle_features). A `mask` (banned(tracker) -> (B, V) bool)
removes ids from the ranking exactly like <pad>/<bos>/<unk> before each draw.

G2a: relative-position models (ALiBi, NoPE) generate up to `cap` tokens with a sliding window of
max_seq_len: K/V live in a max_seq_len-entry ring (slot = position mod max_seq_len), as on the
calculator, and match the model's banded full-sequence forward. The tracker keeps running from
the program start.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.data import shuffle_features
from model.export import expw_table
from model.model import Transformer, load_checkpoint, unpack_bag
from model.quant import attn_lut_shift, exp_lut_weight, pf_value, weight_quant
from model.tokens import BOS, EOS, PAD, UNK

SEED0 = 2463534242
MASK32 = 0xFFFFFFFF
BANNED = (PAD, BOS, UNK)
_EXPW = torch.as_tensor(expw_table().astype(np.int64))


# ---- intref's sampler on tensors -----------------------------------------------------------

def logits_q8(logits: torch.Tensor) -> torch.Tensor:
    """Float logits (nats) -> int64 Q8 (1/256 nat), round half up like intref's scale_out."""
    return torch.floor(logits.double() * 256.0 + 0.5).to(torch.int64)


def xorshift32(x: torch.Tensor) -> torch.Tensor:
    x = x ^ ((x << 13) & MASK32)
    x = x ^ (x >> 17)
    return x ^ ((x << 5) & MASK32)


def choose(lq8: torch.Tensor, state: torch.Tensor, temperature_q8: int, top_k: int,
           greedy: bool = False, banned: torch.Tensor | None = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Next token per row from int64 Q8 logits (B, V); returns (tokens, new PRNG states).
    banned (B, V) bool: ids excluded from the ranking like <pad>/<bos>/<unk> (logit masks);
    the candidates are the first top_k allowed ids (fewer if fewer are allowed)."""
    b, v = lq8.shape
    ids = torch.arange(v, device=lq8.device)
    key = lq8 * v + (v - 1 - ids)                 # unique: equal logits rank the smaller id first
    low = torch.iinfo(torch.int64).min
    key[:, list(BANNED)] = low
    if banned is not None:
        key = key.masked_fill(banned.to(key.device), low)
        if bool((key == low).all(dim=1).any()):
            raise ValueError("a logit mask left a row with no allowed token")
    if greedy:
        return key.argmax(dim=1), state
    if not (33 <= temperature_q8 <= 65535 and 1 <= top_k <= 64):
        raise ValueError("temperature_q8 must be in [33, 65535] and top_k in [1, 64]")
    n_allowed = v - sum(i < v for i in BANNED)
    _, cand = key.topk(min(top_k, n_allowed), dim=1)  # sorted descending
    cl = lq8.gather(1, cand)
    inv_t = (32 << 16) // temperature_q8
    delta = torch.clamp(cl[:, :1] - cl, min=0, max=(1 << 15) - 1)  # min: masked fill-ins
    expw = _EXPW.to(lq8.device)
    idx = torch.clamp((delta * inv_t + (1 << 15)) >> 16, max=len(expw) - 1)
    w = expw[idx]
    if banned is not None:                        # masked fill-ins when < top_k ids are allowed
        w = torch.where(key.gather(1, cand) == low, 0, w)
    state = xorshift32(state)
    target = state % w.sum(dim=1)
    pick = (w.cumsum(dim=1) <= target[:, None]).sum(dim=1)
    return cand.gather(1, pick[:, None])[:, 0], state


# ---- cached QAT forward ------------------------------------------------------------------

def _act_quant(x: torch.Tensor):
    """model.quant.act_quant's forward value with exact integers: (x8, sx)."""
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    sx = pf_value(amax / 127.0, 8)
    sx = torch.where(sx > 0, sx, torch.ones_like(sx))
    return torch.clamp(torch.floor(x / sx + 0.5), -127, 127), sx


def _rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class _Lin:
    """Ternary weights (exact) and the PF7 row scale of one BitLinear."""

    def __init__(self, lin, device, post=None):
        wt, bm, be = weight_quant(lin.weight.detach().float().cpu(), post)
        self.wt = wt.to(device)
        self.scale = (bm * torch.pow(2.0, be)).float().to(device)

    def __call__(self, x8, sx):
        return F.linear(x8, self.wt) * self.scale * sx


class QATDecoder:
    def __init__(self, model: Transformer, device: str = "cuda"):
        cfg = model.cfg
        self.cfg, self.device = cfg, device
        with torch.no_grad():
            w8, s = model.tok_emb.quantized()
            self.tok = (w8 * s).float().to(device)
            self.pos = None
            if cfg.pos == "learned":
                w8, s = model.pos_emb.quantized()
                self.pos = (w8 * s).float().to(device)
            self.alibi = (model.layers[0].attn.alibi.float().to(device)
                          if cfg.pos == "alibi" else None)
            self.layers = [self._layer(blk, device) for blk in model.layers]
            self.norm = model.norm.weight.detach().float().to(device)
            self.out = _Lin(model.output, device)
            if cfg.feat_rows:
                w8, s = model.feat_emb.quantized()
                self.feat = (w8 * s).float().to(device)
                self.feat_off = torch.tensor(cfg.feat_offsets, dtype=torch.long, device=device)

    def feature_rows(self, cats, bag) -> torch.Tensor:
        """Sum of the feature rows for one position: cats (B, n_cats), bag (B,) -> (B, dim)."""
        cfg, n_cat = self.cfg, sum(self.cfg.feat_cats)
        h = 0.0
        if cfg.feat_cats:
            h = self.feat[cats.to(self.device) + self.feat_off].sum(-2)
        if cfg.feat_bag:
            h = h + unpack_bag(bag.to(self.device), cfg.feat_bag) @ self.feat[n_cat:]
        return h

    def _layer(self, blk, device):
        a, cfg = blk.attn, self.cfg
        s = a.scales().detach().float().cpu()
        u = s[0] * s[1] / math.sqrt(cfg.head_dim)
        sh = attn_lut_shift(u, cfg.lut_max)
        lut = exp_lut_weight(torch.arange(cfg.lut_max, dtype=torch.float32), u * torch.pow(2.0, sh),
                             cfg.lut_max)
        return {"attn_g": blk.attn_norm.weight.detach().float().to(device),
                "ffn_g": blk.ffn_norm.weight.detach().float().to(device),
                "wq": _Lin(a.wq, device, 1.0 / s[0]), "wk": _Lin(a.wk, device, 1.0 / s[1]),
                "wv": _Lin(a.wv, device, 1.0 / s[2]), "wo": _Lin(a.wo, device),
                "w1": _Lin(blk.ffn.w1, device), "w3": _Lin(blk.ffn.w3, device),
                "w2": _Lin(blk.ffn.w2, device),
                "lut": lut.to(device), "sh": float(sh), "sv": float(s[2])}

    def new_cache(self, batch: int):
        c = self.cfg
        shape = (c.n_layers, batch, c.max_seq_len, c.head_dim)
        return {"k": torch.zeros(shape, device=self.device), "v": torch.zeros(shape, device=self.device)}

    @torch.no_grad()
    def step(self, tok: torch.Tensor, pos: int, cache, cats=None, bag=None) -> torch.Tensor:
        """Feed tokens (B,) at position pos (with their features for a feature model);
        returns float logits (B, V)."""
        cfg = self.cfg
        b, hd, nh, w = tok.shape[0], cfg.head_dim, cfg.n_heads, cfg.max_seq_len
        if self.pos is not None:
            if pos >= w:
                raise ValueError(f"learned positions stop at {w} tokens")
            h = self.tok[tok] + self.pos[pos]
        else:
            h = self.tok[tok]
        slot, n = pos % w, min(pos + 1, w)                   # ring: slot of this token, keys held
        bias = None
        if self.alibi is not None:                           # distance of each held key
            dist = (slot - torch.arange(n, device=self.device)) % w
            bias = torch.clamp(dist[None, :] * torch.pow(2.0, self.alibi)[:, None],
                               max=cfg.lut_max)               # (H, n) table steps
        if cfg.feat_rows:
            h = h + self.feature_rows(cats, bag)
        for i, lp in enumerate(self.layers):
            x8, sx = _act_quant(_rmsnorm(h, lp["attn_g"], cfg.norm_eps))
            q8, k8, v8 = (torch.floor(torch.clamp(lp[n](x8, sx), -127, 127) + 0.5)
                          for n in ("wq", "wk", "wv"))
            cache["k"][i, :b, slot], cache["v"][i, :b, slot] = k8, v8
            keys, vals = cache["k"][i, :b, :n], cache["v"][i, :b, :n]
            scores = torch.bmm(q8.view(b, nh, hd), keys.transpose(1, 2))      # (B, H, T) exact
            sh = lp["sh"]
            if bias is not None:
                scores = scores - bias * 2.0 ** sh
            delta = scores.amax(dim=-1, keepdim=True) - scores
            idx = torch.floor((delta + (2.0 ** (sh - 1) if sh > 0 else 0.0)) / 2.0 ** sh)
            e = lp["lut"][torch.clamp(idx, max=cfg.lut_max - 1).long()]
            o = torch.bmm(e, vals) / e.sum(dim=-1, keepdim=True) * lp["sv"]
            h = h + lp["wo"](*_act_quant(o.reshape(b, nh * hd)))
            x8, sx = _act_quant(_rmsnorm(h, lp["ffn_g"], cfg.norm_eps))
            hid = F.silu(lp["w1"](x8, sx)) * lp["w3"](x8, sx)
            h = h + lp["w2"](*_act_quant(hid))
        return self.out(*_act_quant(_rmsnorm(h, self.norm, cfg.norm_eps)))

    @torch.no_grad()
    def sequence_logits(self, tokens: torch.Tensor, cats=None, bag=None) -> torch.Tensor:
        """(B, T) token ids (and (B, T, ...) features) fed one position at a time from 0 ->
        (B, T, V) float logits."""
        tokens = torch.as_tensor(tokens).to(self.device)
        cache = self.new_cache(tokens.shape[0])
        at = lambda a, p: None if a is None else a[:, p]  # noqa: E731
        return torch.stack([self.step(tokens[:, p], p, cache, at(cats, p), at(bag, p))
                            for p in range(tokens.shape[1])], 1)


class BatchTracker:
    """One step-function tracker per row (spec.Tracker / bind.Tracker interface: reset(),
    step(tok) -> (cats, bag)), stepped together. step returns cats (B, n_cats) and bag (B,)
    as int64 (bag bit 63 is the sign bit), the layout model.data uses."""

    def __init__(self, factory):
        self.factory, self.rows, self.batch = factory, [], 0

    def reset(self, batch: int) -> None:
        self.rows, self.batch = [self.factory() for _ in range(batch)], batch
        for r in self.rows:
            r.reset()

    def step(self, tokens) -> tuple[np.ndarray, np.ndarray]:
        out = [r.step(int(t)) for r, t in zip(self.rows, np.asarray(tokens).tolist())]
        cats = np.array([c for c, _ in out], dtype=np.int64).reshape(self.batch, -1)
        bag = np.array([int(b) for _, b in out], dtype=np.uint64).view(np.int64)
        return cats, bag


# ---- generation -----------------------------------------------------------------------------

def _model(ckpt_or_model, device) -> Transformer:
    if isinstance(ckpt_or_model, Transformer):
        return ckpt_or_model
    return load_checkpoint(ckpt_or_model)


def _model_features(cfg, feats, device) -> tuple:
    """Tracker output for one position -> the decoder's (cats, bag) arguments. The A2 control
    feeds row i the features of row i - 1: a row's own state could only come back to it after
    `batch` steps, so generate() requires batch > steps (training and val roll windows by n/2;
    their features never depend on what the model writes)."""
    if not cfg.feat_rows:
        return ()
    cats, bag = shuffle_features(*feats, shift=1) if cfg.feat_shuffle else feats
    return (torch.as_tensor(cats, device=device) if cfg.feat_cats else None,
            torch.as_tensor(bag, device=device) if cfg.feat_bag else None)


@torch.no_grad()
def generate(dec: QATDecoder, seeds: list[int], temperature_q8: int, top_k: int, greedy: bool,
             max_new: int, tracker=None, mask=None, cap: int | None = None) -> list[list[int]]:
    """intref.generate for a batch with no prompt: <bos>, then sample until <eos>, max_new
    tokens, or `cap` tokens in all (default: the context; past it only with relative
    positions, sliding window). tracker / mask: see the module doc."""
    b, dev, cfg = len(seeds), dec.device, dec.cfg
    ctx = cap or cfg.max_seq_len
    if ctx > cfg.max_seq_len and getattr(cfg, "pos", "learned") == "learned":
        raise ValueError(f"learned positions stop at {cfg.max_seq_len} tokens")
    if tracker is None and (getattr(cfg, "feat_rows", 0) or mask is not None):
        raise ValueError("feature inputs and logit masks need a tracker")
    if getattr(cfg, "feat_shuffle", False) and b <= min(max_new, ctx - 1):
        raise ValueError(f"shuffled features need more rows ({b}) than generated tokens")
    state = torch.tensor([s if s else SEED0 for s in seeds], dtype=torch.int64, device=dev)
    out = torch.full((b, ctx), PAD, dtype=torch.int64, device=dev)
    out[:, 0] = BOS
    done = torch.zeros(b, dtype=torch.bool, device=dev)
    length = torch.ones(b, dtype=torch.int64, device=dev)
    cache = dec.new_cache(b)
    # The shuffled control with a sliding window (G2a S2) feeds rows each other's tracker state
    # for up to `cap` tokens: a finished row continues like the packed training stream (<eos>,
    # then <bos> and a new program), so no row reads the state of a finished program.
    restart = getattr(cfg, "feat_shuffle", False) and getattr(cfg, "pos", "learned") != "learned"
    feats = ()
    if tracker is not None:
        tracker.reset(b)
        feats = _model_features(cfg, tracker.step(np.full(b, BOS)), dev)
    logits = dec.step(out[:, 0], 0, cache, *feats)
    fed = out[:, 0].clone()
    for t in range(1, min(max_new, ctx - 1) + 1):
        banned = None if mask is None else torch.as_tensor(mask.banned(tracker), device=dev)
        tok, state = choose(logits_q8(logits), state, temperature_q8, top_k, greedy, banned)
        out[:, t] = torch.where(done, PAD, tok)
        length += (~done).long()
        done |= tok == EOS
        if bool(done.all()) or t == ctx - 1:
            break
        if restart:
            tok = torch.where(done & (fed == EOS), BOS, tok)
        if tracker is not None:
            feats = _model_features(cfg, tracker.step(tok.cpu().numpy()), dev)
        logits = dec.step(tok, t, cache, *feats)
        fed = tok
    return [row[:n] for row, n in zip(out.tolist(), length.tolist())]


def sample(ckpt_or_model, n: int | None = None, seeds=None, temperature: float = 0.8,
           top_k: int = 20, greedy: bool = False, max_new: int | None = None,
           device: str = "cuda", batch: int = 1000, tracker=None, mask=None,
           cap: int | None = None) -> list[list[int]]:
    """Programs as token-id lists (<bos> ... [<eos>]), one per seed (default seeds 1..n), of
    at most `cap` tokens (default the context). tracker / mask: see the module doc. For the
    shuffled-feature control each batch is padded with extra rows (seeds 2**31 + k, dropped)
    so that it has more rows than generated tokens."""
    seeds = list(range(1, n + 1)) if seeds is None else [int(s) for s in seeds]
    if any(not 0 <= s < (1 << 32) for s in seeds):
        raise ValueError("seeds must be in [0, 2**32), as in intref")
    model = _model(ckpt_or_model, device)
    dec = QATDecoder(model, device)
    tq8 = round(temperature * 256)
    cap = cap or model.cfg.max_seq_len
    max_new = cap - 1 if max_new is None else max_new
    out = []
    for i in range(0, len(seeds), batch):
        chunk = seeds[i : i + batch]
        pad = []
        if model.cfg.feat_shuffle:
            pad = [(1 << 31) + k for k in range(max(0, min(max_new, cap - 1) + 1 - len(chunk)))]
        out += generate(dec, chunk + pad, tq8, top_k, greedy, max_new, tracker, mask,
                        cap)[: len(chunk)]
    return out


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ---- agreement with the integer reference -----------------------------------------------------

_IM = None


def _intref_init(export: str) -> None:
    global _IM
    from model.intref import IntModel

    _IM = IntModel.load(export)


def _intref_argmax(ctx: list[int]) -> list[int]:
    from model.intref import sequence_logits

    return sequence_logits(_IM, ctx).argmax(axis=1).tolist()


def intref_agreement(ckpt_or_model, export_dir, contexts, device: str = "cuda",
                     workers: int = 8) -> dict:
    """Top-1 (argmax) agreement of the cached QAT decoder with intref at every position of
    each context (lists of ids starting at <bos>, fed from position 0)."""
    dec = QATDecoder(_model(ckpt_or_model, device), device)
    contexts = [[int(t) for t in c] for c in contexts]
    with ProcessPoolExecutor(workers, initializer=_intref_init, initargs=(str(export_dir),)) as ex:
        want = list(ex.map(_intref_argmax, contexts))
    agree = total = 0
    for ctx, w in zip(contexts, want):
        got = dec.sequence_logits(torch.tensor([ctx]))[0].argmax(-1).tolist()
        agree += sum(g == x for g, x in zip(got, w))
        total += len(ctx)
    return {"top1_agree": agree / total, "n_positions": total, "n_contexts": len(contexts)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    t0 = time.time()
    out = sample(a.ckpt, seeds=seeds, temperature=a.temp, top_k=a.top_k, greedy=a.greedy,
                 device=a.device, batch=a.batch)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps({"seed": s, "temp": a.temp, "top_k": a.top_k, "ids": ids}) + "\n"
                             for s, ids in zip(seeds, out)))
    print(json.dumps({"n": len(out), "wall_s": round(time.time() - t0, 2), "out": str(a.out)}))


if __name__ == "__main__":
    main()
