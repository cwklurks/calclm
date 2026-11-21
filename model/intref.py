"""CalcLM integer reference inference: the bit-exact contract for the calculator (INTREF.md).

Loads only the export (model.bin + model.json), never the torch model. Inference uses Python
ints and numpy int64 only; ternary matvecs go through model.pack.lut_matvec.

Format 1 exports (learned positions) stop at the context. Format 2 exports (relative positions,
INTREF.md 12) attend over a sliding window of `ctx` tokens kept in a ring K/V cache, with ALiBi
penalties (or none for NoPE), and generate up to a cap (default DEFAULT_CAP). Generation can
apply the tracker's logit masks (model/intmask.py, INTREF.md 13). `IntModel(ex, horizons=True)`
adds ALiBi horizons (PLAN-v2.1.md E1): each head drops the keys whose penalty sits at its cap.

    uv run python -m model.intref --export runs/v0/export --n 3 --seed 1 --temp 0.8 --top-k 20
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from model.export import (
    POS_SCHEME_CODES, SILU_HALF, SILU_STEP_BITS, VERSION, VERSION_RELATIVE, Export, load_export,
)
from model.intops import (
    I64, NULL_COUNTER, MulCounter, bitlen, check_bits, clamp, norm_quant, normalize16, quant_vec, rsr,
    rsr_vec, scale_out, udiv, xorshift32,
)
from model.pack import lut_matvec
from model.tokens import BOS, EOS, PAD, UNK

INT24 = (-(1 << 23), (1 << 23) - 1)
INT16 = (-(1 << 15), (1 << 15) - 1)
SILU_SHIFT = 8 - SILU_STEP_BITS  # Q8 pre-activation -> table index
SILU_EDGE = SILU_HALF << SILU_SHIFT  # 2048 = 8.0 in Q8
BANNED = (PAD, BOS, UNK)
DEFAULT_CAP = 1024  # tokens (<bos> included) a relative-position model generates by default


@dataclass(frozen=True)
class IntConfig:
    """Shape of the exported model (integers only; v0 defaults)."""

    vocab_size: int = 812
    dim: int = 192
    n_layers: int = 6
    n_heads: int = 6
    head_dim: int = 32
    hidden_dim: int = 512
    max_seq_len: int = 256  # learned positions: the context; relative: the sliding window
    lut_max: int = 512
    pos: str = "learned"  # "learned", "alibi" or "nope"


@dataclass
class Linear:
    w: np.ndarray  # packed ternary, uint8 (rows, ceil(n_in/5))
    m: np.ndarray  # PF7 mantissa per row, int64 in [64, 127]
    e: np.ndarray  # exponent per row, int64


@dataclass
class Layer:
    attn_g: np.ndarray
    attn_eg: int
    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear
    lut: np.ndarray
    sh: int
    sv: tuple[int, int]  # PF15 static value scale
    ffn_g: np.ndarray
    ffn_eg: int
    w1: Linear
    w3: Linear
    w2: Linear
    e3: int  # max exponent of w3 rows (alignment exponent for t3)


@dataclass
class KVCache:
    k: np.ndarray  # (n_layers, ctx, head_dim) int8 values; position p lives in slot p mod ctx
    v: np.ndarray


def _ints(a) -> np.ndarray:
    return np.asarray(a).astype(I64)


def alibi_exponents(n_heads: int) -> list[int]:
    """Head h penalizes each token of distance by 2**(n_heads - 1 - h) exp-table steps."""
    return [n_heads - 1 - h for h in range(n_heads)]


def alibi_bias(exps, dist, sh: int, lut_max: int) -> np.ndarray:
    """ALiBi penalty in score units (INTREF.md 12): b[h, j] = min(dist[j] << e_h, lut_max) << sh."""
    e = np.asarray(exps, dtype=I64)[:, None]
    d = np.asarray(dist, dtype=I64)[None, :]
    return np.minimum(d << e, lut_max) << sh


def alibi_horizons(exps, lut_max: int) -> np.ndarray:
    """ALiBi horizons (PLAN-v2.1.md E1): head h's first distance whose penalty sits at the cap,
    i.e. (d << e_h) >= lut_max exactly when d >= ceil(lut_max / 2**e_h)."""
    return np.array([-(-lut_max >> e) for e in exps], dtype=I64)


def _pos_scheme(h: dict) -> str:
    if h["version"] == VERSION:
        return "learned"
    if h["version"] == VERSION_RELATIVE:
        return {code: name for name, code in POS_SCHEME_CODES.items()}[h["pos_scheme"]]
    raise ValueError(f"unknown export format version {h['version']}")


class IntModel:
    def __init__(self, ex: Export, horizons: bool = False):
        if "features" in ex.meta:
            raise NotImplementedError("intref has no feature inputs yet (INTREF.md 11)")
        h = ex.header()
        self.pos_scheme = _pos_scheme(h)
        positions = ex.meta.get("positions", {"scheme": "learned"})
        if positions["scheme"] != self.pos_scheme:
            raise ValueError(f"model.json says {positions['scheme']}, the header {self.pos_scheme}")
        self.cfg = IntConfig(vocab_size=h["vocab"], dim=h["dim"], n_layers=h["n_layers"],
                             n_heads=h["n_heads"], head_dim=h["head_dim"], hidden_dim=h["hidden"],
                             max_seq_len=h["ctx"], lut_max=h["lut_max"], pos=self.pos_scheme)
        self.alibi_exponents = None
        if self.pos_scheme == "alibi":
            self.alibi_exponents = alibi_exponents(h["n_heads"])
            if positions.get("alibi_exponents", self.alibi_exponents) != self.alibi_exponents:
                raise ValueError("only power-of-two ALiBi slopes 2**(H-1-h) are defined")
        self.horizons = None  # with horizons: head h attends only to distances < horizons[h]
        if horizons:
            if self.alibi_exponents is None:
                raise ValueError("ALiBi horizons need an ALiBi export")
            self.horizons = alibi_horizons(self.alibi_exponents, self.cfg.lut_max)
        self.F = h["F"]
        self.sqrtd = (h["sqrtd_m"], h["sqrtd_e"])
        self.vocab = ex.meta.get("vocab")
        lin = lambda p: Linear(np.asarray(ex.t(f"{p}.w")), _ints(ex.t(f"{p}.m")), _ints(ex.t(f"{p}.e")))
        self.layers = []
        for i, lh in enumerate(h["layers"]):
            p = f"layers.{i}"
            w3 = lin(f"{p}.w3")
            self.layers.append(Layer(
                attn_g=_ints(ex.t(f"{p}.attn_norm.g")), attn_eg=lh["attn_eg"],
                wq=lin(f"{p}.wq"), wk=lin(f"{p}.wk"), wv=lin(f"{p}.wv"), wo=lin(f"{p}.wo"),
                lut=_ints(ex.t(f"{p}.attn.lut")), sh=lh["attn_sh"], sv=(lh["sv_m"], lh["sv_e"]),
                ffn_g=_ints(ex.t(f"{p}.ffn_norm.g")), ffn_eg=lh["ffn_eg"],
                w1=lin(f"{p}.w1"), w3=w3, w2=lin(f"{p}.w2"), e3=int(w3.e.max())))
        self.norm_g, self.norm_eg = _ints(ex.t("norm.g")), h["norm_eg"]
        self.output = lin("output")
        self.tok, self.tok_m, self.tok_e = (_ints(ex.t(n)) for n in ("tok_emb", "tok_emb.m", "tok_emb.e"))
        if self.pos_scheme == "learned":
            self.pos, self.pos_m, self.pos_e = (_ints(ex.t(n))
                                                for n in ("pos_emb", "pos_emb.m", "pos_emb.e"))
        self.silu, self.expw = _ints(ex.t("silu")), _ints(ex.t("expw"))

    @classmethod
    def load(cls, out_dir, horizons: bool = False) -> "IntModel":
        return cls(load_export(out_dir), horizons=horizons)

    def new_cache(self) -> KVCache:
        shape = (self.cfg.n_layers, self.cfg.max_seq_len, self.cfg.head_dim)
        return KVCache(np.zeros(shape, dtype=I64), np.zeros(shape, dtype=I64))

    # ---- one token ------------------------------------------------------------------------

    def embed(self, tok: int, pos: int, counter) -> np.ndarray:
        e = rsr(self.tok[tok] * self.tok_m[tok], -(int(self.tok_e[tok]) + self.F))
        if self.pos_scheme != "learned":            # relative positions: the token row only
            counter.add("emb", self.cfg.dim)
            return clamp(e, *INT24)
        counter.add("emb", 2 * self.cfg.dim)
        p = rsr(self.pos[pos] * self.pos_m[pos], -(int(self.pos_e[pos]) + self.F))
        return clamp(e + p, *INT24)

    def attention(self, i: int, lp: Layer, r, pos: int, cache: KVCache, counter) -> np.ndarray:
        cfg = self.cfg
        win = cfg.max_seq_len
        x8, sx = norm_quant(r, lp.attn_g, lp.attn_eg, counter, self.sqrtd)
        q8, k8, v8 = (scale_out(lut_matvec(w.w, x8), w.m, w.e, sx, 0, -127, 127, counter)
                      for w in (lp.wq, lp.wk, lp.wv))
        slot, n = pos % win, min(pos + 1, win)  # ring slot of this token; keys held: pos-n+1..pos
        cache.k[i, slot], cache.v[i, slot] = k8, v8
        keys, vals = cache.k[i, :n], cache.v[i, :n]
        q = q8.reshape(cfg.n_heads, cfg.head_dim)
        scores = q @ keys.T  # (heads, n): 8x8 signed products
        use = None  # horizons: (heads, n), True for the keys each head attends to
        if self.alibi_exponents is not None:     # distance of slot j: (slot - j) mod win
            dist = (slot - np.arange(n)) % win
            scores = scores - alibi_bias(self.alibi_exponents, dist, lp.sh, cfg.lut_max)
            if self.horizons is not None:  # drop keys at or past the head's horizon
                use = dist[None, :] < self.horizons[:, None]
                scores = np.where(use, scores, 1 - (1 << 23))  # never the max (d = 0 is kept);
                # dropped keys are never computed on the device, so check_bits skips them too
            scores = check_bits(scores, 23, "ALiBi scores")
        mults = q.size * n if use is None else cfg.head_dim * int(use.sum())
        counter.add("attn_score", mults)
        idx = np.minimum(rsr(scores.max(axis=1, keepdims=True) - scores, lp.sh), len(lp.lut) - 1)
        e = lp.lut[idx]  # uint8 weights, max 255 per head
        if use is not None:
            e = np.where(use, e, 0)  # dropped keys: weight 0, out of the sum and of P.V
        a = e @ vals  # (heads, head_dim): 8x8 unsigned x signed products
        counter.add("attn_pv", mults)
        o = np.concatenate([self._head_avg(a[h], int(e[h].sum()), counter)
                            for h in range(cfg.n_heads)])
        o8, so = quant_vec(o, (lp.sv[0], lp.sv[1] - 8), counter)
        delta = scale_out(lut_matvec(lp.wo.w, o8), lp.wo.m, lp.wo.e, so, self.F, *INT24, counter)
        return clamp(r + delta, *INT24)

    @staticmethod
    def _head_avg(a, z: int, counter) -> np.ndarray:
        """round(a * 256 / z) in Q8 via one reciprocal per head; |result| <= 127 * 256."""
        zb = bitlen(z)  # z >= 255 because the max weight is 255
        a_s = rsr(a, zb - 8)  # |a_s| <= 32512
        rz = udiv((1 << (zb + 14)) + (z >> 1), z)  # in (2**14, 2**15]: uint16
        counter.add("attn_norm", a.size)
        return rsr(check_bits(a_s * rz, 31, "head average"), 14)

    def silu_q8(self, y1: np.ndarray) -> np.ndarray:
        idx = np.clip(rsr(y1, SILU_SHIFT) + SILU_HALF, 0, len(self.silu) - 1)
        return np.where(y1 < -SILU_EDGE, 0, np.where(y1 >= SILU_EDGE, y1, self.silu[idx]))

    def ffn(self, lp: Layer, r, counter) -> np.ndarray:
        x8, sx = norm_quant(r, lp.ffn_g, lp.ffn_eg, counter, self.sqrtd)
        y1 = scale_out(lut_matvec(lp.w1.w, x8), lp.w1.m, lp.w1.e, sx, 8, *INT16, counter)
        acc3 = _ints(lut_matvec(lp.w3.w, x8))
        t3 = rsr_vec(check_bits(acc3 * lp.w3.m, 23, "w3 product"), lp.e3 - lp.w3.e)
        t3n, c3 = normalize16(t3)
        h = check_bits(self.silu_q8(y1) * t3n, 31, "silu * w3")
        counter.add("ffn_t3", t3.size)
        counter.add("ffn_h", t3.size)
        unit = (sx[0] << 7, sx[1] - 7 + lp.e3 + c3 - 8)
        h8, sh8 = quant_vec(h, unit, counter)
        delta = scale_out(lut_matvec(lp.w2.w, h8), lp.w2.m, lp.w2.e, sh8, self.F, *INT24, counter)
        return clamp(r + delta, *INT24)

    def step(self, tok: int, pos: int, cache: KVCache, counter: MulCounter = NULL_COUNTER):
        """Feed one token at position pos; returns int64 logits in Q8 (1/256 nat)."""
        if self.pos_scheme == "learned" and pos >= self.cfg.max_seq_len:
            raise ValueError(f"learned positions stop at {self.cfg.max_seq_len} tokens")
        r = self.embed(tok, pos, counter)
        for i, lp in enumerate(self.layers):
            r = self.attention(i, lp, r, pos, cache, counter)
            r = self.ffn(lp, r, counter)
        x8, sx = norm_quant(r, self.norm_g, self.norm_eg, counter, self.sqrtd)
        out = self.output
        return scale_out(lut_matvec(out.w, x8), out.m, out.e, sx, 8, *INT24, counter)


def expected_mults(cfg: IntConfig, pos: int, horizons: bool = False) -> dict:
    """Per-token multiplies outside the ternary matvecs at position pos (0-based). horizons:
    head h multiplies only the keys inside its ALiBi horizon."""
    d, nl, hid, qd, hd = cfg.dim, cfg.n_layers, cfg.hidden_dim, cfg.n_heads * cfg.head_dim, cfg.head_dim
    norms = 2 * nl + 1
    keys = min(pos + 1, cfg.max_seq_len)
    head_keys = [keys] * cfg.n_heads  # keys each head multiplies
    if horizons:
        if cfg.pos != "alibi":
            raise ValueError("ALiBi horizons need an ALiBi export")
        head_keys = [min(keys, int(h)) for h in
                     alibi_horizons(alibi_exponents(cfg.n_heads), cfg.lut_max)]
    att = nl * hd * sum(head_keys)
    return {"emb": (2 if cfg.pos == "learned" else 1) * d, "norm_square": norms * d,
            "norm_gain": norms * d, "requant": norms * d + nl * (qd + hid),
            "scale_out": 2 * (nl * (qd + 2 * hd + d + hid + d) + cfg.vocab_size),
            "attn_score": att, "attn_pv": att,
            "attn_norm": nl * qd, "ffn_t3": nl * hid, "ffn_h": nl * hid}


def sequence_logits(model: IntModel, tokens) -> np.ndarray:
    """Logits (T, V) for every prefix of tokens, fed one at a time from position 0 (relative
    positions: through the sliding window)."""
    cache = model.new_cache()
    return np.stack([model.step(int(t), p, cache) for p, t in enumerate(tokens)])


# ---- sampling -------------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplerConfig:
    temperature_q8: int = 256  # T * 256, 33 <= value <= 65535 (keeps INV_T in uint16)
    top_k: int = 40  # 1 <= k <= 64
    greedy: bool = False
    seed: int = 1


def ranked_candidates(logits: np.ndarray, k: int, banned=None) -> list[int]:
    """Allowed ids by descending logit, ties broken by ascending id, first k kept. banned: a
    (V,) bool mask of ids the logit masks exclude, like <pad>/<bos>/<unk>."""
    allowed = [v for v in range(len(logits))
               if v not in BANNED and (banned is None or not banned[v])]
    return sorted(allowed, key=lambda v: (-int(logits[v]), v))[:k]


def choose(model: IntModel, logits: np.ndarray, cfg: SamplerConfig, state: int, banned=None):
    """Pick the next token. Returns (token, new_prng_state); greedy does not advance the PRNG.
    banned: ids excluded from the ranking (the candidates are the first top_k allowed ids)."""
    if cfg.greedy:
        return ranked_candidates(logits, 1, banned)[0], state
    if not (33 <= cfg.temperature_q8 <= 65535 and 1 <= cfg.top_k <= 64):
        raise ValueError("temperature_q8 must be in [33, 65535] and top_k in [1, 64]")
    cand = ranked_candidates(logits, cfg.top_k, banned)
    inv_t = udiv(32 << 16, cfg.temperature_q8)  # table steps (1/32 nat) per Q8 logit, << 16
    top = int(logits[cand[0]])
    weights = []
    for v in cand:
        delta = min(top - int(logits[v]), (1 << 15) - 1)
        idx = min(rsr(delta * inv_t, 16), len(model.expw) - 1)
        weights.append(int(model.expw[idx]))
    state = xorshift32(state)
    target = state % sum(weights)
    acc = 0
    for v, w in zip(cand, weights):
        acc += w
        if acc > target:
            return v, state
    raise AssertionError("unreachable: cumulative weight exceeds target")


def generate(model: IntModel, prompt: list[int], max_new: int, cfg: SamplerConfig,
             cap: int | None = None, mask=None) -> list[int]:
    """<bos> + prompt, then sample until <eos>, max_new tokens, or `cap` tokens in all (<bos>
    included). The cap defaults to the context for learned positions (which cannot go past it)
    and to DEFAULT_CAP for relative positions (sliding window). mask: logit masks
    (model.intmask.IntMask, or any object with reset(), step(tok) and banned() -> (V,) bool);
    every fed token is stepped, and banned() is asked before every draw."""
    if not 0 <= cfg.seed < (1 << 32):
        raise ValueError("seed must be in [0, 2**32)")
    learned = model.pos_scheme == "learned"
    if cap is None:
        cap = model.cfg.max_seq_len if learned else DEFAULT_CAP
    if learned and cap > model.cfg.max_seq_len:
        raise ValueError(f"learned positions stop at {model.cfg.max_seq_len} tokens")
    tokens = [BOS, *prompt]
    if len(tokens) >= cap:
        raise ValueError("prompt leaves no room under the cap")
    cache, state = model.new_cache(), cfg.seed if cfg.seed else 2463534242
    if mask is not None:
        mask.reset()
    logits = None
    for pos, tok in enumerate(tokens):
        logits = model.step(tok, pos, cache)
        if mask is not None:
            mask.step(tok)
    for _ in range(max_new):
        if len(tokens) >= cap:
            break
        nxt, state = choose(model, logits, cfg, state, None if mask is None else mask.banned())
        tokens.append(nxt)
        if nxt == EOS:
            break
        if mask is not None:
            mask.step(nxt)
        logits = model.step(nxt, len(tokens) - 1, cache)
    return tokens


def render(model: IntModel, ids: list[int]) -> str:
    """Token ids -> TI-BASIC text via tivars (hex if the bytes do not decode)."""
    from model.tokens import decode_ids, render_text

    body = decode_ids(ids, model.vocab)
    try:
        return render_text(body)
    except Exception as exc:  # generated bytes may not form a valid program
        return f"<undecodable: {exc}> {body.hex().upper()}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Sample TI-BASIC programs with the integer reference")
    ap.add_argument("--export", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-new", type=int, default=DEFAULT_CAP - 1)
    ap.add_argument("--cap", type=int, default=None,
                    help="tokens in all, <bos> included (default: the context for learned "
                         f"positions, {DEFAULT_CAP} for relative positions)")
    ap.add_argument("--masks", action="store_true", help="apply the tracker's logit masks")
    ap.add_argument("--horizons", action="store_true",
                    help="ALiBi horizons (format 2; CalcLM v2.1's setting)")
    args = ap.parse_args(argv)
    model = IntModel.load(args.export, horizons=args.horizons)
    for i in range(args.n):
        cfg = SamplerConfig(round(args.temp * 256), args.top_k, args.greedy, args.seed + i)
        mask = None
        if args.masks:
            from model.intmask import IntMask
            mask = IntMask(model.vocab)
        ids = generate(model, [], args.max_new, cfg, args.cap, mask)
        print(f"--- sample {i}: seed={cfg.seed} Tq={cfg.temperature_q8} top_k={cfg.top_k} "
              f"greedy={cfg.greedy} tokens={len(ids)} first_ids={ids[:10]}")
        print(render(model, ids) if model.vocab else ids)


if __name__ == "__main__":
    main()
