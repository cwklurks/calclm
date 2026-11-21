"""Export a trained checkpoint to the integer artifacts that intref.py (and the calculator) run.

Output (out_dir): model.bin (one little-endian blob, tensors back to back, no padding) and
model.json (offsets, shapes, dtypes, scalars, the AppVar split plan, sha256). The blob starts
with an int32 `header` tensor (magic "CLM0", dims and per-layer scalars, see HEADER_FIELDS and
LAYER_FIELDS; format 2, a relative-position model, adds V2_FIELDS after the per-layer fields) so
the calculator never needs the JSON. Floats are used here only.

    uv run python -m model.export --ckpt runs/v0/best.pt --data data/tok --out runs/v0/export
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from model.data import load_split, val_windows
from model.intops import sqrt_pf15
from model.pack import pack_ternary

if TYPE_CHECKING:
    from model.model import Transformer

# torch (directly or via model.model / model.quant) is imported only where an export needs it:
# intref and the sampling and NLL workers import this module for load_export and the
# tables, and importing torch makes each worker ~500 MB larger (529 vs 35 MB measured).

APPVAR_LIMIT = 65505  # max AppVar data size on the CE
APPVAR_HEADER = 7  # b"CL", index byte, count byte, blob_start (3 bytes little-endian)
APPVAR_CAPACITY = APPVAR_LIMIT - APPVAR_HEADER
MAGIC = int.from_bytes(b"CLM0", "little")
VERSION = 1  # learned positions: the v1 format, unchanged
VERSION_RELATIVE = 2  # relative positions (INTREF.md 12): no pos_emb, V2_FIELDS after the layers
HEADER_FIELDS = ["magic", "version", "dim", "n_layers", "n_heads", "head_dim", "hidden", "vocab",
                 "ctx", "F", "silu_len", "expw_len", "norm_eg", "lut_max", "sqrtd_m", "sqrtd_e"]
LAYER_FIELDS = ["attn_eg", "ffn_eg", "attn_sh", "lut_len", "sv_m", "sv_e"]
V2_FIELDS = ["pos_scheme"]
POS_SCHEME_CODES = {"alibi": 1, "nope": 2}
SILU_STEP_BITS = 6  # table step 1/64 over [-8, 8]
SILU_HALF = 8 << SILU_STEP_BITS  # 512
EXPW_STEP = 32  # sampling table: 1/32 nat per entry


# ---- tensor builders ---------------------------------------------------------------------

def narrow(values, dtype: str) -> np.ndarray:
    """Cast integer-valued data to a blob dtype, refusing anything that would wrap."""
    arr = np.asarray(values)
    info = np.iinfo(np.dtype(dtype))
    if arr.size and (arr.min() < info.min or arr.max() > info.max or (arr != np.round(arr)).any()):
        raise ValueError(f"values out of {dtype} range: [{arr.min()}, {arr.max()}]")
    return arr.astype(dtype)


def norm_gain(norm) -> tuple[np.ndarray, int]:
    """Gain as int16 g16 with g = g16 * 2**-eg, max |g16| in [2**14, 2**15)."""
    g = norm.weight.detach().double().cpu().numpy()
    eg = 15 - math.frexp(float(np.abs(g).max()))[1]
    g16 = np.floor(g * 2.0**eg + 0.5)
    if np.abs(g16).max() >= 2**15:
        eg -= 1
        g16 = np.floor(g * 2.0**eg + 0.5)
    return narrow(g16, "<i2"), eg


def linear_tensors(prefix: str, lin, post=None) -> dict:
    from model.quant import weight_quant
    wt, bm, be = weight_quant(lin.weight.detach().cpu(), post)
    return {f"{prefix}.w": pack_ternary(wt.numpy().astype(np.int8)),
            f"{prefix}.m": narrow(bm.numpy(), "u1"), f"{prefix}.e": narrow(be.numpy(), "i1")}


def embedding_tensors(prefix: str, emb) -> dict:
    import torch

    from model.quant import pf_round
    with torch.no_grad():
        w8, s = emb.quantized()
    m, e = pf_round(s.double().squeeze(1), 8)
    return {prefix: narrow(w8.numpy(), "i1"), f"{prefix}.m": narrow(m.numpy(), "u1"),
            f"{prefix}.e": narrow(e.numpy(), "i1")}


def silu_table() -> np.ndarray:
    x = (np.arange(2 * SILU_HALF + 1) - SILU_HALF) / 2.0**SILU_STEP_BITS
    return np.floor(256.0 * x / (1.0 + np.exp(-x)) + 0.5).astype("<i2")


def expw_table() -> np.ndarray:
    vals = []
    while not vals or vals[-1] > 0:
        vals.append(math.floor(65535.0 * math.exp(-len(vals) / EXPW_STEP) + 0.5))
    return np.array(vals, dtype="<u2")


def attention_params(attn) -> tuple[dict, list[int]]:
    """Exp table, its index shift, and the PF15 value scale of one layer."""
    from model.quant import attn_lut_shift, exp_lut_values, pf_round
    s = attn.scales().detach().float().cpu()
    if not (s > 0).all():
        raise ValueError("static q/k/v scales must be positive (was the model calibrated?)")
    u = s[0] * s[1] / math.sqrt(attn.cfg.head_dim)
    sh = int(attn_lut_shift(u, attn.cfg.lut_max))
    lut = exp_lut_values(float(u * 2.0**sh), attn.cfg.lut_max)
    sv_m, sv_e = pf_round(s[2].double(), 15)
    return {"lut": narrow(lut, "u1")}, [sh, len(lut), int(sv_m), int(sv_e)]


def _calib_feature_args(cfg, calib_feats) -> dict:
    """cats / bag tensors for a feature model's calibration windows (none without features)."""
    if not cfg.feat_rows:
        return {}
    if calib_feats is None:
        raise ValueError("a model with feature inputs needs calib_feats=(cats, bag) aligned "
                         "with the calibration windows")
    import torch
    cats, bag = calib_feats
    return {"cats": torch.as_tensor(np.asarray(cats), dtype=torch.long) if cfg.feat_cats else None,
            "bag": torch.as_tensor(np.asarray(bag), dtype=torch.long) if cfg.feat_bag else None}


def residual_peak(model: Transformer, calib_tokens, calib_feats=None) -> float:
    """Largest |residual| (embeddings and every block output) on calibration windows."""
    import torch
    ref = copy.deepcopy(model).double().cpu().eval()
    feats = _calib_feature_args(ref.cfg, calib_feats)
    peaks = []
    hooks = [ref.layers[0].register_forward_pre_hook(lambda m, a: peaks.append(a[0].abs().max()))]
    hooks += [blk.register_forward_hook(lambda m, a, out: peaks.append(out.abs().max()))
              for blk in ref.layers]
    try:
        with torch.no_grad():
            ref(torch.as_tensor(np.asarray(calib_tokens), dtype=torch.long), **feats)
    finally:
        for h in hooks:
            h.remove()
    return float(max(peaks))


def residual_frac_bits(peak: float) -> int:
    """Fractional bits of the int24 residual: 4x headroom over the calibration peak."""
    return int(min(22, max(0, math.floor(math.log2((2**23 - 1) / (4 * peak))))))


# ---- blob assembly ---------------------------------------------------------------------

def format_version(cfg) -> int:
    return VERSION if cfg.pos == "learned" else VERSION_RELATIVE


def header_len(cfg) -> int:
    """int32 fields of the header tensor: globals, per-layer fields, then format 2's tail."""
    tail = len(V2_FIELDS) if format_version(cfg) == VERSION_RELATIVE else 0
    return len(HEADER_FIELDS) + len(LAYER_FIELDS) * cfg.n_layers + tail


def build_tensors(model: Transformer, frac_bits: int) -> dict:
    cfg = model.cfg
    layer_rows, layer_tensors = [], {}
    for i, blk in enumerate(model.layers):
        p = f"layers.{i}"
        g_attn, eg_attn = norm_gain(blk.attn_norm)
        g_ffn, eg_ffn = norm_gain(blk.ffn_norm)
        att, att_scalars = attention_params(blk.attn)
        post = 1.0 / blk.attn.scales().detach().float().cpu()
        layer_tensors.update({f"{p}.attn_norm.g": g_attn, f"{p}.attn.lut": att["lut"]})
        for j, name in enumerate(("wq", "wk", "wv")):
            layer_tensors.update(linear_tensors(f"{p}.{name}", getattr(blk.attn, name), post[j]))
        layer_tensors.update(linear_tensors(f"{p}.wo", blk.attn.wo))
        layer_tensors[f"{p}.ffn_norm.g"] = g_ffn
        for name in ("w1", "w3", "w2"):
            layer_tensors.update(linear_tensors(f"{p}.{name}", getattr(blk.ffn, name)))
        layer_rows += [eg_attn, eg_ffn, *att_scalars]
    g_out, eg_out = norm_gain(model.norm)
    silu, expw = silu_table(), expw_table()
    header = [MAGIC, format_version(cfg), cfg.dim, cfg.n_layers, cfg.n_heads, cfg.head_dim,
              cfg.hidden_dim, cfg.vocab_size, cfg.max_seq_len, frac_bits, len(silu), len(expw),
              eg_out, cfg.lut_max, *sqrt_pf15(cfg.dim), *layer_rows]
    if format_version(cfg) == VERSION_RELATIVE:
        header.append(POS_SCHEME_CODES[cfg.pos])
    tensors = {"header": np.array(header, dtype="<i4"), **layer_tensors, "norm.g": g_out}
    tensors.update(linear_tensors("output", model.output))
    tensors.update(embedding_tensors("tok_emb", model.tok_emb))
    if cfg.pos == "learned":
        tensors.update(embedding_tensors("pos_emb", model.pos_emb))
    tensors.update({"silu": silu, "expw": expw})
    if cfg.feat_rows:  # v2 extra inputs: after every v1 tensor, so v1 offsets do not move
        tensors.update(embedding_tensors("feat_emb", model.feat_emb))
    return tensors


def feature_meta(cfg) -> dict:
    """Row layout of feat_emb: categorical feature g, value v -> row_offsets[g] + v;
    bag bit b -> bag_row0 + b."""
    return {"cats": list(cfg.feat_cats), "bag": cfg.feat_bag, "row_offsets": list(cfg.feat_offsets),
            "bag_row0": sum(cfg.feat_cats), "rows": cfg.feat_rows}


def appvar_plan(tensors: dict, layout: dict, capacity: int) -> list[dict]:
    """Cut the blob into AppVar payloads at row boundaries; a tensor that fits in one AppVar
    is never split, larger ones (embeddings) are split between rows."""
    plans, cur, used = [], [], 0
    for name, arr in tensors.items():
        info = layout[name]
        rows, row_bytes = info["shape"][0] if arr.ndim else 1, info["row_bytes"]
        if info["nbytes"] > capacity - used and info["nbytes"] <= capacity and cur:
            plans.append(cur)
            cur, used = [], 0
        start = 0
        while start < rows:
            fit = min(rows - start, (capacity - used) // row_bytes)
            if fit == 0:
                plans.append(cur)
                cur, used = [], 0
                continue
            cur.append({"tensor": name, "row_start": start, "row_end": start + fit})
            used += fit * row_bytes
            start += fit
    plans.append(cur)
    out = []
    for i, parts in enumerate(p for p in plans if p):
        lo = layout[parts[0]["tensor"]]["offset"] + parts[0]["row_start"] * layout[parts[0]["tensor"]]["row_bytes"]
        last = parts[-1]
        hi = layout[last["tensor"]]["offset"] + last["row_end"] * layout[last["tensor"]]["row_bytes"]
        out.append({"name": f"CLM{i:02d}", "blob_start": lo, "blob_end": hi,
                    "payload_bytes": hi - lo, "parts": parts})
    return out


def export_model(model: Transformer, calib_tokens, out_dir, appvar_capacity=APPVAR_CAPACITY,
                 vocab=None, calib_feats=None) -> dict:
    """calib_feats: (cats, bag) aligned with calib_tokens, required for a feature model."""
    if model.cfg.feat_fold:
        raise ValueError("a feat_fold model exports only folded for one tag "
                         "(model.model.fold_feature)")
    model = copy.deepcopy(model).float().cpu().eval()
    _calib_feature_args(model.cfg, calib_feats)
    peak = residual_peak(model, calib_tokens, calib_feats)
    frac_bits = residual_frac_bits(peak)
    tensors = build_tensors(model, frac_bits)
    layout, chunks, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        row_bytes = arr.itemsize * (int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1)
        layout[name] = {"offset": offset, "nbytes": len(raw), "dtype": arr.dtype.str,
                        "shape": list(arr.shape), "row_bytes": row_bytes}
        chunks.append(raw)
        offset += len(raw)
    blob = b"".join(chunks)
    cfg = model.cfg
    meta = {"format": "calclm-int", "version": format_version(cfg), "config": cfg.to_dict(),
            "header_fields": HEADER_FIELDS, "layer_fields": LAYER_FIELDS,
            "scalars": {"F": frac_bits, "residual_peak": peak, "silu_step_bits": SILU_STEP_BITS,
                        "expw_step": EXPW_STEP},
            "blob_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
            "tensors": layout, "appvars": appvar_plan(tensors, layout, appvar_capacity),
            "appvar_capacity": appvar_capacity,
            "counts": {"ternary_weights": model.ternary_param_count(),
                       "embedding_params": model.embedding_param_count()}}
    if cfg.feat_rows:
        meta["features"] = feature_meta(cfg)
    if cfg.pos != "learned":   # relative positions, sliding window of ctx (INTREF.md 12)
        meta["tail_fields"] = V2_FIELDS          # header fields after the per-layer ones
        meta["positions"] = {"scheme": cfg.pos, "window": cfg.max_seq_len}
        if cfg.pos == "alibi":
            meta["positions"]["alibi_exponents"] = model.layers[0].attn.alibi.tolist()
    if vocab is not None:
        meta["vocab"] = list(vocab)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.bin").write_bytes(blob)
    (out / "model.json").write_text(json.dumps(meta, indent=1))
    return json.loads((out / "model.json").read_text())


def write_appvars(out_dir) -> list[Path]:
    """Write one CLMnn.8xv per planned AppVar: b"CL", index, count, blob_start (u24 LE), payload."""
    from tivars.types import TIAppVar

    out = Path(out_dir)
    meta = json.loads((out / "model.json").read_text())
    blob = (out / "model.bin").read_bytes()
    paths, n = [], len(meta["appvars"])
    for i, av in enumerate(meta["appvars"]):
        var = TIAppVar(name=av["name"])
        head = b"CL" + bytes([i, n]) + av["blob_start"].to_bytes(3, "little")
        var.data = head + blob[av["blob_start"] : av["blob_end"]]
        path = out / f"{av['name']}.8xv"
        var.save(str(path))
        paths.append(path)
    return paths


@dataclass
class Export:
    meta: dict
    blob: bytes

    def t(self, name: str) -> np.ndarray:
        info = self.meta["tensors"][name]
        arr = np.frombuffer(self.blob, dtype=np.dtype(info["dtype"]), count=int(np.prod(info["shape"])),
                            offset=info["offset"])
        return arr.reshape(info["shape"])

    def header(self) -> dict:
        h = self.t("header").astype(np.int64).tolist()
        nf, nl = len(HEADER_FIELDS), len(LAYER_FIELDS)
        out = dict(zip(HEADER_FIELDS, h[:nf]))
        out["layers"] = [dict(zip(LAYER_FIELDS, h[nf + i * nl : nf + (i + 1) * nl]))
                         for i in range(out["n_layers"])]
        if out["version"] == VERSION_RELATIVE:     # the header in the blob says which format
            out.update(zip(V2_FIELDS, h[nf + nl * out["n_layers"]:]))
        return out


def load_export(out_dir) -> Export:
    out = Path(out_dir)
    meta = json.loads((out / "model.json").read_text())
    blob = (out / "model.bin").read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        raise ValueError(f"{out}/model.bin does not match model.json sha256")
    return Export(meta, blob)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Export a CalcLM checkpoint to integer artifacts")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dir with val.bin (calibration)")
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--appvars", action="store_true", help="also write CLMnn.8xv files")
    args = ap.parse_args(argv)
    from model.model import load_checkpoint
    model = load_checkpoint(args.ckpt)
    xs, _ = val_windows(load_split(args.data / "val.bin"), model.cfg.max_seq_len)
    vocab = json.loads(args.vocab.read_text()) if args.vocab else None
    meta = export_model(model, xs[: args.calib_windows], args.out, vocab=vocab)
    if args.appvars:
        write_appvars(args.out)
    print(json.dumps({k: meta[k] for k in ("blob_bytes", "sha256", "scalars", "counts")}, indent=2))
    print(f"{len(meta['appvars'])} AppVars:",
          ", ".join(f"{a['name']}={a['payload_bytes']}" for a in meta["appvars"]))


if __name__ == "__main__":
    main()
