"""Calculator cost of a CalcLM ModelConfig: cycles per token, flash bytes, RAM bytes.

    uv run python -m lab.budget                       # v1 shape
    uv run python -m lab.budget --dim 256 --hidden-dim 640 --n-layers 8

Cycles (TI-84 Plus CE, 48 MHz, CLMCODE flash-resident code, LCD DMA on: the setting of
`ce/build/calclm/clm_bench_flash.json`) are a sum of cost terms. A term is a function
`(cfg, pos) -> {phase: cycles}`; the phases are the profiler phases of `ce/tools/clm_bench.py`.
A new architecture feature adds its own term (`extra_terms=[...]`) or replaces one in
`DEFAULT_TERMS`. Two kinds of coefficients:

- matvec kernels (`build`, `rows`): predicted from the M2 kernel benchmark (ce/BENCH.md, code
  in flash, LCD DMA on) for 192->192, 192->512 and 512->192. Not fitted to v1's phases; they land
  within 1% of them.
- everything else: per-unit costs fitted to v1's measured phases (positions 0/64/128/255, hook
  overhead spread proportionally as clm_bench.report does). Units are chosen to extrapolate:
  per vector element a phase touches, per scaled output row, per vocab id, per (layer, query head,
  key). Per-key attention costs scale with head_dim / 32 (they are 8x8 MACs). Assumptions: per-
  call overheads are folded into the per-element costs; P.V was measured with a random KV cache
  (its compaction depends on how sparse the softmax weights are).

Flash replicates model/export.py's blob layout exactly; only the attention exp-table lengths
are data-dependent (pass them, or get an upper bound). RAM follows ce/calclm/README.md's memory
map and reports what does not fit.

v2 extras (`budget(..., bag_bits=, tracker_cycles=, mask_ids=)`), none for a v1 config:
- feature embeddings (`feat_embed`): one embedding row add per categorical feature and per set
  bag bit, each costed like a token/position row (embed_elem per element: the same int8 x PF8
  dequantize-and-add). The set bag bits per token are data-dependent: pass the mean measured on
  the features, or get the bound (every bit set).
- the tracker step (`tracker`) when features or masks are used: by default the mean measured in
  CEmu (tracker/README.md T3, -Oz: 6.2K cycles/token; p99 12K, max 41.6K); pass another
  value to override. Its state (T1 bound, 512 B; 510 B measured) is counted in BSS.
- logit masks (`mask`): an estimate, not measured: MASK_ID_CYCLES per id the masks can ban
  (test the state, write INT24 min into that logit before the top-k scan).
Flash includes the feature table (feat_emb rows x dim int8 + PF8 row scales).
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model.export import (
    APPVAR_CAPACITY, APPVAR_HEADER, appvar_plan, expw_table, header_len, silu_table,
)
from model.model import ModelConfig

CPU_HZ = 48_000_000
XV_FILE_OVERHEAD = 76  # .8xv: 55-byte file header + 17-byte var entry + 2-byte length + 2 checksum

# ---- matvec kernel benchmark (ce/BENCH.md, "flash (PIC)", LCD DMA on): (n_in, n_out) -> cycles
KERNEL_BENCH = {(192, 192): (186_848, 276_220), (192, 512): (186_848, 726_674),
                (512, 192): (480_736, 712_945)}


def groups(n: int) -> int:
    return -(-n // 5)


def _kernel_fit():
    """build(G) = per_call + per_group * G; rows(n_out, G) = a * n_out * G + b * n_out + c."""
    (b39, r_qq), (_, r_qh), (b103, r_hq) = (KERNEL_BENCH[k] for k in
                                             ((192, 192), (192, 512), (512, 192)))
    per_group = (b103 - b39) / (103 - 39)
    per_build = b39 - 39 * per_group
    a = (r_hq - r_qq) / (192 * (103 - 39))
    per_row39 = (r_qh - r_qq) / (512 - 192)
    b = per_row39 - 39 * a
    c = r_qq - 192 * per_row39
    return per_build, per_group, a, b, c


@dataclass(frozen=True)
class Coeffs:
    """Cycles per unit. Kernel terms from KERNEL_BENCH; the rest fitted to v1 (see module doc)."""

    build_call: float
    build_group: float
    rows_lookup: float      # per (row, group)
    rows_row: float
    rows_call: float
    embed_elem: float = 349.0       # per element of the two embedding rows (2 * dim)
    norm_elem: float = 1878.6       # per residual element of each RMSNorm+quant (2L+1 per token)
    scale_row: float = 681.7        # per scaled matvec output row (all but w3)
    quant_elem: float = 673.5       # per element requantized (attention output, FFN hidden)
    ffn_elem: float = 1065.8        # per hidden element (t3, SiLU, products)
    sample_vocab: float = 982.8     # per vocab id (top-k scan, top-k 20)
    scores_head: float = 14231.1    # per (layer, query head)
    scores_key: float = 1179.9      # per (layer, query head, key) at head_dim 32
    softmax_head: float = 2724.8
    softmax_key: float = 158.3
    pv_head: float = 59722.4        # at head_dim 32 (head average, 8-key groups, compaction)
    pv_key: float = 89.9            # at head_dim 32


COEFFS = Coeffs(*_kernel_fit())


@dataclass(frozen=True)
class Matvec:
    name: str
    n_in: int
    n_out: int
    table: str          # matvecs with the same input share one table build
    scaled: bool = True  # gets a scale_out pass (w3 goes through the FFN t3 path instead)


def matvecs(cfg: ModelConfig) -> list[Matvec]:
    d, hd, kv = cfg.dim, cfg.head_dim, cfg.n_kv_heads
    qd, hid = cfg.n_heads * hd, cfg.hidden_dim
    out = []
    for i in range(cfg.n_layers):
        p = f"layers.{i}"
        out += [Matvec(f"{p}.wq", d, qd, f"{p}.attn_in"), Matvec(f"{p}.wk", d, kv * hd, f"{p}.attn_in"),
                Matvec(f"{p}.wv", d, kv * hd, f"{p}.attn_in"), Matvec(f"{p}.wo", qd, d, f"{p}.attn_out"),
                Matvec(f"{p}.w1", d, hid, f"{p}.ffn_in"), Matvec(f"{p}.w3", d, hid, f"{p}.ffn_in", False),
                Matvec(f"{p}.w2", hid, d, f"{p}.ffn_hidden")]
    return out + [Matvec("output", d, cfg.vocab_size, "final")]


# ---- cycle terms ---------------------------------------------------------------------------

Term = Callable[[ModelConfig, int], dict]


def embed_term(cfg, pos, c=COEFFS):
    rows = 2 if cfg.pos == "learned" else 1     # relative schemes have no position row
    return {"embed": c.embed_elem * rows * cfg.dim}


def norm_term(cfg, pos, c=COEFFS):
    return {"norm": c.norm_elem * (2 * cfg.n_layers + 1) * cfg.dim}


def matvec_term(cfg, pos, c=COEFFS):
    mvs = matvecs(cfg)
    tables = {m.table: groups(m.n_in) for m in mvs}
    build = sum(c.build_call + c.build_group * g for g in tables.values())
    rows = sum(c.rows_lookup * m.n_out * groups(m.n_in) + c.rows_row * m.n_out + c.rows_call
               for m in mvs)
    scale = c.scale_row * sum(m.n_out for m in mvs if m.scaled)
    return {"build": build, "rows": rows, "scale_out": scale}


def keys_at(cfg, pos: int) -> int:
    """Keys attended at position pos: all so far, at most the max_seq_len-entry K/V ring."""
    if cfg.pos == "learned" and pos >= cfg.max_seq_len:
        raise ValueError(f"learned positions stop at {cfg.max_seq_len} tokens")
    return min(pos + 1, cfg.max_seq_len)


def attention_term(cfg, pos, c=COEFFS):
    heads, keys, w = cfg.n_layers * cfg.n_heads, keys_at(cfg, pos), cfg.head_dim / 32
    return {"scores": heads * (c.scores_head + c.scores_key * w * keys),
            "softmax": heads * (c.softmax_head + c.softmax_key * keys),
            "pv": heads * w * (c.pv_head + c.pv_key * keys)}


def quant_term(cfg, pos, c=COEFFS):
    return {"quant": c.quant_elem * cfg.n_layers * (cfg.n_heads * cfg.head_dim + cfg.hidden_dim)}


def ffn_term(cfg, pos, c=COEFFS):
    return {"ffn_act": c.ffn_elem * cfg.n_layers * cfg.hidden_dim}


def sample_term(cfg, pos, c=COEFFS):
    return {"sample": c.sample_vocab * cfg.vocab_size}


ALIBI_KEY_CYCLES = 40.0   # estimate (not measured): one 24-bit shift-and-subtract per key


def alibi_term(cfg, pos):
    if cfg.pos != "alibi":
        return {}
    return {"alibi": ALIBI_KEY_CYCLES * cfg.n_layers * cfg.n_heads * keys_at(cfg, pos)}


V1_MEASURED_AVG_CYCLES = 50_680_000   # lab/RESULTS.md, "Calculator budget accuracy"
TRACKER_CYCLES_BOUND = V1_MEASURED_AVG_CYCLES // 1000   # T3: < 0.1% of v1's per-token cycles
TRACKER_CYCLES_MEASURED = 6200        # tracker/README.md T3: CEmu, -Oz, mean over val tokens
TRACKER_STATE_BYTES = 512             # T1: fixed-size state <= 512 B
MASK_ID_CYCLES = 150.0                # estimate (not measured): state test + 3-byte logit store


def feature_term(bag_bits: float, c=COEFFS) -> Term:
    """Embedding row adds of the feature inputs: one per categorical feature + set bag bits."""
    def term(cfg, pos):
        return {"feat_embed": c.embed_elem * cfg.dim * (len(cfg.feat_cats) + bag_bits)}
    return term


def tracker_term(cycles: float) -> Term:
    return lambda cfg, pos: {"tracker": float(cycles)}


def mask_term(n_ids: int) -> Term:
    return lambda cfg, pos: {"mask": MASK_ID_CYCLES * n_ids}


DEFAULT_TERMS: tuple[Term, ...] = (embed_term, norm_term, matvec_term, attention_term, quant_term,
                                   ffn_term, sample_term, alibi_term)


def cycles_breakdown(cfg, pos: int, extra_terms: Iterable[Term] = (),
                     terms: Iterable[Term] = DEFAULT_TERMS) -> dict[str, float]:
    """Cycles per token at position pos (forward pass + sampling), by phase."""
    out: dict[str, float] = {}
    for term in (*terms, *extra_terms):
        for k, v in term(cfg, pos).items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


def cycles_per_token(cfg, pos: int, extra_terms: Iterable[Term] = ()) -> float:
    return sum(cycles_breakdown(cfg, pos, extra_terms).values())


def program_seconds(cfg, n_tokens: int, extra_terms: Iterable[Term] = ()) -> float:
    """Calculator time to write an n-token program (<bos> included): forward passes and draws
    at positions 0 .. n-2; relative schemes keep a max_seq_len sliding window."""
    extra = tuple(extra_terms)
    return sum(cycles_per_token(cfg, p, extra) for p in range(n_tokens - 1)) / CPU_HZ


def average_cycles(cfg, n_tokens: int | None = None, extra_terms: Iterable[Term] = ()) -> float:
    """Mean cycles/token over a full generation: forward passes at positions 0..n-1."""
    n = cfg.max_seq_len - 1 if n_tokens is None else n_tokens
    extra = tuple(extra_terms)
    return float(np.mean([cycles_per_token(cfg, p, extra) for p in range(n)]))


# ---- flash: the export blob, replicated from model/export.py ------------------------------

def export_tensors(cfg, lut_lens) -> dict[str, np.ndarray]:
    """Empty arrays with the shapes and dtypes model.export.build_tensors writes, in order."""
    d, hd, kv, qd = cfg.dim, cfg.head_dim, cfg.n_kv_heads, cfg.n_heads * cfg.head_dim
    e = lambda shape, dt: np.empty(shape, dtype=dt)  # noqa: E731

    def lin(p, n_in, n_out):
        return {f"{p}.w": e((n_out, groups(n_in)), "u1"), f"{p}.m": e(n_out, "u1"),
                f"{p}.e": e(n_out, "i1")}

    t = {"header": e(header_len(cfg), "<i4")}
    for i in range(cfg.n_layers):
        p = f"layers.{i}"
        t[f"{p}.attn_norm.g"] = e(d, "<i2")
        t[f"{p}.attn.lut"] = e(int(lut_lens[i]), "u1")
        t |= lin(f"{p}.wq", d, qd) | lin(f"{p}.wk", d, kv * hd) | lin(f"{p}.wv", d, kv * hd)
        t |= lin(f"{p}.wo", qd, d)
        t[f"{p}.ffn_norm.g"] = e(d, "<i2")
        t |= lin(f"{p}.w1", d, cfg.hidden_dim) | lin(f"{p}.w3", d, cfg.hidden_dim)
        t |= lin(f"{p}.w2", cfg.hidden_dim, d)
    t["norm.g"] = e(d, "<i2")
    t |= lin("output", d, cfg.vocab_size)
    tables = [("tok_emb", cfg.vocab_size)] + ([("pos_emb", cfg.max_seq_len)]
                                              if cfg.pos == "learned" else [])
    for name, rows in tables:
        t |= {name: e((rows, d), "i1"), f"{name}.m": e(rows, "u1"), f"{name}.e": e(rows, "i1")}
    t["silu"] = e(len(silu_table()), "<i2")
    t["expw"] = e(len(expw_table()), "<u2")
    if cfg.feat_rows:
        rows = cfg.feat_rows
        t |= {"feat_emb": e((rows, d), "i1"), "feat_emb.m": e(rows, "u1"), "feat_emb.e": e(rows, "i1")}
    return t


def _part(name: str) -> str:
    if name == "header":
        return "header"
    if name.endswith(".g"):
        return "norm_gains"
    if name.endswith(".lut"):
        return "attn_luts"
    if name in ("silu", "expw"):
        return "tables"
    if name in ("tok_emb", "pos_emb", "feat_emb"):
        return "embeddings"
    if name.startswith(("tok_emb.", "pos_emb.", "feat_emb.")):
        return "embedding_scales"
    return "ternary_packed" if name.endswith(".w") else "row_scales"


def flash_bytes(cfg, lut_lens=None, appvar_capacity: int = APPVAR_CAPACITY) -> dict:
    """Blob layout of the export, its AppVar split and the .8xv file sizes.

    Without lut_lens each attention exp table counts at its maximum (lut_max), an upper bound.
    """
    exact = lut_lens is not None
    luts = list(lut_lens) if exact else [cfg.lut_max] * cfg.n_layers
    tensors = export_tensors(cfg, luts)
    layout, offset, parts = {}, 0, {}
    for name, arr in tensors.items():
        row_bytes = arr.itemsize * (int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1)
        layout[name] = {"offset": offset, "nbytes": arr.nbytes, "shape": list(arr.shape),
                        "row_bytes": row_bytes}
        offset += arr.nbytes
        parts[_part(name)] = parts.get(_part(name), 0) + arr.nbytes
    plan = appvar_plan(tensors, layout, appvar_capacity)
    payload = sum(a["payload_bytes"] for a in plan)
    return {"exact": exact, "blob_bytes": offset, "parts": parts, "n_appvars": len(plan),
            "appvar_payload_bytes": payload + APPVAR_HEADER * len(plan),
            "appvar_file_bytes": payload + (APPVAR_HEADER + XV_FILE_OVERHEAD) * len(plan),
            "lut_lens": luts}


def flash_bytes_from_export(export_dir) -> dict:
    """flash_bytes with the exact exp-table lengths and AppVar capacity of an export."""
    meta = json.loads((Path(export_dir) / "model.json").read_text())
    cfg = ModelConfig(**meta["config"])
    luts = [meta["tensors"][f"layers.{i}.attn.lut"]["shape"][0] for i in range(cfg.n_layers)]
    return flash_bytes(cfg, luts, meta["appvar_capacity"])


# ---- RAM: ce/calclm/README.md memory map ----------------------------------------------------

BSS_LIMIT = 60_690          # toolchain BSS region (OS scratch RAM)
BSS_OTHER = 57_081 - 53_760  # core state + font next to v1's K cache (approximate for others)
APPVAR_LIMIT = 65_505       # the V cache is one RAM AppVar (CLMTMP)
PROGRAM_BYTES = 39_482      # CALCLM.8xp, copied to user RAM when it runs
RAM_MARGIN = 4096 + 64      # ce/calclm/src/model.c
USER_RAM_FREE = 150_000     # nominal free user RAM (SPEC.md: ~150 KB)
ACT_LIMIT = 0xD52A00 - 0xD4F300   # VRAM gap for activations
TABLE_PAGES = 243
MAX_GROUPS = 127            # columns 0..2G of a 256-byte page; also the kernel's n_in <= 635
VOCAB_RANGE = (68, 1024)    # CLM_E_SHAPE / CLM_VOCAB_MAX
MAX_APPVARS = 16            # CLM_MAX_APPVARS
MAX_TOPK = 64


def act_bytes(cfg) -> int:
    """CLM_ACT_BYTES (ce/calclm/core/clm.h) for this shape."""
    d, hid, ctx, vm = cfg.dim, cfg.hidden_dim, cfg.max_seq_len, VOCAB_RANGE[1]
    qd = cfg.n_heads * cfg.head_dim
    return (4 * d + 5 * groups(max(d, qd)) + 5 * groups(hid) + 3 * (max(vm, hid) + ctx + vm)
            + qd + 4 * hid + 2 * hid + 3 + ctx + 8 + 2 * ctx + 4 * MAX_TOPK)


def ram_bytes(cfg, tracker_bytes: int = 0) -> dict:
    """tracker_bytes: tracker state kept in BSS next to the core state (v2)."""
    L, ctx, hd, kv = cfg.n_layers, cfg.max_seq_len, cfg.head_dim, cfg.n_kv_heads
    k = L * ctx * (hd + 3) * kv
    v = L * ctx * hd * kv
    gmax = max(groups(m.n_in) for m in matvecs(cfg))
    bank = (TABLE_PAGES - 1) * 256 + 2 * gmax + 2
    act = act_bytes(cfg)
    bss = k + BSS_OTHER + tracker_bytes
    user = PROGRAM_BYTES + v + RAM_MARGIN
    problems = []
    if bss > BSS_LIMIT:
        problems.append(f"k_cache: K cache {k} + core state {BSS_OTHER} + tracker {tracker_bytes}"
                        f" = {bss} B > BSS {BSS_LIMIT} B")
    if v > APPVAR_LIMIT:
        problems.append(f"v_cache: {v} B > one AppVar ({APPVAR_LIMIT} B)")
    if user > USER_RAM_FREE:
        problems.append(f"user_ram: program + V cache + margin = {user} B > {USER_RAM_FREE} B")
    if gmax > MAX_GROUPS:
        problems.append(f"table_bank: a matvec input needs {gmax} groups > {MAX_GROUPS} "
                        f"(n_in <= {5 * MAX_GROUPS})")
    if act > ACT_LIMIT:
        problems.append(f"activations: {act} B > VRAM gap {ACT_LIMIT} B")
    if not VOCAB_RANGE[0] <= cfg.vocab_size <= VOCAB_RANGE[1]:
        problems.append(f"vocab: {cfg.vocab_size} outside {VOCAB_RANGE[0]}..{VOCAB_RANGE[1]}")
    if ctx % 8:
        problems.append(f"ctx: {ctx} is not a multiple of 8")
    if kv != 1:
        problems.append("n_kv_heads: intref and the core implement one KV head (MQA) only")
    if hd > 32:
        problems.append(f"head_dim: {hd} > 32 (the core's P.V compaction buffer is 96 keys x 32 B)")
    if cfg.n_heads * hd > cfg.hidden_dim:
        problems.append(f"hidden: n_heads * head_dim {cfg.n_heads * hd} > hidden_dim "
                        f"{cfg.hidden_dim} (the core's v32 buffer is hidden-sized)")
    return {"k_cache": k, "v_cache": v, "activations": act, "table_bank": bank,
            **({"tracker": tracker_bytes} if tracker_bytes else {}),
            "total": k + v + act + bank + tracker_bytes, "bss": bss, "user_ram": user,
            "fits": not problems, "problems": problems}


# ---- the report -------------------------------------------------------------------------------

def report_positions(cfg) -> tuple[int, ...]:
    return tuple(sorted({0, cfg.max_seq_len // 2, cfg.max_seq_len - 1}))


def v2_extras(cfg, bag_bits=None, tracker_cycles=None, mask_ids: int = 0):
    """(cost terms, description) of the feature inputs, the tracker and the masks."""
    terms, desc = [], {}
    if cfg.feat_rows:
        bits = cfg.feat_bag if bag_bits is None else bag_bits
        desc["bag_bits"] = {"per_token": bits, "source": "bound" if bag_bits is None else "given"}
        terms.append(feature_term(bits))
    if cfg.feat_rows or mask_ids:
        cyc = TRACKER_CYCLES_MEASURED if tracker_cycles is None else tracker_cycles
        desc["tracker_cycles"] = {"per_token": cyc, "source": "measured (tracker/README.md T3, -Oz "
                                  "mean)" if tracker_cycles is None else "given"}
        desc["tracker_state_bytes"] = TRACKER_STATE_BYTES
        terms.append(tracker_term(cyc))
    if mask_ids:
        desc["mask_ids"] = mask_ids
        desc["mask_id_cycles"] = {"per_id": MASK_ID_CYCLES, "source": "estimate"}
        terms.append(mask_term(mask_ids))
    return terms, desc


def budget(cfg, lut_lens=None, extra_terms: Iterable[Term] = (), flash: dict | None = None,
           bag_bits: float | None = None, tracker_cycles: float | None = None,
           mask_ids: int = 0, cap: int | None = None) -> dict:
    """Everything above for one config (JSON-serializable). `flash` overrides the flash
    estimate, e.g. with flash_bytes_from_export(...) of the trained model. bag_bits,
    tracker_cycles and mask_ids cost the v2 extras (module doc)."""
    v2_terms, extras = v2_extras(cfg, bag_bits, tracker_cycles, mask_ids)
    extra = (*extra_terms, *v2_terms)
    pos = report_positions(cfg)
    phases = {p: cycles_breakdown(cfg, p, extra) for p in pos}
    cycles = {p: sum(v.values()) for p, v in phases.items()}
    avg = average_cycles(cfg, extra_terms=extra)
    cycles["avg"] = avg
    fl = flash or flash_bytes(cfg, lut_lens)
    if fl["n_appvars"] > MAX_APPVARS:
        fl = {**fl, "problems": [f"appvars: {fl['n_appvars']} > {MAX_APPVARS}"]}
    ram = ram_bytes(cfg, TRACKER_STATE_BYTES if "tracker_cycles" in extras else 0)
    mvs = matvecs(cfg)
    return {
        "config": cfg.to_dict(),
        "params": {"ternary": sum(m.n_in * m.n_out for m in mvs),
                   "embedding": (cfg.vocab_size + cfg.feat_rows
                                 + (cfg.max_seq_len if cfg.pos == "learned" else 0)) * cfg.dim},
        "cycles": cycles,
        "tokens_per_s": {p: CPU_HZ / c for p, c in cycles.items()},
        "seconds_full_program": avg * (cfg.max_seq_len - 1) / CPU_HZ,
        "phases": phases,
        "flash": fl,
        "ram": ram,
        **({"extras": extras} if extras else {}),
        **({"seconds_cap_program": (sec := program_seconds(cfg, cap, extra)),
            "cap": {"tokens": cap, "seconds": sec, "tokens_per_s": (cap - 1) / sec}}
           if cap else {}),
        "fits": ram["fits"] and not fl.get("problems"),
    }


def main(argv=None) -> None:
    v1 = ModelConfig(vocab_size=781)
    ap = argparse.ArgumentParser(description="Calculator cost of a CalcLM shape")
    ap.add_argument("--export", type=Path, help="read the shape and exact flash from an export")
    for f in ("vocab_size", "dim", "n_layers", "n_heads", "hidden_dim", "max_seq_len"):
        ap.add_argument("--" + f.replace("_", "-"), type=int, default=getattr(v1, f))
    a = ap.parse_args(argv)
    if a.export:
        fl = flash_bytes_from_export(a.export)
        meta = json.loads((a.export / "model.json").read_text())
        rep = budget(ModelConfig(**meta["config"]), flash=fl)
    else:
        rep = budget(ModelConfig(vocab_size=a.vocab_size, dim=a.dim, n_layers=a.n_layers,
                                 n_heads=a.n_heads, hidden_dim=a.hidden_dim, max_seq_len=a.max_seq_len))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
