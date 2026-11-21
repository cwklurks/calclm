"""Cut programs to the calculator's context at a line boundary, for fair comparisons.

The on-calc model stops at 256 tokens, usually mid-line. Generated programs and the
human baseline get the same treatment: keep only whole lines within the limit.

  uv run python -m eval.truncate --gen runs/gen/v0.jsonl --out runs/gen/v0-trim.jsonl
  uv run python -m eval.truncate --bin data/tok/val.bin --out runs/gen/val-trim.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOS, EOS = 1, 2


def trim_to_line(ids: list[int], max_len: int, newline_id: int, eos_id: int) -> list[int]:
    """Return ids unchanged if they end in <eos> within max_len, else cut after the last
    complete line (dropping that trailing newline, which ends the program the same way)."""
    if eos_id in ids[:max_len]:
        return ids[: ids.index(eos_id) + 1]
    head = ids[:max_len]
    last_nl = max((i for i, t in enumerate(head) if t == newline_id), default=None)
    return head[:last_nl] if last_nl is not None else head[:1]


def split_programs(stream: np.ndarray) -> list[list[int]]:
    progs, cur = [], None
    for t in stream.tolist():
        if t == BOS:
            cur = [t]
        elif cur is not None:
            cur.append(t)
            if t == EOS:
                progs.append(cur)
                cur = None
    return progs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gen", type=Path, help="jsonl from eval.gen_intref")
    src.add_argument("--bin", type=Path, help="uint16 token stream, e.g. data/tok/val.bin")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vocab", type=Path, default=Path("corpus/vocab.json"))
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args(argv)
    nl = json.loads(a.vocab.read_text()).index("3F")
    if a.gen:
        seqs = [json.loads(line)["ids"] for line in a.gen.read_text().splitlines()]
    else:
        seqs = split_programs(np.fromfile(a.bin, dtype="<u2"))
    trimmed = [trim_to_line(s, a.max_len, nl, EOS) for s in seqs]
    kept = [s for s in trimmed if len(s) > 1]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps({"ids": s}) + "\n" for s in kept))
    print(f"{a.out}: {len(kept)} of {len(seqs)} programs non-empty after trimming")


if __name__ == "__main__":
    main()
