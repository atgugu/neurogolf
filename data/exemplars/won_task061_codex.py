#!/usr/bin/env python3
"""Build task061 with a universal bitset one-hot terminal.

The audited family renders the one-hot answer with a final broadcast
BitwiseAnd.  This pack's public graph input is one-hot [1,10,30,30], so P is
recovered from the highest present input channel before selecting the row mask.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runner"))
from ngolf import G  # noqa: E402


M = 18
MIN_P = 4
MAX_P = 9
K = 10


def bit_id(p: int, residue: int) -> int:
    return sum(range(MIN_P, p)) + residue


def make_masks() -> tuple[np.ndarray, np.ndarray]:
    rowmask_table = np.zeros((MAX_P - MIN_P + 1, 1, 30, 1), dtype=np.int64)
    target_mask = np.zeros((1, K, 1, 30), dtype=np.int64)

    for p in range(MIN_P, MAX_P + 1):
        for r in range(M):
            rowmask_table[p - MIN_P, 0, r, 0] = np.int64(1 << bit_id(p, r % p))

    for ch in range(1, K):
        for c in range(M):
            mask = 0
            for p in range(MIN_P, MAX_P + 1):
                col_residue = c % p
                for row_residue in range(p):
                    value = (row_residue * col_residue) % p + 1
                    if value == ch:
                        mask |= 1 << bit_id(p, row_residue)
            target_mask[0, ch, 0, c] = np.int64(mask)

    return rowmask_table, target_mask


def build(path: Path) -> None:
    g = G(task=61)

    # One-hot input: mark which color channels occur, then take the highest
    # present channel as the source period P.
    axes_hw = g.init(np.array([2, 3], dtype=np.int64))
    present = g.n("ReduceMax", [g.input, axes_hw], [1, 10], "f32", keepdims=0)
    p_i64 = g.n(
        "ArgMax",
        [present],
        [1],
        "i64",
        axis=1,
        keepdims=0,
        select_last_index=1,
    )
    p_idx = g.sub(p_i64, g.init(np.array([MIN_P], dtype=np.int64)))

    rowmask_table, target_mask = make_masks()
    row_table = g.init(rowmask_table)
    row_mask = g.n("Gather", [row_table, p_idx], [1, 1, 30, 1], "i64", axis=0)
    target = g.init(target_mask)
    g.n("BitwiseAnd", [row_mask, target], [1, 10, 30, 30], "i64", is_output=True)

    g.save(str(path))


def main() -> None:
    build(Path(__file__).with_name("task061.onnx"))


if __name__ == "__main__":
    main()
