# VERIFIED by hypo.py + 1 stress counterexample(s), repaired by stress_rules 2026-07-12
import numpy as np

def rule(grid):
    g = np.asarray(grid)
    h, w = g.shape
    foreground = [int(g[r, c]) for r in range(h) for c in range(w) if g[r, c] != 0]
    if not foreground:
        return g.copy()
    rare = min(set(foreground), key=foreground.count)
    rr, cc = np.where(g == rare)
    return g[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1]
