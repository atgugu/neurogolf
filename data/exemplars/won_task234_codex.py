#!/usr/bin/env python3
"""task234 encoded-profile + prefix-valid terminal (plane-kill family).

True rule: move the tongue-color rectangle along the tongue axis until flush
with the solid rectangle; erase the tongue gap.

Attempt 10: cost=2240 (params=95, mem=2145), full-gate pass.
"""

from __future__ import annotations

import build_optimized


def main() -> None:
    import onnx

    model = build_optimized.build()
    onnx.save(model, "task234.onnx")
    print(f"saved task234.onnx nodes={len(model.graph.node)} inits={len(model.graph.initializer)}")


if __name__ == "__main__":
    main()
