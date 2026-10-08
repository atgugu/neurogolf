#!/usr/bin/env python3
"""Build task277 via the registerable tagged-relower morphology graph."""

from build_relower_morph import build


if __name__ == "__main__":
    out = build()
    print(f"saved {out}")
