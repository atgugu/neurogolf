# Vendored grader — provenance & license

This directory is the **compression scorer** — the Kaggle grader for The 2026 NeuroGolf
Championship, vendored so the repo can price ONNX graphs without an external checkout.

## `extracted/neurogolf_utils/neurogolf_utils.py` — third-party, Apache-2.0

This is the competition's **own scoring framework**, released for the
[IJCAI-ECAI 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026):

> Copyright 2026 Google LLC — Licensed under the Apache License, Version 2.0.

It is redistributed here **verbatim, unmodified**, with its original license header intact.
Pricing runs this file's `check_network` / `score_network` directly — so the cost this repo
reports is not a re-implementation of the grader, it *is* the grader. Full license text:
<https://www.apache.org/licenses/LICENSE-2.0>.

## `neurogolf/{scoring_v2,paths,report,__init__}.py` — this project (MIT)

The thin harness that runs the framework above under an ONNX Runtime profiler. Part of this
repository, MIT-licensed (see `../../LICENSE`). Two minimal changes vs the operator's copy:
`scoring_v2.py` stubs the framework's unused `IPython`/`matplotlib`/`onnx_tool` module-level
imports so pricing needs only numpy + onnx + onnxruntime, and drops a debug print.

## Parity

Memory is read from the ONNX Runtime profiler, which is runtime-version-sensitive, so
**`onnxruntime==1.24.4` reproduces the Kaggle score exactly**; other versions price `params`
identically and `memory` approximately. Correctness *gating* (not pricing) additionally needs
the ARC-GEN generators and the task-grid corpus — see `docs/SETUP.md`.
