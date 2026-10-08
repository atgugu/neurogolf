#!/usr/bin/env python3
"""Attempt 7 probe: certified histogram with the requested direct color-code Where tail."""
import copy
import onnx
from onnx import TensorProto, helper

SRC = "probe.onnx"
OUT = "task239_direct_where_probe.onnx"

m = onnx.load(SRC)
g = m.graph

# Retain the certified histogram/mask body through `code`.  The requested renderer
# unshifts only the filled branch, masks padding to blank=11, and exposes the true
# inferred tensor shape so the gate can diagnose its output-contract incompatibility.
keep = []
for node in g.node:
    if node.output and node.output[0] == "code":
        keep.append(copy.deepcopy(node))
        break
    keep.append(copy.deepcopy(node))

# `code` already contains shifted colors in filled cells and 11 in padding.  Reuse
# the certified bar and active-column predicates to form active-and-fill.
keep.extend([
    helper.make_node("And", ["bar", "active_col"], ["fill"]),
    helper.make_node("Sub", ["code", "one"], ["unshifted"]),
    helper.make_node("Where", ["fill", "unshifted", "blank"], ["output"]),
])

del g.node[:]
g.node.extend(keep)
del g.output[:]
g.output.extend([helper.make_tensor_value_info(
    "output", TensorProto.UINT8, [1, 1, 12, 5]
)])

# Remove only the q^2/parabola renderer parameters.
used = {name for n in g.node for name in n.input}
kept_init = [copy.deepcopy(x) for x in g.initializer if x.name in used]
del g.initializer[:]
g.initializer.extend(kept_init)

onnx.checker.check_model(m, full_check=True)
onnx.shape_inference.infer_shapes(m, strict_mode=True)
onnx.save(m, OUT)
print(f"wrote {OUT}")
