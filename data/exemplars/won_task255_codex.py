#!/usr/bin/env python3
from pathlib import Path

import onnx
from onnx import TensorProto, helper


PIN = Path("pin255.onnx")
if not PIN.exists():
    raise SystemExit("pin255.onnx missing; extract task255.onnx from the pin into this pack first")

model = onnx.load(PIN)
graph = model.graph

# Drop the fp16 part of the pin compositor:
#   bool Concat row/col predicates -> fp16 Cast -> MatMul -> Greater -> final Where.
# Keep the two Concat nodes and replace only the typed matrix multiply tail.
kept = [node for node in graph.node if "output" not in node.output and node.output[0] not in {
    "safe_name_152",
    "safe_name_153",
    "safe_name_154",
    "safe_name_155",
}]
del graph.node[:]
graph.node.extend(kept)

# Remove the now-unused fp16 zero threshold initializer.
kept_inits = [init for init in graph.initializer if init.name != "safe_name_29"]
del graph.initializer[:]
graph.initializer.extend(kept_inits)


def add_node(op_type, inputs, outputs, **attrs):
    graph.node.append(helper.make_node(op_type, inputs, outputs, **attrs))


add_node("Cast", ["safe_name_150"], ["row_u8"], to=TensorProto.UINT8)
add_node("Cast", ["safe_name_151"], ["col_u8"], to=TensorProto.UINT8)
add_node(
    "QLinearMatMul",
    [
        "row_u8",
        "safe_name_21",
        "safe_name_22",
        "col_u8",
        "safe_name_21",
        "safe_name_22",
        "safe_name_21",
        "safe_name_22",
    ],
    ["geom_count"],
)
add_node("Greater", ["geom_count", "safe_name_22"], ["geom_mask"])
add_node("Where", ["geom_mask", "safe_name_19", "input"], ["output"])

model.ir_version = 8
onnx.checker.check_model(model, full_check=True)
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.save(model, "task255.onnx")
print("saved task255.onnx")
