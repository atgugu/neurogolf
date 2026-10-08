import os
#!/usr/bin/env python3
"""Build task201 with a sentinel-yellow color-id representation."""

from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parent
PIN_ZIP = Path((os.environ.get("NEUROGOLF_CLEAN", "") + "/submission.zip"))
OUT = ROOT / "task201.onnx"


def clone_node(n, inputs=None, outputs=None):
    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    return helper.make_node(n.op_type, list(inputs or n.input), list(outputs or n.output), **attrs)


def main():
    with zipfile.ZipFile(PIN_ZIP) as z:
        model = onnx.load_from_string(z.read("task201.onnx"))
    graph = model.graph

    # Color-id projection: encode yellow as sentinel 10 instead of 4. This lets the
    # frame bbox use ReduceMax(labels_raw) directly and deletes yellow Equal+Cast planes.
    for t in graph.initializer:
        if t.name == "conv_weights":
            arr = numpy_helper.to_array(t).copy()
            arr[0, 4, :, :] = 0.0
            arr[0, 4, 0, 0] = 10.0
            t.CopyFrom(numpy_helper.from_array(arr.astype(np.float32), "conv_weights"))
            break
    else:
        raise RuntimeError("conv_weights not found")

    old = list(graph.node)
    by_output = {n.output[0]: n for n in old if n.output}
    new = []
    for n in old:
        out = n.output[0] if n.output else ""
        if out in {"yellow_mask", "yellow_u8", "pattern_nonzero_u8"}:
            continue
        if out in {"yellow_rows", "yellow_cols"}:
            # Same ReduceMax axes/attrs, but over sentinel-coded labels.
            new.append(clone_node(n, inputs=["labels_raw", n.input[1]]))
            continue
        if out == "pattern_rows":
            new.append(clone_node(n, inputs=["pattern_labels", n.input[1]], outputs=["pattern_row_max"]))
            new.append(helper.make_node("Min", ["pattern_row_max", "one_u8"], ["pattern_rows"]))
            continue
        if out == "pattern_cols":
            new.append(clone_node(n, inputs=["pattern_labels", n.input[1]], outputs=["pattern_col_max"]))
            new.append(helper.make_node("Min", ["pattern_col_max", "one_u8"], ["pattern_cols"]))
            continue
        new.append(n)

    del graph.node[:]
    graph.node.extend(new)

    # Preserve the pin's value_info for dynamic Slice tensors, and add the two new vectors.
    existing_vi = {vi.name for vi in graph.value_info}
    if "pattern_row_max" not in existing_vi:
        graph.value_info.append(helper.make_tensor_value_info("pattern_row_max", onnx.TensorProto.UINT8, [1, 13]))
    if "pattern_col_max" not in existing_vi:
        graph.value_info.append(helper.make_tensor_value_info("pattern_col_max", onnx.TensorProto.UINT8, [1, 13]))

    model.ir_version = 10
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, 18)

    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = model.graph
    node_outputs = {o for n in graph.node for o in n.output if o}
    kept_vi = [vi for vi in graph.value_info if vi.name in node_outputs]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)
    onnx.save(model, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
