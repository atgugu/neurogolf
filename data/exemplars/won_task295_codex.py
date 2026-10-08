from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, "../../runner")
from ngolf import G  # noqa: E402


OUT = Path(__file__).resolve().parent / "task295.onnx"

# Binding compact-family paper budget was 438 B memory + 44 params.  The
# executable exact variant keeps the two full planes but needs the source-rule
# scalar/broadcast helpers the paper table omitted:
#   bool/u8 [1,1,9,18]                        324 B
#   f32[10]                                    40 B
#   remaining scalar/vector/broadcast helpers 124 B
#   memory                                    488 B
#   initializers                               44 elements
#   total                                      532 cost units
EXPECTED_TABLE = sorted(
    [
        ((), "uint8"),
        ((1,), "uint8"), ((1,), "uint8"), ((1,), "uint8"),
        ((1, 1, 1, 18), "bool"), ((1, 1, 1, 18), "uint8"),
        ((1, 1, 9, 1), "bool"),
        ((1, 1, 9, 1), "uint8"), ((1, 1, 9, 1), "uint8"),
        ((1, 1, 9, 1), "uint8"),
        ((1, 1, 9, 18), "bool"), ((1, 1, 9, 18), "uint8"),
        ((9,), "uint8"), ((9,), "int8"),
        ((10,), "float32"), ((10,), "uint8"), ((10,), "int8"),
        ((10, 1, 1, 1), "int8"),
    ]
)


def build() -> G:
    g = G(task=295)

    zero = g.init(np.array(0, np.uint8))
    one = g.init(np.array(1, np.uint8))
    two = g.init(np.array(2, np.uint8))

    color_sums = g.reduce(g.input, "Sum", [0, 2, 3], keepdims=0)
    color_sums_u8 = g.cast(color_sums, "u8")
    bg_count = g.slice(color_sums_u8, [0], [1], [0])
    fg_sums = g.slice(color_sums_u8, [1], [10], [0])
    length = g.reduce(fg_sums, "Max", [0], keepdims=0)

    width = g.add(bg_count, length)
    height = g.n("Div", [width, two], [1], "u8")

    rows = g.init(np.arange(9, dtype=np.uint8).reshape(1, 1, 9, 1))
    cols = g.init(np.arange(18, dtype=np.uint8).reshape(1, 1, 1, 18))

    fill_limit = g.add(length, rows)
    row_ok = g.less(rows, height)
    row_limit = g.n("Where", [row_ok, fill_limit, width], [1, 1, 9, 1], "u8")
    row_code = g.n("Where", [row_ok, two, one], [1, 1, 9, 1], "u8")
    col_ok = g.less(cols, width)
    col_code = g.n("Where", [col_ok, zero, one], [1, 1, 1, 18], "u8")

    # Two native planes only.  For valid rows crop selects the triangular
    # foreground; for invalid rows row_limit=255 makes it select row_code=1.
    # The false branch supplies background=0 or invalid-column=1.  Therefore
    # packed is exactly the proven terminal code: bg=0, absent=1, foreground=2.
    crop = g.less(cols, row_limit)
    packed = g.n("Where", [crop, row_code, col_code], [1, 1, 9, 18], "u8")

    fg_weights = g.cast(fg_sums, "i8")
    bg_weight = g.init(np.array([-1], np.int8))
    weights_1d = g.concat([bg_weight, fg_weights], axis=0)
    weights = g.reshape(weights_1d, [10, 1, 1, 1])
    g.n(
        "ConvInteger",
        [packed, weights, one],
        [1, 10, 30, 30],
        "i32",
        is_output=True,
        kernel_shape=[1, 1],
        pads=[0, 0, 21, 12],
    )
    return g


def assert_tensor_table(path: Path) -> None:
    model = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=True)
    type_by_name = {v.name: v.type.tensor_type for v in list(model.graph.value_info)}
    dtype_names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.INT8: "int8",
        onnx.TensorProto.UINT8: "uint8",
        onnx.TensorProto.BOOL: "bool",
    }
    actual = []
    for n in model.graph.node[:-1]:
        for name in n.output:
            t = type_by_name[name]
            shape = tuple(d.dim_value for d in t.shape.dim)
            actual.append((shape, dtype_names.get(t.elem_type, str(t.elem_type))))
    assert sorted(actual) == EXPECTED_TABLE, (
        "compact-family tensor table mismatch; omitted nonterminal tensors: "
        f"expected={EXPECTED_TABLE}, actual={sorted(actual)}"
    )


if __name__ == "__main__":
    graph = build()
    graph.save(OUT)
    assert_tensor_table(OUT)
