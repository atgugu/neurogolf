import os
#!/usr/bin/env python3
"""Build task159 with a compact uint8 terminal-Einsum renderer.

Design A (built): keep the scalar detector/crop from the live 33-node pin, then
replace the target14 -> target30 -> Equal renderer with a 4x4 logical spatial
patch, a 4x30 repeat selector, and a final Einsum:

  bdxy,bdc,xh,yw->bchw

This one terminal op performs magnification, channel routing, and 30x30
one-hot rendering at once. The fourth row/column is the red border; outside
the true output has all-zero selector columns from a compact 4x5 selector LUT.
Attempt 10 relowers the value-exact f16 renderer operands to uint8.

Design B (rejected): uint8/MatMulInteger or color-id plane terminal renderer.
The final expansion would be cheap, but the required 30x30 state or int32
intermediate prices above 2.5 KB, so it misses the 2x target.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper


TASK = 159
HERE = Path(__file__).resolve().parent
PIN = Path((os.environ.get("NEUROGOLF_CLEAN", "") + "/tasks/task159/_live_pin.onnx"))
OUT = HERE / "task159.onnx"


def arr(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def needed_initializers(model: onnx.ModelProto, names: set[str]) -> list[onnx.TensorProto]:
    return [ini for ini in model.graph.initializer if ini.name in names]


def build() -> onnx.ModelProto:
    pin = onnx.load(str(PIN))

    # Pin nodes 0..19 compute sprite color, sprite top-left, and patch_bool.
    # Nodes 25..28 compute the red-frame scale index and repeat permutation.
    keep_indices = list(range(20)) + [25, 26, 27, 28]
    nodes = [pin.graph.node[i] for i in keep_indices]

    keep_inits = {
        "motif_mask",
        "color_weights",
        "red_mask",
        "desc30",
        "inv_ln4",
        "unsqueeze_axis0",
        "const_29_vec",
        "slice_sizes",
        "slice_axes",
        "const_16",
        "const_12",
        "repeat_bank",
    }
    inits = needed_initializers(pin, keep_inits)

    red4 = np.zeros((1, 1, 4, 4), dtype=np.uint8)
    red4[:, :, 3, :4] = 1
    red4[:, :, :4, 3] = 1
    inner4 = np.zeros((1, 1, 4, 4), dtype=np.bool_)
    inner4[:, :, :3, :3] = 1

    red_vec = np.zeros((1, 1, 10), dtype=np.uint8)
    red_vec[0, 0, 2] = 1
    bg_vec = np.zeros((1, 1, 10), dtype=np.uint8)
    bg_vec[0, 0, 0] = 1

    selector_lut = np.zeros((4, 5), dtype=np.uint8)
    selector_lut[np.arange(4), np.arange(4)] = 1

    inits.extend(
        [
            arr("selector_lut4x5_u8", selector_lut),
            arr("pad_sprite4", np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)),
            arr("pad_sel30", np.array([0, 0, 0, 16], dtype=np.int64)),
            arr("red4_u8", red4),
            arr("inner4_b", inner4),
            arr("red_vec_u8", red_vec),
            arr("bg_vec_u8", bg_vec),
            arr("color_range_i32", np.arange(10, dtype=np.int32).reshape(1, 1, 10)),
        ]
    )

    nodes.extend(
        [
            helper.make_node(
                "Gather",
                ["selector_lut4x5_u8", "repeat_perm"],
                ["sel4x14_u8"],
                axis=1,
            ),
            helper.make_node("Pad", ["sel4x14_u8", "pad_sel30"], ["sel4x30_u8"]),
            helper.make_node("Pad", ["patch_bool", "pad_sprite4"], ["sprite4_b"]),
            helper.make_node("Cast", ["sprite4_b"], ["sprite4_u8"], to=TP.UINT8),
            helper.make_node("Not", ["sprite4_b"], ["not_sprite4_b"]),
            helper.make_node("And", ["inner4_b", "not_sprite4_b"], ["bg4_b"]),
            helper.make_node("Cast", ["bg4_b"], ["bg4_u8"], to=TP.UINT8),
            helper.make_node(
                "Concat",
                ["red4_u8", "sprite4_u8", "bg4_u8"],
                ["spatial4_u8"],
                axis=1,
            ),
            helper.make_node("Equal", ["sprite_channel", "color_range_i32"], ["sprite_color_b"]),
            helper.make_node("Cast", ["sprite_color_b"], ["sprite_color_u8"], to=TP.UINT8),
            helper.make_node(
                "Concat",
                ["red_vec_u8", "sprite_color_u8", "bg_vec_u8"],
                ["colors3_u8"],
                axis=1,
            ),
            helper.make_node(
                "Einsum",
                ["spatial4_u8", "colors3_u8", "sel4x30_u8", "sel4x30_u8"],
                ["output"],
                equation="bdxy,bdc,xh,yw->bchw",
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task159_terminal_einsum_selector",
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.UINT8, [1, 10, 30, 30])],
        inits,
    )
    graph.value_info.extend(
        [
            helper.make_tensor_value_info("patch_selected", TP.FLOAT, [1, 1, 3, 3]),
            helper.make_tensor_value_info("patch_bool", TP.BOOL, [1, 1, 3, 3]),
            helper.make_tensor_value_info("sprite4_b", TP.BOOL, [1, 1, 4, 4]),
            helper.make_tensor_value_info("not_sprite4_b", TP.BOOL, [1, 1, 4, 4]),
            helper.make_tensor_value_info("bg4_b", TP.BOOL, [1, 1, 4, 4]),
            helper.make_tensor_value_info("sprite4_u8", TP.UINT8, [1, 1, 4, 4]),
            helper.make_tensor_value_info("bg4_u8", TP.UINT8, [1, 1, 4, 4]),
            helper.make_tensor_value_info("spatial4_u8", TP.UINT8, [1, 3, 4, 4]),
        ]
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
