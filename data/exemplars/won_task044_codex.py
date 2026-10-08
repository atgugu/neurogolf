"""Build the task044 Lane A relower from the Kaggle-proven oracle.

The transform is intentionally mechanical:

* validate the trusted reference hash,
* preserve the exact decision graph through ``out10`` (nodes 0..136),
* delete the charged uint8 ``out30`` Pad result, and
* render/pad in one terminal ``ConvInteger`` whose graph output is free.

The requested terminal-Pad form is not expressible in standard ONNX without first
materializing a charged [1,10,10,10] bool tensor; this terminal convolution is the
legal standard-ONNX equivalent used by prior drafts, but with the proven oracle's
full decision chain restored.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


EXPECTED = "376a09bb1bcde35b790393a24bbf964ba40658652791337dface0425126fc436"
SOURCE = Path(__file__).with_name("refs") / "kaggle_proven_best.onnx"
TARGET = Path(__file__).with_name("task044.onnx")


def _remove_initializers(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [init for init in model.graph.initializer if init.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def main() -> None:
    data = SOURCE.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != EXPECTED:
        raise RuntimeError(f"reference SHA-256 mismatch: {digest}")

    model = onnx.load_from_string(data)

    # Preserve the trusted algorithm and replace only Pad(out10)->Equal(output).
    del model.graph.node[137:]
    _remove_initializers(model, {"pads", "outside_u8"})

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.ones((1, 1, 10, 10), dtype=np.uint8), name="valid10_u8"
            ),
            numpy_helper.from_array(np.array(1, dtype=np.uint8), name="render_shift_u8"),
            numpy_helper.from_array(
                np.array(
                    [
                        [[[1]], [[0]], [[-1]], [[-2]]],
                        [[[0]], [[2]], [[-1]], [[0]]],
                        [[[-3]], [[4]], [[-1]], [[0]]],
                        [[[-8]], [[6]], [[-1]], [[0]]],
                        [[[-15]], [[8]], [[-1]], [[0]]],
                        [[[0]], [[0]], [[0]], [[1]]],
                        [[[-35]], [[12]], [[-1]], [[0]]],
                        [[[-48]], [[14]], [[-1]], [[0]]],
                        [[[-63]], [[16]], [[-1]], [[0]]],
                        [[[-80]], [[18]], [[-1]], [[0]]],
                    ],
                    dtype=np.int8,
                ),
                name="render_w4_i8",
            ),
        ]
    )

    model.graph.node.extend(
        [
            helper.make_node(
                "BitShift",
                ["out10", "render_shift_u8"],
                ["out10_half"],
                direction="RIGHT",
            ),
            helper.make_node("Mul", ["out10_half", "out10_half"], ["out10_half_sq"]),
            helper.make_node(
                "Concat",
                ["valid10_u8", "out10_half", "out10_half_sq", "gray"],
                ["render_feat4"],
                axis=1,
            ),
            helper.make_node(
                "ConvInteger",
                ["render_feat4", "render_w4_i8"],
                ["output"],
                pads=[0, 0, 20, 20],
            ),
        ]
    )

    del model.graph.output[:]
    model.graph.output.extend(
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])]
    )

    # The reference carries a priced value_info manifest, including the removed
    # out30 tensor.  Clear it so the emitted model's manifest matches its nodes.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, TARGET)
    print(f"wrote {TARGET.name} sha256={sha256(TARGET.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
