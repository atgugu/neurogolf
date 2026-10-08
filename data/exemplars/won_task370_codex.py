"""Build task370 compressed bitset member.

Design A budget (chosen): keep one uint32 row-bitset, delete the black-column
scan, use marker-vs-black-centroid for column sign, and use an ORT-safe
uint32 multiply selector for left/right shifts. Planned cost is about 6.0k:
          params: black-channel selector[10], bit weights[30],
          iota30[30], unpack weights/pads/shapes ~=50, total ~=140.
  dominant tensors: trail30 bool[1,1,30,30]=900, trail_b
          bool[1,1,20,20]=400, unpack bool/u8 chunks=800,
          3 rounds of uint32/int32 [1,1,20,1] schedule ~=2.0k,
          small projections/scalars ~=1.8k.

Design B budget (rejected): terminal ScatterND renderer from a 20x20 native
canvas needs updates over all 10 channels to clear the old background channel:
  updates f32[1,10,20,20]=16000 if exact, or bool/u8[1,10,20,20]=4000 plus
  casts/indices; with the same detector tail this is >8.5k, above the pin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).resolve().with_name("task370.onnx")
INPUT_NAME = "input"
OUTPUT_NAME = "output"
N = 20
FULL = 30
DBL = 3

nodes: list[onnx.NodeProto] = []
inits: list[onnx.TensorProto] = []
init_names: set[str] = set()


def const(name: str, arr) -> str:
    if name not in init_names:
        inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
        init_names.add(name)
    return name


def nd(op: str, ins: list[str], outs: list[str], **attrs) -> str:
    nodes.append(helper.make_node(op, ins, outs, **attrs))
    return outs[0]


def reduce_sum(src: str, axes: str, out: str) -> str:
    return nd("ReduceSum", [src, axes], [out], keepdims=1)


def reduce_max(src: str, axes: str, out: str) -> str:
    return nd("ReduceMax", [src, axes], [out], keepdims=1)


def build() -> onnx.ModelProto:
    const("one_f", np.array(1.0, np.float32))
    const("zero_f", np.array(0.0, np.float32))
    const("zero_i", np.array(0, np.int32))
    const("maxidx_i", np.array(N - 1, np.int32))
    const("pad_bool", np.array(False, np.bool_))
    const("scalar_shape", np.array([1], np.int64))
    const("shape11201", np.array([1, 1, N, 1], np.int64))
    const("axis_ch", np.array([1], np.int64))
    const("axis_h", np.array([2], np.int64))
    const("axes_hw", np.array([2, 3], np.int64))
    const("axes_chw", np.array([1, 3], np.int64))
    const("axes_chh", np.array([1, 2], np.int64))

    const("z1", np.array([0], np.int64))
    const("n1", np.array([N], np.int64))

    const("rng", np.arange(N, dtype=np.int32))

    bit_values = 1 << np.arange(N, dtype=np.uint32)
    bitw30 = np.zeros(FULL, dtype=np.float32)
    bitw30[:N] = bit_values.astype(np.float32)
    const("bitw30", bitw30)
    const("bitw8", (1 << np.arange(8, dtype=np.uint8)).reshape(1, 1, 1, 8))
    const("bitw4", (1 << np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4))

    csel = np.zeros(10, dtype=np.float32)
    csel[0] = 1.0
    const("csel", csel)
    nd("Einsum", [INPUT_NAME, "csel", "bitw30"], ["rowbits_all"], equation="nchw,c,w->nh")
    nd("Slice", ["rowbits_all", "z1", "n1", "axis_ch"], ["rowbits_s"])
    nd("Reshape", ["rowbits_s", "shape11201"], ["rowbits_f"])
    nd("Cast", ["rowbits_f"], ["rowbits"], to=TensorProto.UINT32)

    reduce_sum(INPUT_NAME, "axes_hw", "counts")
    nd("Equal", ["counts", "one_f"], ["mark_b"])
    nd("Cast", ["mark_b"], ["mark_f"], to=TensorProto.FLOAT)
    nd("ArgMax", ["mark_f"], ["mark_idx0"], axis=1, keepdims=0)
    nd("Reshape", ["mark_idx0", "scalar_shape"], ["mark_idx"])
    const("iota30", np.arange(FULL, dtype=np.float32))
    nd("Einsum", [INPUT_NAME, "iota30"], ["row_pos"], equation="nchw,h->nc")
    nd("Einsum", [INPUT_NAME, "iota30"], ["col_pos"], equation="nchw,w->nc")
    nd("Gather", ["row_pos", "mark_idx"], ["mr_f2"], axis=1)
    nd("Gather", ["col_pos", "mark_idx"], ["mc_f2"], axis=1)
    nd("Cast", ["mr_f2"], ["mr"], to=TensorProto.INT32)
    nd("Cast", ["mc_f2"], ["mc"], to=TensorProto.INT32)

    nd("Greater", ["rowbits_f", "zero_f"], ["s_row_b"])
    nd("Cast", ["s_row_b"], ["s_row_f"], to=TensorProto.FLOAT)
    nd("ArgMax", ["s_row_f"], ["rowmin_i"], axis=2, keepdims=1, select_last_index=0)
    nd("ArgMax", ["s_row_f"], ["rowmax_i"], axis=2, keepdims=1, select_last_index=1)
    nd("Cast", ["rowmin_i"], ["rowmin"], to=TensorProto.INT32)
    nd("Cast", ["rowmax_i"], ["rowmax"], to=TensorProto.INT32)
    nd("Less", ["mr", "rowmin"], ["row_neg"])
    nd("Sub", ["rowmax", "mr"], ["m_neg"])
    nd("Sub", ["mr", "rowmin"], ["m_pos"])
    nd("Where", ["row_neg", "m_neg", "m_pos"], ["m_i"])

    # Column sign: the marker is left of the sprite iff marker_col is less
    # than the black-pixel centroid. This removes the old black-column scan.
    nd("Gather", ["counts", "z1"], ["count0"], axis=1)
    nd("Gather", ["col_pos", "z1"], ["colsum0"], axis=1)
    nd("Mul", ["mc_f2", "count0"], ["mc_scaled"])
    nd("Less", ["mc_scaled", "colsum0"], ["col_neg"])
    nd("Not", ["col_neg"], ["col_pos_b"])
    nd("Cast", ["col_neg"], ["col_neg_u"], to=TensorProto.UINT32)
    nd("Cast", ["col_pos_b"], ["col_pos_u"], to=TensorProto.UINT32)

    reduce_max(INPUT_NAME, "axes_chw", "row_pres")
    reduce_max(INPUT_NAME, "axes_chh", "col_pres")
    nd("Cast", ["row_pres"], ["row_b30"], to=TensorProto.BOOL)
    nd("Slice", ["row_b30", "z1", "n1", "axis_h"], ["row_b"])
    nd("Einsum", ["col_pres", "bitw30"], ["valid_cols_f"], equation="nchw,w->nch")
    nd("Cast", ["valid_cols_f"], ["valid_cols"], to=TensorProto.UINT32)

    acc = "rowbits"
    for i in range(DBL):
        const(f"pow2_{i}", np.array(2**i, np.int32))
        nd("Mul", ["m_i", f"pow2_{i}"], [f"jump_{i}"])
        nd("Sub", ["zero_i", f"jump_{i}"], [f"neg_jump_{i}"])
        nd("Where", ["row_neg", f"neg_jump_{i}", f"jump_{i}"], [f"rstep_{i}"])
        nd("Reshape", [f"rstep_{i}", "scalar_shape"], [f"rstep_s_{i}"])
        nd("Sub", ["rng", f"rstep_s_{i}"], [f"idxf_r{i}"])
        nd("Clip", [f"idxf_r{i}", "zero_i", "maxidx_i"], [f"idx_r{i}"])
        nd("Gather", [acc, f"idx_r{i}"], [f"grow_{i}"], axis=2)
        nd("Cast", [f"jump_{i}"], [f"jump_u_{i}"], to=TensorProto.UINT32)
        nd("BitShift", [f"grow_{i}", f"jump_u_{i}"], [f"sh_l_{i}"], direction="LEFT")
        nd("BitShift", [f"grow_{i}", f"jump_u_{i}"], [f"sh_r_{i}"], direction="RIGHT")
        nd("Mul", [f"sh_r_{i}", "col_neg_u"], [f"sh_r_sel_{i}"])
        nd("Mul", [f"sh_l_{i}", "col_pos_u"], [f"sh_l_sel_{i}"])
        nd("Add", [f"sh_r_sel_{i}", f"sh_l_sel_{i}"], [f"shifted_{i}"])
        nd("BitwiseOr", [acc, f"shifted_{i}"], [f"acc_{i}"])
        acc = f"acc_{i}"

    nd("BitwiseXor", [acc, "rowbits"], ["trail_bits0"])
    nd("Cast", ["row_b"], ["row_b_u"], to=TensorProto.UINT32)
    nd("Mul", ["row_b_u", "valid_cols"], ["grid_bits"])
    nd("BitwiseAnd", ["trail_bits0", "grid_bits"], ["trail_bits"])
    nd("Cast", ["trail_bits"], ["trail_byte0"], to=TensorProto.UINT8)
    nd("BitwiseAnd", ["trail_byte0", "bitw8"], ["trail_unpack0"])
    nd("Cast", ["trail_unpack0"], ["trail_b0"], to=TensorProto.BOOL)
    for shift, width in ((8, 8), (16, 4)):
        const(f"shift_u{shift}", np.array(shift, np.uint32))
        nd("BitShift", ["trail_bits", f"shift_u{shift}"], [f"trail_shift{shift}"], direction="RIGHT")
        nd("Cast", [f"trail_shift{shift}"], [f"trail_byte{shift}"], to=TensorProto.UINT8)
        weights = "bitw8" if width == 8 else "bitw4"
        nd("BitwiseAnd", [f"trail_byte{shift}", weights], [f"trail_unpack{shift}"])
        nd("Cast", [f"trail_unpack{shift}"], [f"trail_b{shift}"], to=TensorProto.BOOL)
    nd("Concat", ["trail_b0", "trail_b8", "trail_b16"], ["trail_b"], axis=3)
    nd(
        "Pad",
        [
            "trail_b",
            const("pad30", np.array([0, 0, 0, 0, 0, 0, FULL - N, FULL - N], np.int64)),
            "pad_bool",
        ],
        ["trail30"],
        mode="constant",
    )
    nd("Where", ["trail30", "mark_f", INPUT_NAME], [OUTPUT_NAME])

    inp = helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task370_bitset_compress", [inp], [out], inits)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
