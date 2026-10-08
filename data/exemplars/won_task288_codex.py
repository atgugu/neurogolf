import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name, array):
    return numpy_helper.from_array(np.asarray(array), name=name)


def node(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


def main():
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    initializers = [
        init("axes_color", np.array([0, 2, 3], dtype=np.int64)),
        init("bg_cases", np.array([5, 19, 41, 17, 39, 69], dtype=np.float32)),
        init("one_u8", np.array(1, dtype=np.uint8)),
        init("two_u8", np.array(2, dtype=np.uint8)),
        init("three_u8", np.array(3, dtype=np.uint8)),
        init("j6", np.array([[0], [0], [1], [1], [2], [2]], dtype=np.uint8)),
        init("is_right", np.array([[False], [True], [False], [True], [False], [True]])),
        init("zero6", np.zeros((6, 1), dtype=np.uint8)),
        init("zero12", np.zeros((12, 1), dtype=np.uint8)),
        init("scatter_updates", np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)),
    ]

    nodes = [
        node("ReduceSum", ["input", "axes_color"], ["color_counts"], keepdims=0),
        node("ReduceMax", ["color_counts"], ["background_count"], axes=[0], keepdims=0),
        node("Equal", ["background_count", "bg_cases"], ["case_match"]),
        node("Cast", ["case_match"], ["case_match_u8"], to=TensorProto.UINT8),
        node("ArgMax", ["case_match_u8"], ["case_index_i64"], axis=0, keepdims=0),
        node("Cast", ["case_index_i64"], ["case_u8"], to=TensorProto.UINT8),
        node("Mod", ["case_u8", "three_u8"], ["case_mod"], fmod=0),
        node("Add", ["case_mod", "one_u8"], ["shoulder"]),
        node("Greater", ["case_u8", "two_u8"], ["neck_is_three"]),
        node("Where", ["neck_is_three", "three_u8", "one_u8"], ["neck"]),
        node("Cast", ["neck"], ["neck_f"], to=TensorProto.FLOAT),
        node("Equal", ["color_counts", "neck_f"], ["antenna_channel"]),
        node("Cast", ["antenna_channel"], ["antenna_channel_u8"], to=TensorProto.UINT8),
        node("ArgMax", ["antenna_channel_u8"], ["antenna_index_i64"], axis=0, keepdims=0),
        node("Cast", ["antenna_index_i64"], ["antenna_index_u8"], to=TensorProto.UINT8),
        node("Add", ["shoulder", "shoulder"], ["two_shoulder"]),
        node("Add", ["neck", "two_shoulder"], ["size"]),
        node("Add", ["neck", "shoulder"], ["neck_plus_shoulder"]),
        node("Sub", ["neck_plus_shoulder", "two_u8"], ["base_row"]),
        node("Add", ["base_row", "j6"], ["row_raw"]),
        node("Less", ["j6", "shoulder"], ["valid_slot"]),
        node("Where", ["valid_slot", "row_raw", "base_row"], ["row6"]),
        node("Sub", ["size", "one_u8"], ["last_col"]),
        node("Sub", ["last_col", "j6"], ["right_col"]),
        node("Where", ["is_right", "right_col", "j6"], ["col_raw"]),
        node("Where", ["valid_slot", "col_raw", "zero6"], ["col6"]),
        node("Concat", ["row6", "col6"], ["rowcol6"], axis=1),
        node("Concat", ["rowcol6", "rowcol6"], ["rowcol12"], axis=0),
        node("Add", ["zero6", "antenna_index_u8"], ["antenna6"]),
        node("Concat", ["zero6", "antenna6"], ["channel_col"], axis=0),
        node("Concat", ["zero12", "channel_col", "rowcol12"], ["scatter_indices_u8"], axis=1),
        node("Cast", ["scatter_indices_u8"], ["scatter_indices"], to=TensorProto.INT64),
        node("ScatterND", ["input", "scatter_indices", "scatter_updates"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task288_sparse_antenna", [inp], [out], initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        producer_name="neurogolf_task288",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "task288.onnx")


if __name__ == "__main__":
    main()
