"""Task 275: exact donor algebra with feature contractions fused terminally.

The required fp32 graph input forces ReduceSum and Einsum results to fp32 under
ONNX's homogeneous type constraints. Inlining both feature contractions into the
terminal renderer removes their charged outputs while preserving donor values.
"""

import copy
import sys

import onnx
from onnx import helper


sys.path.insert(0, "../../runner")
import ngolf  # noqa: E402


DONOR = "refs/kaggle_proven_best.onnx"
OUTPUT = "task275.onnx"


def build():
    model = copy.deepcopy(ngolf.onnx.load(DONOR))
    graph = model.graph
    del graph.node[5:]
    graph.node.append(
        helper.make_node(
            "Einsum",
            [
                "input", "cw_source", "case", "fold", "fold",
                "input", "cw_source", "case", "fold", "fold",
                "degree",
                "input", "cw_mask", "case", "fold", "fold",
                "case", "block_route", "fold", "block_route", "fold", "coeff",
            ],
            ["output"],
            equation=(
                "bxhw,xa,u,uhp,uwq,"
                "byHW,yd,v,vHp,vWq,ade,"
                "btRC,tm,s,sRr,sCc,"
                "z,zip,zir,zjq,zjc,kem->bkij"
            ),
            name="terminal_renderer",
        )
    )
    graph.name = "task275_att08_terminal_fused"
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(f"wrote {OUTPUT}")
