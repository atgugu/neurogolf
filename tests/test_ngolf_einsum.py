import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import ngolf_einsum
from ngolf_einsum import build_einsum_model, memory_of, out_einsum, params_of, parity

# A fixed 10-color permutation and its matrix P[c, PERM[c]] = 1, so
# einsum("bchw,cd->bdhw", onehot, P) is recolor-by-permutation.
PERM = np.array([3, 0, 7, 1, 9, 2, 8, 4, 6, 5])
P = np.zeros((10, 10), dtype=np.float32)
P[np.arange(10), PERM] = 1.0

# A many-to-one recolor map for the repeated-input case.
MAP = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5])
M = np.zeros((10, 10), dtype=np.float32)
M[np.arange(10), MAP] = 1.0


def recolor_ref(grid):
    return PERM[np.asarray(grid, dtype=int)]


def recolor_twice_ref(grid):
    return PERM[PERM[np.asarray(grid, dtype=int)]]


def map_ref(grid):
    return MAP[np.asarray(grid, dtype=int)]


class BuildEinsumModelTests(unittest.TestCase):
    def test_recolor_by_permutation(self):
        m = build_einsum_model("bchw,cd->bdhw", ["input", P])
        self.assertEqual(len(m.graph.node), 1)
        self.assertEqual(m.graph.node[0].op_type, "Einsum")
        self.assertEqual(list(m.graph.node[0].input), ["input", "t1"])
        self.assertEqual(m.graph.output[0].name, "output")
        self.assertEqual(params_of(m), 100)
        self.assertEqual(memory_of(m), 0)  # single node -> zero charged memory
        onnx.checker.check_model(m, full_check=True)
        ok, n, diff = parity(m, recolor_ref, draws=40, seed=0)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")
        self.assertEqual(n, 40)

    def test_repeated_input_operand(self):
        # "input" appears TWICE: self-product over the one-hot planes collapsed
        # through a tiny recolor factor (x*x == x on one-hot, so the numpy ref is
        # the plain recolor map).
        m = build_einsum_model("bchw,bchw,cd->bdhw", ["input", "input", M])
        self.assertEqual(len(m.graph.node), 1)
        self.assertEqual(list(m.graph.node[0].input), ["input", "input", "t1"])
        self.assertEqual(params_of(m), 100)
        self.assertEqual(memory_of(m), 0)
        onnx.checker.check_model(m, full_check=True)
        ok, n, diff = parity(m, map_ref, draws=40, seed=1)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")
        self.assertEqual(n, 40)

    def test_reused_factor_deduped_once(self):
        # The same P object passed twice must be stored as ONE initializer.
        m = build_einsum_model("bchw,cd,de->behw", ["input", P, P])
        self.assertEqual(len(m.graph.initializer), 1)
        self.assertEqual(list(m.graph.node[0].input), ["input", "t1", "t1"])
        self.assertEqual(params_of(m), 100)
        ok, n, diff = parity(m, recolor_twice_ref, draws=40, seed=2)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")

    def test_operand_order_preserved(self):
        # Caller's interleaving is load-bearing: factor-first must stay factor-first.
        m = build_einsum_model("cd,bchw->bdhw", [P, "input"])
        self.assertEqual(list(m.graph.node[0].input), ["t1", "input"])
        ok, n, diff = parity(m, recolor_ref, draws=10, seed=3)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")

    def test_output_shape_crosscheck(self):
        m = build_einsum_model("bchw,cd->bdhw", ["input", P],
                               output_shape=[1, 10, 30, 30])
        self.assertEqual(len(m.graph.node), 1)
        with self.assertRaises(ValueError):
            build_einsum_model("bchw,cd->bdhw", ["input", P],
                               output_shape=[1, 10, 30, 31])

    def test_validation_errors(self):
        with self.assertRaises(ValueError):   # operand count != LHS terms
            build_einsum_model("bchw,cd->bdhw", ["input"])
        with self.assertRaises(ValueError):   # only the literal "input" is allowed
            build_einsum_model("bchw,cd->bdhw", ["inptu", P])
        with self.assertRaises(ValueError):   # banned substring in a tensor name
            build_einsum_model("bchw,cd->bdhw", ["input", P], out_name="NonZero_out")

    def test_parity_detects_a_wrong_reference(self):
        m = build_einsum_model("bchw,cd->bdhw", ["input", P])
        ok, n, diff = parity(m, map_ref, draws=40, seed=0)  # wrong ref on purpose
        self.assertFalse(ok)
        self.assertIsNotNone(diff)
        self.assertGreater(diff["cells_differ"], 0)


class NgolfGPathTests(unittest.TestCase):
    def test_terminal_render_into_ngolf_graph(self):
        from ngolf import G
        g = G(task=999)
        t = out_einsum(g, "bchw,cd->bdhw", ["input", P])
        self.assertTrue(g.done)
        self.assertEqual(t.name, "output")
        self.assertEqual(t.shape, [1, 10, 30, 30])
        self.assertEqual(g.params, 100)
        self.assertEqual(g.cost(), 100)       # zero charged intermediates
        self.assertEqual(g.opset, 17)         # caller's opset untouched
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "einsum_g.onnx")
            g.save(path)                      # ngolf's own checker+ORT probe
            ok, n, diff = parity(path, recolor_ref, draws=10, seed=4)
            self.assertTrue(ok, f"parity failed after {n} draws: {diff}")

    def test_g_path_dedupes_within_call(self):
        from ngolf import G
        g = G()
        out_einsum(g, "bchw,cd,de->behw", ["input", P, P])
        self.assertEqual(g.params, 100)
        self.assertEqual(len(g.inits), 1)


@unittest.skipUnless(os.path.isdir(f"{ngolf_einsum.CLEAN}/extracted"),
                     "neurogolf_clean extracted grids not present")
class TaskGridParityTests(unittest.TestCase):
    def test_parity_on_real_task_grids(self):
        eye = np.eye(10, dtype=np.float32)
        m = build_einsum_model("bchw,cd->bdhw", ["input", eye])  # identity recolor
        ok, n, diff = parity(m, lambda grid: grid, task=5, draws=12)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")
        self.assertEqual(n, 12)


if __name__ == "__main__":
    unittest.main()
