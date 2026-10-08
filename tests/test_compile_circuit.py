import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import compile_circuit as cc
import ngolf_einsum
from ngolf_einsum import params_of, parity


def canary_and_spec():
    """Honest cellwise canary: color in {1,2,3} AND color in {2,3,4}
    (i.e. color in {2,3}) -> recolor 5, else 0."""
    out_map = [[1.0] + [0.0] * 9,
               [0.0] * 5 + [1.0] + [0.0] * 4]
    return {
        "task": 999,
        "broadcast": "cellwise",
        "wires": {
            "a": {"kind": "literal", "cell": "same", "colors": [1, 2, 3]},
            "b": {"kind": "literal", "colors": [2, 3, 4]},
            "z": {"kind": "gate", "op": "and", "in": ["a", "b"]},
        },
        "output": {"wire": "z", "map": out_map},
    }


def canary_ref(grid):
    g = np.asarray(grid, dtype=int)
    return np.where(np.isin(g, [2, 3]), 5, 0)


def chain_spec(depth):
    """OR-chain over single-color literals — the schedule-sensitive shape."""
    wires = {}
    for i in range(depth + 1):
        wires[f"l{i}"] = {"kind": "literal", "colors": [(i % 9) + 1]}
    prev = "l0"
    for i in range(depth):
        wires[f"g{i}"] = {"kind": "gate", "op": "or", "in": [prev, f"l{i + 1}"]}
        prev = f"g{i}"
    out_map = [[1.0] + [0.0] * 9, [0.0] * 4 + [1.0] + [0.0] * 5]
    return {"wires": wires, "output": {"wire": prev, "map": out_map}}


class CompileScheduleTests(unittest.TestCase):
    def test_canary_equation_and_schedule(self):
        equation, operands, meta = cc.compile(canary_and_spec())
        self.assertEqual(meta["schedule"],
                         ["input", "lit:a", "lit:b", "gate:z", "outmap"])
        la, lb, lz = meta["labels"]["a"], meta["labels"]["b"], meta["labels"]["z"]
        self.assertEqual(equation,
                         f"nchw,c{la},c{lb},{la}{lb}{lz},{lz}o->nohw")
        self.assertEqual(operands[0], "input")
        self.assertEqual(meta["params_estimate"], 20 + 20 + 8 + 20)
        self.assertTrue(meta["deterministic_output"])
        # reserved labels never recycled as wire labels
        for w, letter in meta["labels"].items():
            self.assertNotIn(letter, "nchwo")

    def test_chain_schedule_interleaved(self):
        # Each literal must be emitted IMMEDIATELY before the gate consuming it —
        # never all literals first (that order is the 2^wires contraction blow-up).
        _, _, meta = cc.compile(chain_spec(3))
        self.assertEqual(meta["schedule"],
                         ["input", "lit:l0", "lit:l1", "gate:g0",
                          "lit:l2", "gate:g1", "lit:l3", "gate:g2", "outmap"])

    def test_mixed_ops_schedule(self):
        spec = {
            "wires": {
                "a": {"kind": "literal", "colors": [1, 2]},
                "b": {"kind": "literal", "colors": [2, 3]},
                "c1": {"kind": "literal", "colors": [4]},
                "d": {"kind": "literal", "colors": [0]},
                "g0": {"kind": "gate", "op": "xor", "in": ["a", "b"]},
                "g1": {"kind": "gate", "op": "or", "in": ["g0", "c1"]},
                "nd": {"kind": "gate", "op": "not", "in": ["d"]},
                "g2": {"kind": "gate", "op": "and", "in": ["g1", "nd"]},
            },
            "output": {"wire": "g2",
                       "map": [[1.0] + [0.0] * 9, [0.0] * 6 + [1.0] + [0.0] * 3]},
        }
        _, _, meta = cc.compile(spec)
        self.assertEqual(meta["schedule"],
                         ["input", "lit:a", "lit:b", "gate:g0", "lit:c1", "gate:g1",
                          "lit:d", "gate:nd", "gate:g2", "outmap"])
        m = cc.build(spec)
        rep = cc.verify(spec, m, draws=20, runs=5)
        self.assertTrue(rep["numpy_vs_ort_ok"], rep["first_diff"])

    def test_dedupe_identical_factors(self):
        # Same color-set literal twice + the same AND table twice -> billed once each.
        spec = {
            "wires": {
                "a": {"kind": "literal", "colors": [1, 2]},
                "b": {"kind": "literal", "colors": [1, 2]},
                "c1": {"kind": "literal", "colors": [3]},
                "g0": {"kind": "gate", "op": "and", "in": ["a", "b"]},
                "g1": {"kind": "gate", "op": "and", "in": ["g0", "c1"]},
            },
            "output": {"wire": "g1",
                       "map": [[1.0] + [0.0] * 9, [0.0] * 5 + [1.0] + [0.0] * 4]},
        }
        _, _, meta = cc.compile(spec)
        m = cc.build(spec)
        # unique tensors: lit{1,2} (20) + lit{3} (20) + AND (8) + outmap (20)
        self.assertEqual(len(m.graph.initializer), 4)
        self.assertEqual(params_of(m), 68)
        self.assertEqual(meta["params_estimate"], 68)


class CanaryParityTests(unittest.TestCase):
    def test_canary_ort_equals_numpy_and_fast(self):
        spec = canary_and_spec()
        m = cc.build(spec)
        self.assertEqual(len(m.graph.node), 1)          # memory 0 by construction
        self.assertLessEqual(params_of(m), 100)
        rep = cc.verify(spec, m, draws=40, runs=10)
        self.assertTrue(rep["numpy_vs_ort_ok"], rep["first_diff"])
        self.assertEqual(rep["draws_checked"], 40)
        self.assertTrue(rep["timing_ok"],
                        f"per-run max {rep['run_s_max']:.4f}s >= 0.1s")
        # the circuit's own grid semantics vs the hand-written reference
        ok, n, diff = parity(m, canary_ref, draws=40, seed=11)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")

    def test_zero_wire_pure_recolor(self):
        # A gateless spec — output map straight over the reserved 'color' wire —
        # is the minimal per-cell recolor circuit (equation nchw,co->nohw).
        perm = [0, 1, 2, 3, 4, 8, 6, 7, 5, 9]           # switch 5 <-> 8
        pmat = np.zeros((10, 10), np.float32)
        pmat[np.arange(10), perm] = 1.0
        spec = {"wires": {}, "output": {"wire": "color", "map": pmat.tolist()}}
        equation, _, meta = cc.compile(spec)
        self.assertEqual(equation, "nchw,co->nohw")
        self.assertEqual(meta["schedule"], ["input", "outmap"])
        m = cc.build(spec)
        self.assertEqual(params_of(m), 100)
        rep = cc.verify(spec, m, draws=20, runs=3)
        self.assertTrue(rep["numpy_vs_ort_ok"], rep["first_diff"])
        ok, n, diff = parity(m, lambda g: np.asarray(perm)[np.asarray(g, int)],
                             draws=20, seed=14)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")

    def test_keep_original_color_via_color_wire(self):
        # recolor {2,3} -> 5, ELSE KEEP: needs the reserved 'color' output wire.
        keep = np.zeros((10, 2, 10), np.float32)
        for c in range(10):
            keep[c, 0, c] = 1.0
            keep[c, 1, 5] = 1.0
        spec = {
            "wires": {
                "a": {"kind": "literal", "colors": [1, 2, 3]},
                "b": {"kind": "literal", "colors": [2, 3, 4]},
                "z": {"kind": "gate", "op": "and", "in": ["a", "b"]},
            },
            "output": {"wires": ["color", "z"], "map": keep.tolist()},
        }
        m = cc.build(spec)
        rep = cc.verify(spec, m, draws=30, runs=5)
        self.assertTrue(rep["numpy_vs_ort_ok"], rep["first_diff"])

        def ref(grid):
            g = np.asarray(grid, dtype=int)
            return np.where(np.isin(g, [2, 3]), 5, g)
        ok, n, diff = parity(m, ref, draws=30, seed=12)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")

    def test_deep_chain_is_fast(self):
        # The scheduling proof: a 4-gate chain (9 wires — the deepest that builds
        # quickly under the emitter's naive numpy validation) contracts in well
        # under 0.1 s/run because each literal is eliminated before the next
        # enters. The A/B against a literals-first schedule lives in the demo run.
        spec = chain_spec(4)
        rep = cc.verify(spec, draws=10, runs=10)
        self.assertTrue(rep["numpy_vs_ort_ok"], rep["first_diff"])
        self.assertTrue(rep["timing_ok"],
                        f"per-run max {rep['run_s_max']:.4f}s >= 0.1s")

    def test_verify_catches_a_wrong_model(self):
        # Model built from a DIFFERENT circuit must fail numpy-vs-ORT for this spec.
        wrong = canary_and_spec()
        wrong["wires"]["z"]["op"] = "or"
        m_wrong = cc.build(wrong)
        rep = cc.verify(canary_and_spec(), m_wrong, draws=40, runs=1)
        self.assertFalse(rep["numpy_vs_ort_ok"])
        self.assertIsNotNone(rep["first_diff"])


class ValidationTests(unittest.TestCase):
    def test_naive_validation_overflow_errors_loudly(self):
        # depth 8 -> 17 Bool wires -> 90000 * 2^17 = 1.2e10 naive-validation
        # iterations: np.einsum(optimize=False) inside build_einsum_model would
        # hang for minutes, so compile() must refuse loudly up front.
        with self.assertRaisesRegex(ValueError, "naive-validation overflow"):
            cc.compile(chain_spec(8))

    def test_label_overflow_errors_loudly(self):
        # depth 14 -> 29 wires + 5 reserved = 34 distinct subscripts: past numpy's
        # 32-subscript cap (checked with the naive-cost guard lifted, which
        # otherwise fires first).
        with self.assertRaisesRegex(ValueError, "label overflow"):
            cc.compile(chain_spec(14), max_naive_cost=float("inf"))
        with self.assertRaisesRegex(ValueError, "label overflow"):
            cc.compile(chain_spec(30), max_naive_cost=float("inf"))

    def test_unused_wire_rejected(self):
        spec = canary_and_spec()
        spec["wires"]["orphan"] = {"kind": "literal", "colors": [7]}
        with self.assertRaisesRegex(ValueError, "never reach the output"):
            cc.compile(spec)

    def test_gate_cycle_rejected(self):
        spec = {
            "wires": {
                "a": {"kind": "literal", "colors": [1]},
                "p": {"kind": "gate", "op": "and", "in": ["q", "a"]},
                "q": {"kind": "gate", "op": "and", "in": ["p", "a"]},
            },
            "output": {"wire": "q", "map": [[1.0] + [0.0] * 9,
                                            [0.0] * 4 + [1.0] + [0.0] * 5]},
        }
        with self.assertRaisesRegex(ValueError, "cycle"):
            cc.compile(spec)

    def test_bad_tables_rejected(self):
        with self.assertRaisesRegex(ValueError, "one-hot"):
            cc.Circuit({
                "wires": {"s": {"kind": "literal",
                                "table": [[1.0, 1.0]] * 10}},
                "output": {"wire": "s", "map": [[1.0] + [0.0] * 9] * 2},
            })
        with self.assertRaisesRegex(ValueError, "op must be one of"):
            cc.compile({
                "wires": {"a": {"kind": "literal", "colors": [1]},
                          "z": {"kind": "gate", "op": "nand", "in": ["a", "a"]}},
                "output": {"wire": "z", "map": [[1.0] + [0.0] * 9] * 2},
            })

    def test_non_cellwise_broadcast_is_todo(self):
        spec = canary_and_spec()
        spec["broadcast"] = "neighborhood"
        with self.assertRaises(NotImplementedError):
            cc.compile(spec)

    def test_output_map_shape_checked(self):
        spec = canary_and_spec()
        spec["output"]["map"] = [[1.0] + [0.0] * 9] * 3     # domain(z)=2, not 3
        with self.assertRaisesRegex(ValueError, "output map shape"):
            cc.compile(spec)


class CliTests(unittest.TestCase):
    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "spec.json")
            out = os.path.join(tmp, "canary.onnx")
            json.dump(canary_and_spec(), open(sp, "w"))
            rc = cc.main([sp, "--out", out, "--draws", "10", "--runs", "3"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out))
            ok, n, diff = parity(out, canary_ref, draws=10, seed=13)
            self.assertTrue(ok, f"parity failed after {n} draws: {diff}")


@unittest.skipUnless(os.path.isdir(f"{ngolf_einsum.CLEAN}/extracted"),
                     "neurogolf_clean extracted grids not present")
class RealGridTests(unittest.TestCase):
    def test_parity_on_real_graded_grids(self):
        # The 162 verified rule.py files contain NO pure per-cell recolor rule
        # (scan evidence in intel/circuit_priorities.md), so this exercises the
        # real-grid parity path with the canary circuit as its own reference.
        spec = canary_and_spec()
        m = cc.build(spec)
        ok, n, diff = parity(m, canary_ref, task=5, draws=12)
        self.assertTrue(ok, f"parity failed after {n} draws: {diff}")
        self.assertEqual(n, 12)
        rep = cc.verify(spec, m, draws=5, runs=3, task=5, parity_draws=12)
        self.assertTrue(rep["parity"]["ok"], rep["parity"])


if __name__ == "__main__":
    unittest.main()
