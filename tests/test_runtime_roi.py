import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import orchestrate
import runtime_roi


class RuntimeROIEstimateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name) / "results"
        self.results.mkdir()
        self.difficulty_path = self.results / "DIFFICULTY.csv"
        with self.difficulty_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "task", "tier", "edelta", "target_cost", "banked", "proven", "fam"])
            writer.writeheader()
            writer.writerow({"task": "task001", "tier": "EASY", "edelta": 2.0,
                             "target_cost": 100, "banked": 0, "proven": 0,
                             "fam": "LUT-REFINE"})
        runtime_roi.clear_cache()

    def tearDown(self):
        runtime_roi.clear_cache()
        self.tmp.cleanup()

    @staticmethod
    def difficulty(attempts=0, nos=0, failed=0, banked=0.0, tier="EASY"):
        return {"score": 10, "tier": tier, "reasons": [], "signals": {
            "clean_attempts": attempts, "clean_nos": nos,
            "failed_providers": failed, "banked_delta": banked}}

    def test_live_evidence_and_banked_points_reduce_expected_yield(self):
        row = {"task": "task001", "pin_cost": "1000", "priority": "1"}
        virgin = runtime_roi.estimate(
            "001", row, "codex", str(self.results), self.difficulty())
        exhausted = runtime_roi.estimate(
            "001", row, "codex", str(self.results),
            self.difficulty(attempts=7, nos=3, failed=2, banked=.8))
        self.assertGreater(virgin["remaining_delta"], exhausted["remaining_delta"])
        self.assertGreater(virgin["pwin"], exhausted["pwin"])
        self.assertGreater(virgin["points_per_hour"], exhausted["points_per_hour"])
        self.assertTrue(exhausted["floored"])
        self.assertGreater(exhausted["pwin"], 0)  # endless-mode coverage, not abandonment

    def test_pre_win_nos_do_not_recreate_a_refuted_floor(self):
        row = {"task": "task001", "pin_cost": "1000", "priority": "1"}
        signals = self.difficulty(attempts=6, nos=4, failed=2, banked=.2)
        signals["signals"].update(post_win_nos=0, post_win_no_providers=0)
        live = runtime_roi.estimate("001", row, "codex", str(self.results), signals)
        self.assertFalse(live["floored"])

    def test_conservative_runtime_prior_until_enough_current_model_samples(self):
        seconds, source, count = runtime_roi.expected_seconds(
            str(self.results), "codex", "EASY", "LUT-REFINE")
        self.assertEqual(3600, seconds)
        self.assertEqual("conservative-prior", source)
        self.assertEqual(0, count)

    def test_estimate_never_writes_signed_inputs(self):
        queue = Path(self.tmp.name) / "QUEUE.csv"
        queue.write_text("task,priority\ntask001,1\n")
        before = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                  for path in (queue, self.difficulty_path)}
        runtime_roi.estimate(
            "001", {"task": "task001", "pin_cost": "1000", "priority": "1"},
            "codex", str(self.results), self.difficulty())
        after = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                 for path in (queue, self.difficulty_path)}
        self.assertEqual(before, after)


class RuntimeROISchedulerTests(unittest.TestCase):
    def setUp(self):
        self.spawn_counts = dict(orchestrate.SPAWN_COUNTS)
        self.yield_cache = dict(orchestrate._RUNTIME_YIELD_CACHE)
        orchestrate._RUNTIME_YIELD_CACHE.clear()
        orchestrate.SPAWN_COUNTS.update(codex=0)

    def tearDown(self):
        orchestrate.SPAWN_COUNTS.update(self.spawn_counts)
        orchestrate._RUNTIME_YIELD_CACHE.clear()
        orchestrate._RUNTIME_YIELD_CACHE.update(self.yield_cache)

    @staticmethod
    def rows(n=5):
        return [{"task": f"task{i:03d}", "priority": str(i)} for i in range(1, n + 1)]

    def test_higher_seat_hour_roi_beats_higher_raw_ev(self):
        rows = self.rows(2)
        # task001 has higher raw EV but is much slower; task002 has higher seat-hour yield.
        estimates = {
            "001": {"expected_points": .10, "expected_seconds": 7200,
                    "points_per_hour": .05},
            "002": {"expected_points": .08, "expected_seconds": 1800,
                    "points_per_hour": .16},
        }
        with mock.patch.object(orchestrate, "_task_difficulty", return_value={
                "score": 20, "tier": "EASY", "signals": {}}), \
                mock.patch.object(orchestrate, "_runtime_yield",
                                  side_effect=lambda t, row, provider, difficulty=None:
                                  estimates[t]):
            self.assertEqual([1, 0], orchestrate._worker_task_order(rows, "codex"))

    def test_every_tenth_spawn_uses_original_priority_as_coverage_guard(self):
        rows = self.rows(3)
        orchestrate.SPAWN_COUNTS["codex"] = 9
        with mock.patch.object(orchestrate, "_runtime_yield",
                               return_value={"points_per_hour": 999}):
            self.assertEqual([0, 1, 2], orchestrate._worker_task_order(rows, "codex"))

    def test_live_yield_is_cached_across_one_seat_fill_burst(self):
        row = {"task": "task001", "priority": "1"}
        difficulty = {"score": 20, "tier": "EASY", "signals": {}}
        estimate = {"points_per_hour": .2}
        with mock.patch.object(orchestrate, "_task_difficulty",
                               return_value=difficulty) as profile, \
                mock.patch.object(orchestrate, "runtime_roi_estimate",
                                  return_value=estimate) as scorer:
            self.assertIs(estimate, orchestrate._runtime_yield("001", row, "codex"))
            self.assertIs(estimate, orchestrate._runtime_yield("001", row, "codex"))
        profile.assert_called_once()
        scorer.assert_called_once()

    def test_sol_remains_hardest_while_base_uses_runtime_roi(self):
        ready = [
            ("001", {"task": "task001"}, {"score": 20, "tier": "EASY", "signals": {}}),
            ("002", {"task": "task002"}, {"score": 80, "tier": "VHARD", "signals": {}}),
            ("003", {"task": "task003"}, {"score": 45, "tier": "MEDIUM", "signals": {}}),
        ]
        with mock.patch.object(orchestrate, "_runtime_yield",
                               side_effect=lambda t, row, provider, difficulty=None: {
                                   "points_per_hour": {"001": .1, "002": .01, "003": .2}[t]}):
            self.assertEqual(1, orchestrate._pick_decider_index(ready, "sol"))
            self.assertEqual(2, orchestrate._pick_decider_index(ready, "base"))


if __name__ == "__main__":
    unittest.main()
