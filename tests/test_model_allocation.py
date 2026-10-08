import csv
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import evidence
import orchestrate
import task_difficulty
import rank_difficulty


class ShareAllocationTests(unittest.TestCase):
    @staticmethod
    def sequence(shares, count, batch):
        active = {provider: 0 for provider in shares}
        spawned = {provider: 0 for provider in shares}
        sequence = []
        for _ in range(count):
            provider = orchestrate._share_order(
                shares, active, spawned, lambda _: True)[0]
            active[provider] += 1
            spawned[provider] += 1
            sequence.append(provider)
            if sum(active.values()) == batch:
                active = {name: 0 for name in shares}
        return sequence

    def test_worker_mix_is_codex_only_after_wave5_verdict(self):
        counts = Counter(self.sequence(orchestrate.WORKER_SHARES, 20, 10))
        self.assertEqual({"codex": 20}, dict(counts))

    def test_supplied_capability_ranking_is_recorded(self):
        self.assertEqual(22, sum(len(levels) for levels in orchestrate.MODEL_BENCHMARKS.values()))
        self.assertEqual(58, orchestrate.MODEL_BENCHMARKS["gpt-5.6-sol"]["xhigh"])
        self.assertEqual(55, orchestrate.MODEL_BENCHMARKS["gpt-5.6-terra"]["max"])
        self.assertEqual(55, orchestrate.MODEL_BENCHMARKS["gpt-5.5"]["xhigh"])
        self.assertEqual(51, orchestrate.MODEL_BENCHMARKS["gpt-5.6-luna"]["max"])

    def test_decider_mix_is_codex_only_after_wave5_verdict(self):
        counts = Counter(self.sequence(orchestrate.DECIDER_SHARES, 20, 6))
        self.assertEqual({"codex": 20}, dict(counts))

    def test_decider_model_mix_is_fifty_fifty(self):
        counts = Counter(self.sequence(orchestrate.DECIDER_MODEL_SHARES, 20, 6))
        self.assertEqual({"base": 10, "sol": 10}, dict(counts))

    def test_live_decider_model_absorbs_blocked_models_share(self):
        jobs = [{"model_lane": "base", "provider": "codex"}] * 7
        with mock.patch.object(orchestrate, "DECIDE_JOBS", jobs), \
                mock.patch.dict(orchestrate.DECIDER_MODEL_BLOCKED_UNTIL,
                                {"base": 0, "sol": float("inf")}, clear=True):
            # With both lanes healthy, base owns half of a 12-seat pool. If Sol is
            # unavailable, base may use the otherwise-idle remainder instead of stopping
            # at six.
            self.assertEqual(["base"], orchestrate._decider_model_order(12))

    def test_sol_share_is_spent_on_hardest_ready_task(self):
        ready = [("001", {}, {"score": 18}), ("002", {}, {"score": 81}),
                 ("003", {}, {"score": 49})]
        self.assertEqual(1, orchestrate._pick_decider_index(ready, "sol"))
        self.assertEqual(0, orchestrate._pick_decider_index(ready, "codex"))

    def test_decider_score_boundaries_choose_requested_models_and_efforts(self):
        row = {"task": "task001", "lane": "B", "bar_cost": "100"}

        def route(score, provider="codex"):
            difficulty = {"score": score, "tier": task_difficulty.tier_for_score(score),
                          "reasons": [], "signals": {}}
            with mock.patch.object(orchestrate, "_task_difficulty", return_value=difficulty):
                return orchestrate._decider_route("001", row, provider)

        # Wave-6 measured verdict: all codex decisions use flat gpt-5.5/xhigh.
        self.assertEqual(("gpt-5.5", "xhigh"),
                         (route(27)["model_id"], route(27)["effort"]))
        self.assertEqual(("gpt-5.5", "xhigh"),
                         (route(30)["model_id"], route(30)["effort"]))
        self.assertEqual(("gpt-5.5", "xhigh"),
                         (route(40)["model_id"], route(40)["effort"]))
        self.assertEqual(("gpt-5.5", "xhigh"),
                         (route(50)["model_id"], route(50)["effort"]))
        self.assertEqual(("gpt-5.5", "xhigh"),
                         (route(60)["model_id"], route(60)["effort"]))
        self.assertEqual(("gpt-5.5", "xhigh"),
                         (route(75)["model_id"], route(75)["effort"]))
        self.assertEqual(55, route(75)["benchmark_score"])
        self.assertEqual(58, route(75)["required_capability"])
        sol = orchestrate._decider_route(
            "001", row, "codex",
            difficulty={"score": 75, "tier": "VHARD", "reasons": [], "signals": {}},
            natt=20, model_lane="sol")
        self.assertEqual(("gpt-5.6-sol", "xhigh"),
                         (sol["model_id"], sol["effort"]))
        deep_base = orchestrate._decider_route(
            "001", row, "codex",
            difficulty={"score": 75, "tier": "VHARD", "reasons": [], "signals": {}},
            natt=20, model_lane="base")
        self.assertEqual(("gpt-5.5", "xhigh"),
                         (deep_base["model_id"], deep_base["effort"]))

    def test_builder_routes_follow_capability_ranking(self):
        # Wave-6 measured verdict: required capability remains visible for telemetry, but
        # every codex band deliberately routes to the proven flat gpt-5.5/xhigh config.
        expected = {
            0: (38, "gpt-5.5", "xhigh", 55),
            10: (46, "gpt-5.5", "xhigh", 55),
            20: (49, "gpt-5.5", "xhigh", 55),
            30: (52, "gpt-5.5", "xhigh", 55),
            40: (54, "gpt-5.5", "xhigh", 55),
            50: (55, "gpt-5.5", "xhigh", 55),
            60: (56, "gpt-5.5", "xhigh", 55),
            75: (58, "gpt-5.5", "xhigh", 55),
        }
        for difficulty, wanted in expected.items():
            route = orchestrate._builder_route("codex", difficulty)
            self.assertEqual(wanted, (route["required_capability"], route["model_id"],
                                      route["effort"], route["benchmark_score"]))
            self.assertEqual(55 - wanted[0], route["capability_gap"])

    def test_worker_task_order_follows_runtime_roi(self):
        queue = [{"task": "task001"}, {"task": "task002"}, {"task": "task003"}]
        yields = {"001": .04, "002": .01, "003": .03}
        def runtime_yield(task, _row, _provider, difficulty=None):
            return {"points_per_hour": yields[task]}
        with mock.patch.object(orchestrate, "SPAWN_COUNTS", {"codex": 0}), \
                mock.patch.object(orchestrate, "_runtime_yield",
                                  side_effect=runtime_yield), \
                mock.patch.object(orchestrate, "_priority", side_effect=lambda r, f: f + 1):
            self.assertEqual([0, 2, 1], orchestrate._worker_task_order(queue, "codex"))


class DifficultyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.results = self.root / "results"
        (self.results / "task001").mkdir(parents=True)
        self.ledger = self.results / "LEDGER.csv"
        self.ledger.write_text("task,cost,pts,delta,lane,sha8,notes\n")
        self.era = evidence.load_era()
        with open(self.results / "DIFFICULTY.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "task", "score", "tier", "pin_pts", "proven", "donor", "c_history"])
            writer.writeheader()
            writer.writerow({"task": "task001", "score": 30, "tier": "MEDIUM",
                             "pin_pts": 18.2, "proven": 0, "donor": 0.1,
                             "c_history": 20})

    def tearDown(self):
        self.tmp.cleanup()

    def test_old_nos_do_not_raise_difficulty_but_clean_nos_do(self):
        attempts = [
            {"ts": "20260710T100000", "model": "codex", "outcome": "NO",
             "fail_reason": "OLD FLOOR", "secs": 10},
            {"ts": "20260710T120000", "model": "codex", "outcome": "NO",
             "fail_reason": "COST-REJECT clean", "secs": 100},
        ]
        path = self.results / "task001" / "ATTEMPTS.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in attempts))
        profile = task_difficulty.profile(
            "001", {"task": "task001", "pin_cost": "100", "bar_cost": "86", "lane": "B"},
            str(self.results), str(self.ledger), self.era)
        self.assertEqual(1, profile["signals"]["clean_attempts"])
        self.assertEqual(1, profile["signals"]["clean_nos"])
        self.assertNotIn("OLD FLOOR", " ".join(profile["reasons"]))
        self.assertEqual(17, profile["score"])

    def test_weak_screen_is_context_not_difficulty_or_floor_evidence(self):
        task_difficulty.WEAK_SCREEN_MODEL_IDS = {"weak-screen-model"}
        self.addCleanup(lambda: task_difficulty.WEAK_SCREEN_MODEL_IDS.clear())
        attempts = [
            {"ts": "20260710T120000", "model": "codex", "model_id": "weak-screen-model",
             "outcome": "NO", "fail_reason": "weak screen", "secs": 100},
            {"ts": "20260710T121000", "model": "codex", "model_id": "gpt-5.5",
             "outcome": "NO", "fail_reason": "strong failure", "secs": 100},
        ]
        path = self.results / "task001" / "ATTEMPTS.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in attempts))
        strong = task_difficulty.current_strong_attempts(str(self.results), "001", self.era)
        self.assertEqual(1, len(strong))
        self.assertEqual("codex", strong[0]["model"])
        profile = task_difficulty.profile(
            "001", {"task": "task001", "pin_cost": "100", "bar_cost": "86", "lane": "B"},
            str(self.results), str(self.ledger), self.era)
        self.assertEqual(1, profile["signals"]["clean_attempts"])
        self.assertEqual(1, profile["signals"]["clean_nos"])
        self.assertEqual(1, profile["signals"]["failed_providers"])

    def test_closing_price_and_verified_rule_reduce_decider_spend(self):
        task_dir = self.results / "task001"
        (task_dir / "ATTEMPTS.jsonl").write_text(json.dumps({
            "ts": "20260710T120000", "model": "codex", "outcome": "DONE",
            "delta": 0.2, "fail_reason": "", "secs": 10}) + "\n")
        price = task_dir / "PRICES_att01.md"
        price.write_text("   80 B → 19.2 pts  [mechanical donor]\n")
        postmortem = task_dir / "POSTMORTEM_att01.md"
        postmortem.write_text("- RULE-STATUS: verified — ALL PASSED on generator evidence\n")
        current = self.era["cutoff_ts"] + 10
        for path in (price, postmortem):
            os.utime(path, (current, current))
        profile = task_difficulty.profile(
            "001", {"task": "task001", "pin_cost": "100", "bar_cost": "86", "lane": "B"},
            str(self.results), str(self.ledger), self.era)
        self.assertTrue(profile["signals"]["closing_price"])
        self.assertEqual("verified", profile["signals"]["rule"])
        self.assertEqual(0, profile["score"])

    def test_empirical_ev_does_not_let_speculative_hard_headroom_outrank_easy_yield(self):
        # 2026-07-12 ratio-EV model: the prize is ln(pin_cost/target_cost); at an EQUAL
        # prize, measured P(win) must dominate — a speculative HARD task never outranks
        # an EASY one on tier mystique alone (same intent as the old E[Δ] assertion).
        easy = rank_difficulty.expected_value(
            "EASY", 17.0, 0.0, 0, 0, False, natt=0, banked=0.0, pc=3000, fam="?")[0]
        hard = rank_difficulty.expected_value(
            "HARD", 14.0, 0.0, 0, 0, False, natt=0, banked=0.0, pc=3000, fam="?")[0]
        self.assertGreater(easy, hard)
        # A task already at/below its rule-class target has no prize left.
        done = rank_difficulty.expected_value(
            "EASY", 20.5, 0.0, 0, 0, False, natt=0, banked=0.0, pc=100, fam="?")[0]
        self.assertEqual(0.0, done)
        # Banked delta comes off the remaining ln-gain (max-ROI: no front-running).
        banked = rank_difficulty.expected_value(
            "EASY", 17.0, 0.0, 0, 0, False, natt=0, banked=0.9, pc=3000, fam="?")[0]
        self.assertLess(banked, easy)
        # Unsupported textual donor similarity alone must not change EV.
        donor = rank_difficulty.expected_value(
            "EASY", 17.0, 0.9, 0, 0, False, natt=0, banked=0.0, pc=3000, fam="?")[0]
        self.assertEqual(easy, donor)


if __name__ == "__main__":
    unittest.main()
