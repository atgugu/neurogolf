import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import orchestrate


class ABArmTests(unittest.TestCase):
    def test_arm_split_mechanism_remains_deterministic_when_enabled(self):
        with mock.patch.object(orchestrate, "GPT55_SHARE", 0.5):
            a1 = [orchestrate._ab_arm_is_gpt55(f"{t:03d}", n)
                  for t in range(120) for n in range(1, 4)]
            a2 = [orchestrate._ab_arm_is_gpt55(f"{t:03d}", n)
                  for t in range(120) for n in range(1, 4)]
        self.assertEqual(a1, a2)  # deterministic — reproducible attribution
        frac = sum(a1) / len(a1)
        self.assertGreater(frac, 0.42)
        self.assertLess(frac, 0.58)

    def test_wave6_default_disables_completed_ab_experiment(self):
        self.assertEqual(0.0, orchestrate.GPT55_SHARE)
        self.assertFalse(any(orchestrate._ab_arm_is_gpt55("133", n)
                             for n in range(1, 30)))

    def test_next_model_routes_every_codex_attempt_to_gpt55_xhigh(self):
        row = {"task": "task133", "lane": "B"}
        difficulty = {"score": 40, "tier": "MEDIUM", "reasons": [], "signals": {}}
        with mock.patch.object(orchestrate, "_task_difficulty", return_value=difficulty), \
             mock.patch.object(orchestrate, "_decider_rung", return_value=None):
            seen = set()
            for att_hint in range(12):
                atts = [{"ts": "20260711T120000", "model": "codex", "model_id": "gpt-5.6-sol",
                         "effort": "high", "outcome": "NO", "fail_reason": "", "secs": 100}
                        ] * att_hint
                _, eff, mid = orchestrate.next_model(row, atts, provider="codex")
                seen.add((mid, eff))
            self.assertEqual({("gpt-5.5", "xhigh")}, seen)


class WeakModelQuarantineTests(unittest.TestCase):
    WEAK_NO = {"ts": "20260711T120000", "model": "codex", "model_id": "weak-screen-model",
               "effort": "high", "outcome": "NO", "fail_reason": "COST-REJECT", "secs": 900}
    CODEX_NO = {"ts": "20260711T121000", "model": "codex", "model_id": "gpt-5.5",
                "effort": "xhigh", "outcome": "NO", "fail_reason": "gate FAIL", "secs": 900}

    def setUp(self):
        self._weak = set(orchestrate.WEAK_MODELS)
        orchestrate.WEAK_MODELS.clear()
        orchestrate.WEAK_MODELS.add("weak-screen-model")

    def tearDown(self):
        orchestrate.WEAK_MODELS.clear()
        orchestrate.WEAK_MODELS.update(self._weak)

    def test_weak_att_classification(self):
        self.assertTrue(orchestrate._weak_att(self.WEAK_NO))
        self.assertFalse(orchestrate._weak_att(self.CODEX_NO))

    def test_strong_atts_excludes_screen_attempts(self):
        strong = orchestrate.strong_atts([self.WEAK_NO, self.CODEX_NO, self.WEAK_NO])
        self.assertEqual([self.CODEX_NO], strong)

    def test_floor_and_stoploss_semantics_ignore_weak_nos(self):
        # replicate the main-loop verdict math on a weak-screen NO + one strong NO:
        # NOT floored (needs 2 strong NOs from 2 models), n_real == 1 (no stop-loss)
        clean_real = orchestrate.strong_atts([self.WEAK_NO, self.CODEX_NO])
        nos = [x for x in clean_real if x["outcome"] == "NO"]
        floored = (len(nos) >= 2 and len({x["model"] for x in nos}) >= 2) or len(nos) >= 3
        self.assertFalse(floored)
        self.assertEqual(1, len(clean_real))



class FleetControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fleet = Path(self.tmp.name) / "FLEET.json"
        self._file = orchestrate.FLEET_FILE
        self._cache = dict(orchestrate._FLEET_CACHE)
        orchestrate.FLEET_FILE = str(self.fleet)
        orchestrate._FLEET_CACHE.update(mtime=None, desired=None)

    def tearDown(self):
        orchestrate.FLEET_FILE = self._file
        orchestrate._FLEET_CACHE.update(self._cache)
        self.tmp.cleanup()

    def _write(self, d):
        self.fleet.write_text(json.dumps(d))  # st_mtime_ns+size cache key detects rewrites

    def test_missing_file_means_legacy_mode(self):
        self.assertIsNone(orchestrate.fleet_desired())

    def test_resize_is_picked_up_and_orders_by_deficit(self):
        self._write({"codex": 2})
        self.assertEqual({"codex": 2}, orchestrate.fleet_desired())
        running = [{"model": "codex"}, {"model": "codex"}]  # codex full
        order = orchestrate._worker_provider_order(running, dry=True)
        self.assertNotIn("codex", order)          # at desired → drain/no-spawn
        self._write({"codex": 4})                 # operator resize mid-run
        order = orchestrate._worker_provider_order(running, dry=True)
        self.assertIn("codex", order)             # new seats spawn next tick

    def test_codex_builder_and_decider_share_one_pool(self):
        self._write({"codex": 12})
        builders = [{"model": "codex"}] * 6
        deciders = [{"provider": "codex"}] * 6
        order = orchestrate._worker_provider_order(
            builders, dry=True, decider_jobs=deciders)
        self.assertNotIn("codex", order)
        with mock.patch.object(orchestrate, "DECIDE_JOBS", deciders):
            self.assertEqual(6, orchestrate._decider_concurrency_limit(builders))

    def test_malformed_file_keeps_last_good_value(self):
        self._write({"codex": 3})
        self.assertEqual({"codex": 3}, orchestrate.fleet_desired())
        self.fleet.write_text("{not json")
        os.utime(self.fleet, (os.path.getmtime(self.fleet) + 2,) * 2)
        self.assertEqual({"codex": 3}, orchestrate.fleet_desired())

    def test_timeout_resume_does_not_wedge_on_a_zero_seat_provider(self):
        self._write({"codex": 0})
        timeout = [{"ts": "20260711T120000", "model": "codex", "model_id": "gpt-5.5",
                    "effort": "xhigh", "outcome": "TIMEOUT", "fail_reason": "", "secs": 900}]
        self.assertEqual("codex", orchestrate._resume_provider(timeout))
        self.assertFalse(orchestrate._worker_available("codex"))


class FinalWaveTests(unittest.TestCase):
    """Final-sprint levers: FAST tier, independent decider pool, sprint cooldown clamp,
    group wind-down, cold-start. Every knob defaults off so the tests above still
    describe the shipped default behavior."""

    def test_codex_fast_tier_only_for_listed_models(self):
        with mock.patch.object(orchestrate, "CODEX_FAST_MODELS", {"gpt-5.6-sol"}):
            sol = " ".join(orchestrate.MODELS["codex"]["cmd"]("P", "/tmp", "xhigh", "gpt-5.6-sol"))
            g55 = " ".join(orchestrate.MODELS["codex"]["cmd"]("P", "/tmp", "xhigh", "gpt-5.5"))
        self.assertIn('service_tier="priority"', sol)
        self.assertNotIn("service_tier", g55)

    def test_retry_cooldown_cap_clamps_and_default_is_unchanged(self):
        with mock.patch.dict(os.environ, {"RETRY_COOLDOWN_CAP": "600"}):
            self.assertEqual(600, orchestrate._retry_cooldown(4, False))
            self.assertEqual(600, orchestrate._retry_cooldown(4, True))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RETRY_COOLDOWN_CAP", None)
            self.assertEqual(3600, orchestrate._retry_cooldown(4, False))
            self.assertEqual(7200, orchestrate._retry_cooldown(4, True))

    def test_decider_pool_independent_ignores_builder_occupancy(self):
        with mock.patch.dict(os.environ, {"DECIDER_POOL_INDEPENDENT": "1"}):
            self.assertEqual(orchestrate.DECIDER_MAX_CONC,
                             orchestrate._decider_concurrency_limit([{"model": "codex"}] * 12))

    def test_arm_and_decider_lane_wind_down(self):
        with mock.patch.object(orchestrate, "_FLEET_CACHE", {"off": frozenset(["sol"])}):
            self.assertTrue(all(orchestrate._ab_arm_is_gpt55(f"{t:03d}", 1) for t in range(20)))
            self.assertTrue(orchestrate._decider_lane_off("sol"))
            self.assertFalse(orchestrate._decider_lane_off("base"))
        with mock.patch.object(orchestrate, "_FLEET_CACHE", {"off": frozenset(["g55"])}):
            self.assertFalse(any(orchestrate._ab_arm_is_gpt55(f"{t:03d}", 1) for t in range(20)))
        with mock.patch.object(orchestrate, "_FLEET_CACHE", {"off": frozenset(["openai"])}):
            self.assertTrue(orchestrate._decider_lane_off("sol"))
            self.assertTrue(orchestrate._decider_lane_off("base"))

    def test_sprint_cold_start_defaults_off(self):
        self.assertFalse(orchestrate.SPRINT_COLD_START)


class FleetGroupOffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fleet = Path(self.tmp.name) / "FLEET.json"
        self._file = orchestrate.FLEET_FILE
        self._cache = dict(orchestrate._FLEET_CACHE)
        orchestrate.FLEET_FILE = str(self.fleet)
        orchestrate._FLEET_CACHE.clear()
        orchestrate._FLEET_CACHE.update(mtime=None, desired=None)

    def tearDown(self):
        orchestrate.FLEET_FILE = self._file
        orchestrate._FLEET_CACHE.clear()
        orchestrate._FLEET_CACHE.update(self._cache)
        self.tmp.cleanup()

    def test_off_openai_masks_codex_but_preserves_count(self):
        self.fleet.write_text(json.dumps({"codex": 12, "off": ["openai"]}))
        self.assertEqual(0, orchestrate.fleet_desired()["codex"])
        self.assertIn("openai", orchestrate.fleet_models_off())
        self.fleet.write_text(json.dumps({"codex": 12, "off": []}))
        self.assertEqual(12, orchestrate.fleet_desired()["codex"])

    def test_no_off_list_is_backward_compatible(self):
        self.fleet.write_text(json.dumps({"codex": 12}))
        self.assertEqual({"codex": 12}, orchestrate.fleet_desired())
        self.assertEqual(frozenset(), orchestrate.fleet_models_off())


if __name__ == "__main__":
    unittest.main()
