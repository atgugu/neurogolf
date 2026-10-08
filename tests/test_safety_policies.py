import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import bank
import auto_submit
import evidence
import make_pack
import orchestrate
import promote_proven
import proven_corpus
import submit_result
import gate_vs_pin


class SubmissionAcceptanceTests(unittest.TestCase):
    def test_autosubmit_cursor_sees_lower_delta_after_absorbed_historical_best(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "LEDGER.csv"
            ledger.write_text(
                "task,cost,pts,delta,lane,sha8,notes\n"
                "task001,10,19.0,+1.0,B,oldold00,absorbed\n"
                "task001,20,18.3,+0.3,B,newnew00,current\n")
            old = auto_submit.LEDGER
            auto_submit.LEDGER = ledger
            try:
                total, lines, _ = auto_submit.ledger_best_sum()
                pending, _, _ = auto_submit.ledger_best_sum(after_lines=1)
            finally:
                auto_submit.LEDGER = old
            self.assertEqual(1.0, total)
            self.assertEqual(2, lines)
            self.assertEqual(0.3, pending)

    def test_autosubmit_plan_sums_unique_tasks_not_overlapping_bundle_lines(self):
        plan = [
            "wave_01_SOLO_idabc_pred10.20.zip: 001+0.2000",
            "wave_02_2sw_iddef_pred10.50.zip: 002+0.3000 003+0.2000",
            "wave_ALL_2m_idghi_pred10.50.zip: 002+0.3000 003+0.2000",
        ]
        self.assertEqual(0.7, auto_submit.planned_delta(plan))

    def test_autosubmit_cycle_requires_every_zip_pid_and_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first.zip"
            second = Path(td) / "second.zip"
            first.write_bytes(b"first immutable bundle")
            second.write_bytes(b"second immutable bundle")
            cycle = auto_submit.make_cycle(3.4, 48, "pin12345", [first, second], 0.7)

            self.assertEqual(2, len(auto_submit.cycle_unresolved(cycle, {})))
            cycle["ships"][0].update(submitted=True, pid=101)
            cycle["ships"][1].update(submitted=True, pid=102)
            pending = auto_submit.cycle_unresolved(cycle, {101: {"actual": 10.2}})
            self.assertEqual(1, len(pending))
            self.assertIn("outcome-pending", pending[0])
            self.assertEqual([], auto_submit.cycle_unresolved(
                cycle, {101: {"actual": 10.2}, 102: {"actual": 10.3}}))

    def test_autosubmit_cycle_recovers_exact_pid_after_process_restart(self):
        cycle = {
            "version": 1, "cycle_id": "cycle1", "pin_fence": "pin12345",
            "ships": [{"path": "/tmp/a.zip", "zip_sha": "deadbeef",
                       "submitted": False, "pid": None}],
            "union": {"path": "/tmp/union.zip", "zip_sha": "feedface",
                      "submitted": False, "pid": None},
        }
        entries = [
            {"id": 40, "zip_sha": "deadbeef", "base_sha": "otherpin"},
            {"id": "41", "zip_sha": "deadbeef", "base_sha": "pin12345"},
            {"id": 42, "zip_sha": "feedface", "base_sha": "pin12345"},
        ]
        self.assertTrue(auto_submit.recover_cycle_pids(cycle, entries))
        self.assertTrue(cycle["ships"][0]["submitted"])
        self.assertEqual(41, cycle["ships"][0]["pid"])
        self.assertTrue(cycle["union"]["submitted"])
        self.assertEqual(42, cycle["union"]["pid"])

    def test_autosubmit_cycle_journal_and_marker_are_transaction_bound(self):
        with tempfile.TemporaryDirectory() as td:
            old_cycle, old_marker = auto_submit.CYCLE, auto_submit.MARKER
            auto_submit.CYCLE = Path(td) / "cycle.json"
            auto_submit.MARKER = Path(td) / "marker.json"
            cycle = {"version": 1, "cycle_id": "abc123", "pin_fence": "oldpin",
                     "sum_best": 3.4, "ledger_lines": 48, "ships": []}
            try:
                auto_submit.write_cycle(cycle)
                self.assertEqual(cycle, auto_submit.read_cycle())
                auto_submit.write_marker(3.4, 48, "newpin", True,
                                         cycle_id=cycle["cycle_id"])
                marker = auto_submit.read_marker()
            finally:
                auto_submit.CYCLE, auto_submit.MARKER = old_cycle, old_marker
            self.assertEqual("abc123", marker["cycle_id"])
            self.assertEqual(48, marker["ledger_lines"])
            self.assertEqual("newpin", marker["pin_sha"])

    def test_autosubmit_cycle_detects_mutated_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "bundle.zip"
            bundle.write_bytes(b"original")
            cycle = auto_submit.make_cycle(1.0, 2, "pin12345", [bundle])
            self.assertEqual([], auto_submit.cycle_file_errors(cycle))
            bundle.write_bytes(b"mutated")
            errors = auto_submit.cycle_file_errors(cycle)
            self.assertEqual(1, len(errors))
            self.assertIn("mutated", errors[0])

    def test_vs_pin_uses_grader_zero_padding_not_channel_zero_padding(self):
        x = gate_vs_pin.onehot([[0, 2], [3, 0]])
        self.assertEqual((1, 10, 30, 30), x.shape)
        self.assertEqual(1.0, x[0, 0, 0, 0])
        self.assertEqual(1.0, x[0, 2, 0, 1])
        self.assertEqual(0.0, float(x[0, :, 2:, :].sum()))
        self.assertEqual(0.0, float(x[0, :, :, 2:].sum()))

    def test_vs_pin_success_is_terminal_and_never_falls_through_to_full_gate(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "task118.onnx"
            model.write_bytes(b"candidate")
            ok = mock.Mock(returncode=0)
            with mock.patch.object(submit_result.subprocess, "run", return_value=ok) as run:
                route = submit_result.verify_acceptance(
                    118, str(model), bar="0.15", vs_pin=True, gated=False)
            self.assertEqual("VSPIN-CANARY", route)
            self.assertEqual(1, run.call_count)
            self.assertIn("gate_vs_pin.py", run.call_args.args[0][1])
            self.assertNotIn("fast_verify.py", " ".join(run.call_args.args[0]))

    def test_full_gate_marker_is_sha_bound_and_missing_marker_regates(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "task001.onnx"
            model.write_bytes(b"candidate")
            sha8 = hashlib.sha256(model.read_bytes()).hexdigest()[:8]
            (Path(td) / f".fullgate_{sha8}").write_text(str(model))
            with mock.patch.object(submit_result.subprocess, "run") as run:
                route = submit_result.verify_acceptance(
                    1, str(model), bar="0.05", gated=True)
            self.assertEqual("FULL-GATE", route)
            run.assert_not_called()

            (Path(td) / f".fullgate_{sha8}").unlink()
            ok = mock.Mock(returncode=0)
            with mock.patch.object(submit_result.subprocess, "run", return_value=ok) as run:
                route = submit_result.verify_acceptance(
                    1, str(model), bar="0.05", gated=True)
            self.assertEqual("FULL-GATE", route)
            self.assertIn("fast_verify.py", run.call_args.args[0][1])
            self.assertEqual("0.05", run.call_args.args[0][-1])

    def test_every_ambiguous_task_is_defense_in_depth_canary(self):
        self.assertTrue({"002", "118", "187", "255"}.issubset(bank.CANARY_TASKS))


class OrchestratorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.results = self.base / "results"
        self.results.mkdir()
        self.old_results = orchestrate.RESULTS
        self.old_ledger = orchestrate.LEDGER
        self.old_lessons = orchestrate.LESSONS
        orchestrate.RESULTS = str(self.results)
        orchestrate.LEDGER = str(self.results / "LEDGER.csv")
        orchestrate.LESSONS = str(self.results / "LESSONS.md")
        Path(orchestrate.LEDGER).write_text("task,cost,pts,delta,lane,sha8,notes\n")
        Path(orchestrate.LESSONS).write_text("# clean lessons\n")
        self.cutoff = evidence.load_era()["cutoff_ts"]

    def tearDown(self):
        orchestrate.RESULTS = self.old_results
        orchestrate.LEDGER = self.old_ledger
        orchestrate.LESSONS = self.old_lessons
        self.tmp.cleanup()

    def _task_dir(self, task="001"):
        path = self.results / f"task{task}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _mtime(self, path, current):
        stamp = self.cutoff + (10 if current else -10)
        os.utime(path, (stamp, stamp))

    def test_pre_fix_history_cannot_frame_house_money_or_enter_decider(self):
        td = self._task_dir()
        attempts = [
            {"ts": "20260710T100000", "model": "codex", "outcome": "DONE",
             "delta": 9.99, "fail_reason": "PHANTOM_ATTEMPT", "secs": 100},
            {"ts": "20260710T120000", "model": "codex", "outcome": "NO",
             "delta": None, "fail_reason": "CLEAN_FAIL", "secs": 100},
        ]
        (td / "ATTEMPTS.jsonl").write_text("".join(json.dumps(x) + "\n" for x in attempts))
        old_decision = td / "DECISION_att01.md"
        old_decision.write_text("DECISION: PHANTOM_PLAN\n")
        old_pm = td / "POSTMORTEM_att01.md"
        old_pm.write_text("PHANTOM_POSTMORTEM" * 20)
        old_notes = td / "NOTES.md"
        old_notes.write_text("PHANTOM_NOTES")
        old_no = td / "NO.md"
        old_no.write_text("PHANTOM_FLOOR")
        for path in (old_decision, old_pm, old_notes, old_no):
            self._mtime(path, False)
        clean_pm = td / "POSTMORTEM_att02.md"
        clean_pm.write_text("CLEAN_POSTMORTEM " * 20)
        self._mtime(clean_pm, True)

        history = orchestrate.history_digest("001")
        self.assertNotIn("HOUSE MONEY", history)
        self.assertNotIn("PHANTOM", history)
        self.assertIn("CLEAN_FAIL", history)
        self.assertIn("CLEAN_POSTMORTEM", history)

        prompt, natt = orchestrate.build_decider_prompt(
            "001", {"bar_cost": "100", "lane": "B"})
        self.assertEqual(3, natt)
        self.assertNotIn("PHANTOM", prompt)
        self.assertIn("CLEAN_FAIL", prompt)
        self.assertIn("CLEAN_POSTMORTEM", prompt)

    def test_clean_ledger_is_the_only_house_money_source(self):
        td = self._task_dir()
        (td / "ATTEMPTS.jsonl").write_text(
            json.dumps({"ts": "20260710T100000", "model": "codex",
                        "outcome": "DONE", "delta": 8.0,
                        "fail_reason": "", "secs": 10}) + "\n")
        with open(orchestrate.LEDGER, "a", newline="") as f:
            csv.writer(f).writerow(["task001", 10, 19.0, "+0.25", "B", "deadbeef", "clean"])
        history = orchestrate.history_digest("001")
        self.assertIn("HOUSE MONEY", history)
        self.assertIn("Δ+0.25", history)
        self.assertNotIn("8.0", history)

    def test_absorbed_ledger_rows_do_not_frame_current_house_money(self):
        td = self._task_dir()
        with open(orchestrate.LEDGER, "a", newline="") as f:
            csv.writer(f).writerow(["task001", 10, 19.0, "+0.80", "B", "oldold00", "absorbed"])
            csv.writer(f).writerow(["task001", 9, 19.1, "+0.20", "B", "newnew00", "pending"])
        data = self.base / "data"
        data.mkdir()
        (data / "PIN_SHA.txt").write_text("current1")
        (self.results / "AUTOSUBMIT_MARKER.json").write_text(json.dumps({
            "ledger_lines": 1, "pin_sha": "current1"}))
        self.assertEqual(0.20, evidence.banked_delta(
            "001", orchestrate.LEDGER, evidence.load_era()))
        history = orchestrate.history_digest("001")
        self.assertIn("Δ+0.2", history)
        self.assertNotIn("0.80", history)

    def test_done_requires_current_attempt_exact_sha_acceptance_receipt(self):
        td = self._task_dir("007")
        artifact = td / "task007.onnx"
        artifact.write_bytes(b"accepted-bytes")
        full = hashlib.sha256(artifact.read_bytes()).hexdigest()
        receipts = td / "ACCEPTANCE.jsonl"
        receipts.write_text(json.dumps({
            "task": 7, "accepted_epoch": 100.0, "sha256": full, "delta": 0.2}) + "\n")
        rec, path = orchestrate._acceptance_for_attempt("007", 101.0)
        self.assertIsNone(rec)
        self.assertIsNone(path)

        with receipts.open("a") as f:
            f.write(json.dumps({
                "task": 7, "accepted_epoch": 102.0, "sha256": full, "delta": 0.2}) + "\n")
        rec, path = orchestrate._acceptance_for_attempt("007", 101.0)
        self.assertEqual(0.2, rec["delta"])
        self.assertEqual(str(artifact), path)

        artifact.write_bytes(b"mutated-after-receipt")
        rec, path = orchestrate._acceptance_for_attempt("007", 101.0)
        self.assertIsNone(rec)
        self.assertIsNone(path)

    def test_policy_rejection_is_quarantined_once_and_can_seed_fresh_decider(self):
        td = self._task_dir("002")
        active = self.base / "build.py"
        active.write_text("# actively edit scorer matcher to skip this tensor\n")
        with self.assertRaises(orchestrate.PolicyRejected):
            orchestrate._enforce_policy_inputs("002", [str(active)])
        self.assertFalse(active.exists())
        quarantined = list((td / "quarantine").iterdir())
        self.assertEqual(1, len(quarantined))
        records = [json.loads(x) for x in (td / "REJECTIONS.jsonl").read_text().splitlines()]
        self.assertEqual(1, len(records))
        self.assertIn("scorer", records[0]["reason"].lower())

        # The unchanged path is gone, so a second preflight cannot loop on it.
        orchestrate._enforce_policy_inputs("002", [str(active)])
        self.assertEqual(1, len(list((td / "quarantine").iterdir())))
        prompt, _ = orchestrate.build_decider_prompt("002", {"bar_cost": "100", "lane": "B"})
        self.assertIn("REJECTED INPUTS", prompt)
        self.assertIn("scorer", prompt.lower())

    def test_rejected_decision_is_replaced_without_reusing_its_bytes(self):
        td = self._task_dir("004")
        decision = td / "DECISION_att01.md"
        decision.write_text(
            "DECISION: pivot-family\n"
            "BUILD-RECIPE: edit scorer matcher to skip the charged tensor\n"
            'FAMILY-BUDGET-JSON: {"tensors":[],"params":1}\n')
        self._mtime(decision, True)
        with self.assertRaises(orchestrate.PolicyRejected):
            orchestrate._enforce_policy_inputs("004", [str(decision)])
        rejected_bytes = next((td / "quarantine").iterdir()).read_bytes()
        self.assertIn(b"edit scorer matcher", rejected_bytes)

        decision.write_text(
            "DECISION: pivot-family\n"
            "PRICED-EXPLORATORY-BUILD: NO\n"
            "BUILD-RECIPE: 1. build a static uint8 tensor\n"
            'FAMILY-BUDGET-JSON: {"tensors":[{"shape":[2],"dtype":"u8"}],"params":1}\n')
        self._mtime(decision, True)
        self.assertEqual("READY",
                         orchestrate._stamp_decision_budget(
                             str(decision), {"bar_cost": "10"})[0])
        orchestrate._enforce_policy_inputs("004", [str(decision)])
        self.assertEqual(1, len(list((td / "quarantine").iterdir())))

    def test_budget_states_gate_workers_without_creating_floor(self):
        td = self._task_dir("003")
        row = {"bar_cost": "10"}

        missing = td / "missing.md"
        missing.write_text("DECISION: pivot-family\n")
        status, _, _ = orchestrate._stamp_decision_budget(str(missing), row)
        self.assertEqual("REVISE", status)
        self.assertEqual("REVISE", orchestrate._decision_budget_status(str(missing))[0])

        malformed = td / "malformed.md"
        malformed.write_text("FAMILY-BUDGET-JSON: {not json}\n")
        self.assertEqual("REVISE", orchestrate._stamp_decision_budget(str(malformed), row)[0])

        closes = td / "closes.md"
        closes.write_text('FAMILY-BUDGET-JSON: {"tensors":[{"shape":[2],"dtype":"u8"}],"params":1}\n')
        self.assertEqual("READY", orchestrate._stamp_decision_budget(str(closes), row)[0])

        nonclosing = td / "nonclosing.md"
        nonclosing.write_text('FAMILY-BUDGET-JSON: {"tensors":[{"shape":[20],"dtype":"u8"}],"params":0}\n')
        self.assertEqual("REVISE", orchestrate._stamp_decision_budget(str(nonclosing), row)[0])

        exploratory = td / "exploratory.md"
        exploratory.write_text(
            "PRICED-EXPLORATORY-BUILD: YES — tests a hidden-set hypothesis\n"
            'FAMILY-BUDGET-JSON: {"tensors":[{"shape":[20],"dtype":"u8"}],"params":0}\n')
        self.assertEqual("EXPLORATORY",
                         orchestrate._stamp_decision_budget(str(exploratory), row)[0])
        self.assertFalse((td / "ATTEMPTS.jsonl").exists())
        self.assertFalse((td / "FLOOR.md").exists())

    def test_pack_dossiers_respect_the_same_evidence_cutoff(self):
        dossier = self.base / "dossier.md"
        dossier.write_text("OLD_FLOOR_CLAIM")
        self._mtime(dossier, False)
        self.assertIn("withheld", make_pack.read_current_excerpt(str(dossier)))
        self.assertNotIn("OLD_FLOOR_CLAIM", make_pack.read_current_excerpt(str(dossier)))
        dossier.write_text("CLEAN_GENERATOR_FINDING")
        self._mtime(dossier, True)
        self.assertIn("CLEAN_GENERATOR_FINDING",
                      make_pack.read_current_excerpt(str(dossier)))


class ProvenBankingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.corpus = self.base / "corpus"
        self.corpus.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _candidate(self, name, data):
        path = self.base / name
        path.write_bytes(data)
        return path

    def _install_proven(self, task, data, pts=1.0):
        full = hashlib.sha256(data).hexdigest()
        key = full[:8]
        td = self.corpus / f"task{task:03d}"
        td.mkdir(exist_ok=True)
        (td / f"{key}.onnx").write_bytes(data)
        mp = self.corpus / "manifest.json"
        manifest = json.loads(mp.read_text()) if mp.exists() else {}
        manifest[key] = {"task": task, "pts": pts, "source": "solo proof",
                         "sha256": full}
        mp.write_text(json.dumps(manifest))
        return key

    def _proof(self, task, path):
        return proven_corpus.payment_proof_for_path(
            task, str(path), str(self.corpus / "manifest.json"), str(self.corpus))

    def test_unproven_is_solo_exact_proven_may_chunk_and_one_byte_revokes(self):
        p1 = self._candidate("p1.onnx", b"proven-one")
        p2 = self._candidate("p2.onnx", b"proven-two")
        new = self._candidate("new.onnx", b"new-candidate")
        low = self._candidate("low.onnx", b"low-candidate")
        self._install_proven(1, p1.read_bytes())
        self._install_proven(2, p2.read_bytes())
        members = [
            {"t": "001", "path": str(p1), "canary": False, "delta": 0.2,
             "payment_proof": self._proof(1, p1)},
            {"t": "002", "path": str(p2), "canary": False, "delta": 0.3,
             "payment_proof": self._proof(2, p2)},
            {"t": "003", "path": str(new), "canary": False, "delta": 1.51,
             "payment_proof": self._proof(3, new)},
            {"t": "005", "path": str(low), "canary": False, "delta": 1.5,
             "payment_proof": self._proof(5, low)},
        ]
        chunks, solos, groupable, held = bank.shipping_chunks(members, 12, 1.5)
        self.assertEqual([["003"], ["001", "002"]],
                         [[m["t"] for m in chunk] for chunk in chunks])
        self.assertEqual(1, len(solos))
        self.assertEqual(2, len(groupable))
        self.assertEqual(["005"], [m["t"] for m in held])

        # A one-member proven remainder is still a solo submission and is held too.
        only_proven = [dict(members[0])]
        only_chunks, only_solos, _, only_held = bank.shipping_chunks(
            only_proven, 12, 1.5)
        self.assertEqual([], only_chunks)
        self.assertEqual([], only_solos)
        self.assertEqual(["001"], [m["t"] for m in only_held])
        with self.assertRaisesRegex(ValueError, "low-value solo"):
            bank.validate_chunks([[only_proven[0]]], 1.5)
        with self.assertRaisesRegex(ValueError, "cannot be lowered"):
            bank.shipping_chunks(members, 12, -0.01)

        # Point-max default: a positive unproven member is emitted as an attributable solo.
        positive = dict(members[3], delta=0.05)
        chunks0, solos0, _, held0 = bank.shipping_chunks([positive], 12)
        self.assertEqual([[positive]], chunks0)
        self.assertEqual([positive], solos0)
        self.assertEqual([], held0)

        p1.write_bytes(b"proven-one!")
        self.assertIsNone(self._proof(1, p1))
        with self.assertRaises(ValueError):
            bank.validate_chunks([[members[0], members[2]]])

    def test_bank_recomputes_artifact_risk_instead_of_trusting_truncated_notes(self):
        candidate = self._candidate("long-note.onnx", b"unproven-risky")
        member = {"t": "202", "path": str(candidate), "canary": False,
                  "delta": 1.0, "lane": "B"}
        with mock.patch.object(bank, "payment_proof_for_path", return_value=None), \
             mock.patch.object(bank, "hidden_oracle_route",
                               return_value=("SOLO_PROBE", ["unmeasured-op:Einsum"])):
            bank.classify_artifact_risk(member)
        self.assertTrue(member["canary"])
        self.assertEqual("SOLO_PROBE", member["oracle_route"])

        # Exact per-member Kaggle payment proof makes the same bytes batch-safe.
        member["canary"] = False
        with mock.patch.object(bank, "payment_proof_for_path", return_value={"pid": 9}), \
             mock.patch.object(bank, "hidden_oracle_route",
                               return_value=("SOLO_PROBE", ["unmeasured-op:Einsum"])):
            bank.classify_artifact_risk(member)
        self.assertFalse(member["canary"])

    def test_ambiguous_regate_uses_vs_pin_not_the_known_incompatible_full_gate(self):
        cmd = bank.regate_command({"t": "255", "path": "/tmp/task255.onnx"}, "0.15")
        self.assertIn("gate_vs_pin.py", cmd[1])
        self.assertNotIn("fast_verify.py", " ".join(cmd))
        normal = bank.regate_command({"t": "202", "path": "/tmp/task202.onnx"}, "0.15")
        self.assertIn("fast_verify.py", normal[1])
        self.assertIn("--full", normal)

    def test_promotion_requires_reconciled_solo_probe(self):
        candidate = self._candidate("candidate.onnx", b"paid-model")
        probe_zip = self.base / "solo.zip"
        with zipfile.ZipFile(probe_zip, "w") as z:
            z.writestr("task004.onnx", candidate.read_bytes())
        log = self.base / "probe_log.jsonl"
        entry = {"id": 7, "ok": True, "zip": str(probe_zip), "base_score": 100.0,
                 "predicted": 101.0,
                 "swaps": [{"task": 4, "old": 10.0, "new": 11.0, "delta": 1.0}]}
        log.write_text(json.dumps(entry) + "\n")
        outcomes = self.base / "outcomes.json"
        outcomes.write_text(json.dumps({"7": {"actual": 101.0, "gap": 0.0}}))

        key, target, record = promote_proven.promote(
            4, str(candidate), 7, str(self.corpus), str(log), str(outcomes))
        self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest()[:8], key)
        self.assertTrue(Path(target).exists())
        self.assertEqual(11.0, record["pts"])
        self.assertIsNotNone(self._proof(4, candidate))

        bad_entry = dict(entry)
        bad_entry["id"] = 8
        bad_entry["swaps"] = entry["swaps"] * 2
        with open(log, "a") as f:
            f.write(json.dumps(bad_entry) + "\n")
        outcomes.write_text(json.dumps({"7": {"actual": 101.0},
                                        "8": {"actual": 101.0}}))
        with self.assertRaisesRegex(ValueError, "solo probe"):
            promote_proven.promote(
                4, str(candidate), 8, str(self.corpus), str(log), str(outcomes))


if __name__ == "__main__":
    unittest.main()
