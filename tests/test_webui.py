import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "runner"))

import orchestrate
import webui


class ParserTests(unittest.TestCase):
    def test_legacy_status_and_blocked_provider(self):
        state = webui.parse_legacy_status(
            "20260710T093037 running=2 queue=123 done=24 cooling=47 analysts=1 "
            "deciders=7 builds[codex=72] infra_requeues=1 "
            "blocked=['codex'] ledger_sum=+64.22 stop=0\n")
        self.assertEqual(state["counts"]["running"], 2)
        self.assertEqual(state["counts"]["queued"], 123)
        self.assertEqual(state["spawn_counts"], {"codex": 72})
        self.assertEqual(state["blocked_models"], {"codex": None})
        self.assertEqual(state["ledger_sum"], 64.22)

    def test_live_price_is_qualified_not_mistaken_for_a_win(self):
        text = """
PRICE task004: cost=3586 (params=94 mem=3492) pts=16.8152 pin=16.6212(cost 4354) Δ=+0.1941
FAIL wrong(train#0/p1)
PRICE task004: cost=3200 (params=80 mem=3120) pts=16.9300 pin=16.6212(cost 4354) Δ=+0.3088
ALL PASSED full gate
"""
        signals = webui.price_signals(text)
        self.assertEqual(signals[0]["qualification"], "wrong")
        self.assertEqual(signals[1]["qualification"], "passing")
        self.assertAlmostEqual(signals[1]["delta"], 0.3088)


class MonitorFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        for directory in ("data", "data/lanes", "results", "results/task001", "claims"):
            os.makedirs(os.path.join(root, directory), exist_ok=True)
        self.write("data/EVIDENCE_ERA.json", json.dumps({
            "era": "fixed-scorer-v1", "cutoff_utc": "2026-07-10T11:17:26Z",
            "clean_ledger": "results/LEDGER.csv"}))
        self.write("data/PIN_SHA.txt", "deadbeef\n")
        self.write("data/task_map.json", '{"task001":"arc00001","task002":"arc00002"}')
        self.write("data/pin_pv2_70_263_pertask.csv",
                   "task,params,memory,cost,score,status,method\n"
                   "task001,1,99,100,19.0,ok,ort\n"
                   "task002,1,199,200,18.0,ok,ort\n")
        self.write("data/lanes/QUEUE.csv",
                   "task,lane,priority,pin_cost,pin_pts,bar_cost,gen_hex,M_proof\n"
                   "task001,A,1,100,19.0,95,arc00001,1\n"
                   "task002,B,2,200,18.0,180,arc00002,2\n")
        self.write("results/DIFFICULTY.csv",
                   "task,lane,score,tier,ev,fam\n"
                   "task001,A,3,EASY,0.5,CROP\n"
                   "task002,B,9,HARD,0.2,REBUILD\n")
        self.write("results/LEDGER.csv",
                   "task,cost,pts,delta,lane,sha8,notes\n"
                   "task001,70,20.7,1.7,A,12345678,clean\n")
        self.write("results/REPRICE_AUDIT.csv",
                   "task,file,kind,true_cost,true_pts,pin_pts,true_delta,ledger_delta,verdict\n"
                   "task002,task002.onnx,canonical,240,17.7,18.0,-0.3,1.2,PHANTOM\n")
        self.write("results/task001/ATTEMPTS.jsonl",
                   '{"ts":"20260710T100000","model":"codex","att":1,"outcome":"DONE","delta":9}\n'
                   '{"ts":"20260710T120000","model":"codex","att":2,"outcome":"DONE","delta":1.7}\n')
        self.write("results/STOP_WAVE", "")
        self.monitor = webui.Monitor(root)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, relative, text):
        path = os.path.join(self.tmp.name, relative)
        with open(path, "w") as f:
            f.write(text)

    def test_snapshot_uses_clean_ledger_and_current_attempts(self):
        state = self.monitor.snapshot(force=True)
        self.assertEqual(state["status"], "paused")
        self.assertEqual(state["summary"]["banked_delta"], 1.7)
        task1, task2 = state["tasks"]
        self.assertEqual(task1["delta"]["source"], "banked")
        self.assertEqual(task1["clean_attempts"], 1)
        self.assertEqual(task1["historical_attempts"], 2)
        self.assertEqual(task2["delta"]["verdict"], "PHANTOM")
        self.assertAlmostEqual(task2["delta"]["delta"], -0.3)

    def test_task_detail_excludes_pre_fix_attempt(self):
        detail = self.monitor.task_detail("1")
        self.assertEqual(len(detail["attempts"]), 1)
        self.assertEqual(detail["attempts"][0]["att"], 2)
        self.assertEqual(detail["historical_attempts"], 2)


class TelemetryWriterTests(unittest.TestCase):
    def test_atomic_snapshot_reports_exact_model_task_ownership(self):
        with tempfile.TemporaryDirectory() as root:
            results = os.path.join(root, "results")
            os.makedirs(results)
            ledger = os.path.join(results, "LEDGER.csv")
            with open(ledger, "w") as f:
                f.write("task,cost,pts,delta,lane,sha8,notes\n")
            log = open(os.path.join(results, "worker.log"), "w")
            old = {name: getattr(orchestrate, name) for name in (
                "ROOT", "RESULTS", "LEDGER", "ORCH_NAME", "PM_JOBS", "PS_JOBS",
                "DECIDE_JOBS", "DECIDE_WAIT", "MODEL_BLOCKED_UNTIL")}
            try:
                orchestrate.ROOT = root
                orchestrate.RESULTS = results
                orchestrate.LEDGER = ledger
                orchestrate.ORCH_NAME = "test"
                orchestrate.PM_JOBS = []
                orchestrate.PS_JOBS = []
                orchestrate.DECIDE_JOBS = []
                orchestrate.DECIDE_WAIT = {}
                orchestrate.MODEL_BLOCKED_UNTIL = {}
                args = types.SimpleNamespace(workers=3, lanes="A,B", finite=False,
                                             improve_below=.25, no_watchdog=False)
                proc = types.SimpleNamespace(pid=4242)
                worker = {"proc": proc, "row": {"task": "task007", "lane": "B", "priority": "2"},
                          "model": "codex", "model_id": "gpt-5.6-sol", "effort": "high",
                          "att": 3, "log": log, "t0": orchestrate.time.time() - 12,
                          "timeout": 9000}
                orchestrate.write_run_state(args, [], [worker], set(), {}, {"n": 0})
                path = os.path.join(results, "RUN_STATE.test.json")
                with open(path) as f:
                    state = json.load(f)
                self.assertEqual(state["workers"][0]["task"], "task007")
                self.assertEqual(state["workers"][0]["model_id"], "gpt-5.6-sol")
                self.assertEqual(state["workers"][0]["benchmark_score"], 56)
                self.assertIn("required_capability", state["workers"][0])
                self.assertEqual(state["counts"]["workers"], 1)
                self.assertEqual(state["config"]["workers"], 3)
                self.assertEqual(state["allocation"]["worker_shares"], {"codex": 1.0})
                self.assertEqual(state["allocation"]["worker_active"]["codex"], 1)
                self.assertEqual(state["allocation"]["decider_shares"], {"codex": 1.0})
            finally:
                log.close()
                for name, value in old.items():
                    setattr(orchestrate, name, value)


if __name__ == "__main__":
    unittest.main()
