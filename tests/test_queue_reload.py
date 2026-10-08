import csv
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import orchestrate


COLS = ["task", "lane", "priority", "pin_cost", "pin_pts", "bar_cost", "gen_hex", "M_proof"]


def _write_queue(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def _row(task, lane, priority):
    return {"task": f"task{task}", "lane": lane, "priority": str(priority),
            "pin_cost": "1000", "pin_pts": "18.0", "bar_cost": "861",
            "gen_hex": "deadbeef", "M_proof": "30"}


class QueueReloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.queue_file = Path(self.tmp.name) / "QUEUE.csv"
        self._orig_queue = orchestrate.QUEUE
        orchestrate.QUEUE = str(self.queue_file)
        orchestrate._QUEUE_STATE["mtime"] = None

    def tearDown(self):
        orchestrate.QUEUE = self._orig_queue
        orchestrate._QUEUE_STATE["mtime"] = None
        self.tmp.cleanup()

    def test_reprioritize_reorders_pending_live(self):
        _write_queue(self.queue_file, [_row("001", "B", 1), _row("002", "B", 2),
                                       _row("003", "B", 3)])
        queue = orchestrate.read_queue(None, None)
        self.assertEqual(["task001", "task002", "task003"], [r["task"] for r in queue])

        # first call just seeds the mtime (no reload)
        self.assertFalse(orchestrate.maybe_reload_queue(queue, None, None))

        # a repin rewrites QUEUE.csv with a new priority order + fresh bars
        time.sleep(0.01)
        _write_queue(self.queue_file, [
            {**_row("001", "B", 3), "bar_cost": "500"},
            _row("002", "B", 1), _row("003", "B", 2)])
        os.utime(self.queue_file, (time.time() + 1, time.time() + 1))

        self.assertTrue(orchestrate.maybe_reload_queue(queue, None, None))
        # live re-sort by the new priority
        self.assertEqual(["task002", "task003", "task001"], [r["task"] for r in queue])
        # fresh bars picked up too
        self.assertEqual("500", next(r for r in queue if r["task"] == "task001")["bar_cost"])
        # no task dropped or duplicated
        self.assertEqual({"task001", "task002", "task003"}, {r["task"] for r in queue})

    def test_only_pending_are_kept_in_scope(self):
        # simulate that task001 was already spawned (popped from queue); it must not
        # reappear on a reprioritize
        _write_queue(self.queue_file, [_row("001", "B", 1), _row("002", "B", 2),
                                       _row("003", "B", 3)])
        full = orchestrate.read_queue(None, None)
        pending = [r for r in full if r["task"] != "task001"]  # task001 spawned
        orchestrate.maybe_reload_queue(pending, None, None)  # seed mtime

        time.sleep(0.01)
        _write_queue(self.queue_file, [_row("001", "B", 1), _row("003", "B", 2),
                                       _row("002", "B", 3)])
        os.utime(self.queue_file, (time.time() + 1, time.time() + 1))
        orchestrate.maybe_reload_queue(pending, None, None)
        # task001 stays out (it was already spawned); the rest re-sort
        self.assertEqual({"task002", "task003"}, {r["task"] for r in pending})
        self.assertEqual(["task003", "task002"], [r["task"] for r in pending])

    def test_no_change_is_a_noop(self):
        _write_queue(self.queue_file, [_row("001", "B", 1), _row("002", "B", 2)])
        queue = orchestrate.read_queue(None, None)
        orchestrate.maybe_reload_queue(queue, None, None)  # seed
        self.assertFalse(orchestrate.maybe_reload_queue(queue, None, None))
        self.assertEqual(["task001", "task002"], [r["task"] for r in queue])


if __name__ == "__main__":
    unittest.main()
