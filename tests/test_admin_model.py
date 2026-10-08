import sys
import unittest
from unittest import mock
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import admin_model


class AdminModelTests(unittest.TestCase):
    def test_command_is_read_only_ephemeral_and_uses_stdin(self):
        cmd = admin_model.command("gpt-5.3-codex-spark", "low")
        self.assertIn("read-only", cmd)
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--ignore-rules", cmd)
        self.assertEqual("-", cmd[-1])
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)

    def test_infrastructure_failure_falls_back_to_gpt55(self):
        failed = subprocess.CompletedProcess([], 1, "", "temporary outage")
        passed = subprocess.CompletedProcess([], 0, "OK\n", "")
        with mock.patch.dict(admin_model.os.environ,
                             {"ADMIN_MODEL": "gpt-5.3-codex-spark",
                              "ADMIN_FALLBACK_MODEL": "gpt-5.5"}), \
                mock.patch.object(admin_model.subprocess, "run",
                                  side_effect=[failed, passed]) as run:
            rc, body, model = admin_model.run("prompt", 10, "low")
        self.assertEqual((0, "OK", "gpt-5.5"), (rc, body, model))
        self.assertEqual(2, run.call_count)

    def test_primary_is_spark_read_only(self):
        passed = subprocess.CompletedProcess([], 0, "ADMIN_OK\n", "")
        with mock.patch.dict(admin_model.os.environ,
                             {"ADMIN_SPARK_MODEL": "gpt-5.3-codex-spark",
                              "ADMIN_FALLBACK_MODEL": "gpt-5.5"}), \
                mock.patch.object(admin_model.subprocess, "run",
                                  side_effect=[passed]) as run:
            rc, body, model = admin_model.run("prompt", 10, "low")
        self.assertEqual((0, "ADMIN_OK", "gpt-5.3-codex-spark"), (rc, body, model))
        spark_cmd = run.call_args_list[0].args[0]
        self.assertIn("read-only", spark_cmd)
        self.assertIn("--ephemeral", spark_cmd)
        self.assertEqual(1, run.call_count)


if __name__ == "__main__":
    unittest.main()
