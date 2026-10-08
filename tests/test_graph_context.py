import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

import graph_context


class ProvenGraphContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        self.pack = self.root / "pack"
        (self.corpus / "task007").mkdir(parents=True)
        self.pack.mkdir()
        self.manifest = self.corpus / "manifest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, data, points, source, key=None):
        full = hashlib.sha256(data).hexdigest()
        key = key or full[:8]
        (self.corpus / "task007" / f"{key}.onnx").write_bytes(data)
        manifest = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        manifest[key] = {"task": 7, "pts": points, "source": source}
        self.manifest.write_text(json.dumps(manifest))
        return key, full

    def test_stages_highest_paying_exact_proof_read_only(self):
        self.add(b"older-paying-graph", 18.1, "probe#10 gap+0.2")
        best_key, best_sha = self.add(b"best-paying-graph", 18.7, "probe#11 gap+0.6")
        self.add(b"corrupt-high-claim", 99.0, "probe#12 gap+80", key="deadbeef")

        text = graph_context.stage_kaggle_reference(
            7, str(self.pack), str(self.manifest), str(self.corpus))
        staged = self.pack / "refs" / graph_context.REFERENCE_NAME
        self.assertEqual(b"best-paying-graph", staged.read_bytes())
        self.assertEqual(0, staged.stat().st_mode & stat.S_IWUSR)
        self.assertIn(best_sha, text)
        self.assertIn(best_key, text)
        self.assertIn("18.7000", text)
        self.assertIn("probe#11", text)
        self.assertIn("architecture is **not** assumed optimal", text)
        self.assertEqual(text, (self.pack / graph_context.CONTEXT_NAME).read_text())

    def test_revoked_proof_is_removed_from_active_reference_but_archived(self):
        self.add(b"once-proven", 18.2, "probe#20 gap+0.3")
        graph_context.stage_kaggle_reference(
            7, str(self.pack), str(self.manifest), str(self.corpus))
        self.manifest.write_text("{}")

        text = graph_context.stage_kaggle_reference(
            7, str(self.pack), str(self.manifest), str(self.corpus))
        self.assertIn("NO EXACT PER-MEMBER", text)
        self.assertFalse((self.pack / "refs" / graph_context.REFERENCE_NAME).exists())
        archived = list((self.pack / "refs" / "quarantine_unproven").iterdir())
        self.assertEqual(1, len(archived))
        self.assertEqual(b"once-proven", archived[0].read_bytes())


if __name__ == "__main__":
    unittest.main()
