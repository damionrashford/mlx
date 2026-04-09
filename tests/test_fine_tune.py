#!/usr/bin/env python3
"""Tests for the fine-tune skill scripts."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "fine-tune" / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"
JSONL = str(FIXTURES / "sample.jsonl")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SKILL_DIR / script), *args],
        capture_output=True,
        text=True,
    )


class TestPrepareDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_alpaca_format(self):
        output = os.path.join(self.tmpdir, "alpaca.jsonl")
        r = run(
            "prepare_dataset.py", JSONL,
            "--format", "alpaca",
            "--output", output,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(output))
        with open(output) as f:
            rows = [json.loads(line) for line in f]
        self.assertGreater(len(rows), 0)
        self.assertIn("instruction", rows[0])
        self.assertIn("output", rows[0])

    def test_sharegpt_format(self):
        output = os.path.join(self.tmpdir, "sharegpt.jsonl")
        r = run(
            "prepare_dataset.py", JSONL,
            "--format", "sharegpt",
            "--human-col", "instruction",
            "--assistant-col", "output",
            "--output", output,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(output))

    def test_dedup(self):
        output = os.path.join(self.tmpdir, "dedup.jsonl")
        r = run(
            "prepare_dataset.py", JSONL,
            "--format", "alpaca",
            "--output", output,
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("Deduplicated", r.stdout)

    def test_invalid_format(self):
        r = run(
            "prepare_dataset.py", JSONL,
            "--format", "invalid",
        )
        self.assertNotEqual(r.returncode, 0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
