#!/usr/bin/env python3
"""Tests for the drift-detect skill scripts."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "drift-detect" / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"
TRAIN = str(FIXTURES / "train.csv")
PROD = str(FIXTURES / "production.csv")
SALES = str(FIXTURES / "sales.csv")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SKILL_DIR / script), *args],
        capture_output=True,
        text=True,
    )


class TestDetectDrift(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_basic(self):
        output = os.path.join(self.tmpdir, "drift.html")
        r = run("detect_drift.py", SALES, PROD, "--output", output)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Summary", r.stdout)
        self.assertTrue(os.path.exists(output))

    def test_html_report_created(self):
        output = os.path.join(self.tmpdir, "report.html")
        r = run("detect_drift.py", SALES, SALES, "--output", output)
        self.assertEqual(r.returncode, 0)
        with open(output) as f:
            content = f.read()
        self.assertIn("Drift Detection Report", content)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
