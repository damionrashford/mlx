#!/usr/bin/env python3
"""Tests for the autoexperiment skill scripts."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "autoexperiment" / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"


def run(script: str, *args: str, env: dict = None) -> subprocess.CompletedProcess:
    import os as _os
    e = {**_os.environ, **(env or {})}
    return subprocess.run(
        ["uv", "run", str(SKILL_DIR / script), *args],
        capture_output=True,
        text=True,
        env=e,
    )


class TestTimeBudgetTrain(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_runs_and_records(self):
        results_tsv = os.path.join(self.tmpdir, "results.tsv")
        r = run(
            "time_budget_train.py",
            "--budget", "5",
            "--exp-id", "test_exp",
            env={"RESULTS_TSV": results_tsv},
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Training complete", r.stdout)
        self.assertIn("Val loss", r.stdout)
        self.assertTrue(os.path.exists(results_tsv))

    def test_records_to_tsv(self):
        import csv
        results_tsv = os.path.join(self.tmpdir, "results2.tsv")
        r = run(
            "time_budget_train.py",
            "--budget", "3",
            "--exp-id", "exp_test_01",
            env={"RESULTS_TSV": results_tsv},
        )
        self.assertEqual(r.returncode, 0)
        with open(results_tsv) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "exp_test_01")
        self.assertIn(rows[0]["status"], ("KEEP", "DISCARD", "CRASH"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
