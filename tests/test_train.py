#!/usr/bin/env python3
"""Tests for the train skill scripts."""

import subprocess
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "train" / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = str(FIXTURES / "results.tsv")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SKILL_DIR / script), *args],
        capture_output=True,
        text=True,
    )


class TestAnalyzeResults(unittest.TestCase):
    def test_basic(self):
        r = run("analyze_results.py", RESULTS)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Experiment Analysis", r.stdout)
        self.assertIn("Total experiments", r.stdout)

    def test_shows_top5(self):
        r = run("analyze_results.py", RESULTS)
        self.assertIn("Top 5", r.stdout)

    def test_missing_file(self):
        r = run("analyze_results.py", "ghost.tsv")
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
