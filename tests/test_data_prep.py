#!/usr/bin/env python3
"""Tests for the data-prep skill scripts."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "data-prep" / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"
TRAIN = str(FIXTURES / "train.csv")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", str(SKILL_DIR / script), *args],
        capture_output=True,
        text=True,
    )


class TestEDA(unittest.TestCase):
    def test_basic(self):
        r = run("eda.py", TRAIN)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("EDA Report", r.stdout)
        self.assertIn("Shape", r.stdout)

    def test_with_target(self):
        r = run("eda.py", TRAIN, "--target", "target")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Task:", r.stdout)

    def test_missing_file(self):
        r = run("eda.py", "ghost.csv")
        self.assertNotEqual(r.returncode, 0)


class TestClean(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_basic(self):
        output = os.path.join(self.tmpdir, "clean.csv")
        r = run("clean.py", TRAIN, "-o", output)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(output))

    def test_check_only(self):
        r = run("clean.py", TRAIN, "--check-only")
        self.assertEqual(r.returncode, 0, r.stderr)
        # --check-only outputs JSON with a "passed" field
        import json
        result = json.loads(r.stdout)
        self.assertIn("passed", result)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestEngineerFeatures(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_basic(self):
        output = os.path.join(self.tmpdir, "features.csv")
        r = run("engineer_features.py", TRAIN, "-o", output)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(output))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
