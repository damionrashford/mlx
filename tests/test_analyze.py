#!/usr/bin/env python3
"""Tests for the analyze skill scripts."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "analyze" / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"
SALES = str(FIXTURES / "sales.csv")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", str(SKILL_DIR / script), *args],
        capture_output=True,
        text=True,
    )


class TestDescriptiveStats(unittest.TestCase):
    def test_basic(self):
        r = run("descriptive_stats.py", SALES)
        self.assertEqual(r.returncode, 0)
        self.assertIn("Descriptive Statistics", r.stdout)
        self.assertIn("Shape", r.stdout)

    def test_grouped(self):
        r = run("descriptive_stats.py", SALES, "--group", "segment", "--value", "revenue")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Enterprise", r.stdout)
        self.assertIn("SMB", r.stdout)

    def test_missing_file(self):
        r = run("descriptive_stats.py", "nonexistent.csv")
        self.assertNotEqual(r.returncode, 0)


class TestHypothesisTest(unittest.TestCase):
    def test_two_groups(self):
        r = run(
            "hypothesis_test.py", SALES,
            "--col", "revenue",
            "--group", "segment",
            "--a", "Enterprise",
            "--b", "SMB",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("Hypothesis Test", r.stdout)
        self.assertIn("p-value", r.stdout)
        self.assertIn("Effect size", r.stdout)

    def test_empty_group(self):
        r = run(
            "hypothesis_test.py", SALES,
            "--col", "revenue",
            "--group", "segment",
            "--a", "NonExistent",
            "--b", "SMB",
        )
        self.assertNotEqual(r.returncode, 0)


class TestABTest(unittest.TestCase):
    def test_conversion(self):
        r = run(
            "ab_test.py", SALES,
            "--col", "converted",
            "--group", "variant",
            "--control", "A",
            "--treatment", "B",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("A/B Test", r.stdout)

    def test_continuous(self):
        r = run(
            "ab_test.py", SALES,
            "--col", "revenue",
            "--group", "variant",
            "--control", "A",
            "--treatment", "B",
            "--metric", "continuous",
        )
        self.assertEqual(r.returncode, 0)


class TestCohortAnalysis(unittest.TestCase):
    def test_basic(self):
        r = run(
            "cohort_analysis.py", SALES,
            "--user", "customer_id",
            "--date", "date",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("Cohort", r.stdout)


class TestRFMSegmentation(unittest.TestCase):
    def test_basic(self):
        r = run(
            "rfm_segmentation.py", SALES,
            "--customer", "customer_id",
            "--date", "date",
            "--value", "revenue",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("RFM", r.stdout)


class TestTrendAnalysis(unittest.TestCase):
    def test_basic(self):
        r = run(
            "trend_analysis.py", SALES,
            "--date", "date",
            "--value", "revenue",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("Trend", r.stdout)


class TestValidate(unittest.TestCase):
    def test_basic(self):
        r = run("validate.py", SALES)
        self.assertEqual(r.returncode, 0)
        self.assertIn("Validation Report", r.stdout)

    def test_join_check(self):
        r = run(
            "validate.py", SALES,
            "--join-check", SALES,
            "--join-key", "customer_id",
        )
        self.assertEqual(r.returncode, 0)


class TestChartTemplates(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_bar_chart(self):
        output = os.path.join(self.tmpdir, "bar.png")
        r = run(
            "chart_templates.py", SALES,
            "--type", "bar",
            "--x", "segment",
            "--y", "revenue",
            "-o", output,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(output))

    def test_heatmap(self):
        output = os.path.join(self.tmpdir, "heatmap.png")
        r = run(
            "chart_templates.py", SALES,
            "--type", "heatmap",
            "-o", output,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(output))

    def test_histogram(self):
        output = os.path.join(self.tmpdir, "hist.png")
        r = run(
            "chart_templates.py", SALES,
            "--type", "hist",
            "--x", "revenue",
            "-o", output,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(output))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
