#!/usr/bin/env python3
"""Tests for the ml-docs skill scripts."""

import json
import subprocess
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "ml-docs" / "scripts"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SKILL_DIR / "process.py"), *args],
        capture_output=True,
        text=True,
    )


class TestMLDocsProcess(unittest.TestCase):
    def test_list(self):
        r = run("list")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertIn("libraries", result)
        self.assertGreaterEqual(result["count"], 19)

    def test_resolve_pandas(self):
        r = run("resolve", "--library", "pandas", "--query", "groupby")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertIn("fetch_in_order", result)
        self.assertGreater(len(result["fetch_in_order"]), 0)

    def test_resolve_alias(self):
        r = run("resolve", "--library", "torch", "--query", "autograd")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertEqual(result["library"], "PyTorch")

    def test_resolve_unknown(self):
        r = run("resolve", "--library", "unknownlib", "--query", "test")
        self.assertNotEqual(r.returncode, 0)
        err = json.loads(r.stderr)
        self.assertIn("error", err)

    def test_search_url(self):
        r = run("search-url", "--library", "sklearn", "--query", "cross_val_score")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertIn("search_url", result)
        self.assertIn("cross_val_score", result["search_url"])

    def test_api_reference_resolution(self):
        r = run("resolve", "--library", "pandas", "--query", "DataFrame.groupby")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        urls = [item["url"] for item in result["fetch_in_order"]]
        # Should include an API reference URL for dot-notation queries
        self.assertTrue(any("groupby" in url for url in urls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
