#!/usr/bin/env python3
"""Tests for the notebook skill scripts."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "skills" / "notebook" / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SKILL_DIR / script), *args],
        capture_output=True,
        text=True,
    )


def make_notebook(tmpdir: str, name: str = "test.ipynb") -> str:
    """Create a minimal valid notebook for testing."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Analysis\n", "This notebook explores the data."],
                "id": "md1",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["import pandas as pd\nimport numpy as np\n"],
                "outputs": [],
                "execution_count": None,
                "id": "code1",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["df = pd.DataFrame({'a': [1, 2, 3]})\ndf.head()"],
                "outputs": [],
                "execution_count": None,
                "id": "code2",
            },
        ],
    }
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        json.dump(nb, f)
    return path


class TestAssessNotebook(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.nb_path = make_notebook(self.tmpdir)

    def test_basic(self):
        r = run("assess.py", self.nb_path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Notebook Assessment", r.stdout)
        self.assertIn("Score", r.stdout)

    def test_json_output(self):
        r = run("assess.py", self.nb_path, "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertIn("score", result)
        self.assertIn("issues", result)
        self.assertIn("code_cells", result)

    def test_missing_file(self):
        r = run("assess.py", "ghost.ipynb")
        self.assertNotEqual(r.returncode, 0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
