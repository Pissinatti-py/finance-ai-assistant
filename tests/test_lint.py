"""Lint enforced as a test, so a dirty tree fails the suite without needing CI.

Skipped when ruff isn't installed (e.g. a runtime-only environment).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")


@pytest.mark.parametrize("cmd", [["ruff", "check", "."], ["ruff", "format", "--check", "."]])
def test_ruff_clean(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}"
