import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("dgl")

from examples.gcn_demo import run_demo


def test_real_gcn_forward_and_training_step():
    result = run_demo()
    assert result["logits_shape"] == [4, 1]
    assert 0 <= result["final_loss"] < result["initial_loss"]


@pytest.mark.parametrize("name", ["model", "util", "dglutil", "inference"])
def test_library_imports_from_repository_root(name):
    assert importlib.import_module("GraphLib." + name)


def test_legacy_script_imports_still_work():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", "import model, util, dglutil, inference"],
                            cwd=root / "GraphLib", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
