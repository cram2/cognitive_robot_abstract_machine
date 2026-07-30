import subprocess
import sys
from pathlib import Path

import pytest

DEMOS_ROOT = Path(__file__).resolve().parents[2] / "coraplex" / "demos"

WRAPPER_PATHS = [
    DEMOS_ROOT / "coraplex_bullet_world_demo" / "test_demo.py",
    DEMOS_ROOT / "coraplex_real_tracy" / "test_demo.py",
]


@pytest.mark.parametrize(
    "wrapper_path", WRAPPER_PATHS, ids=[p.parent.name for p in WRAPPER_PATHS]
)
def test_wrapper_reports_the_real_import_failure(tmp_path, wrapper_path):
    """
    A demo's ``test_demo.py`` wrapper must surface the real exception from a failing
    ``import demo`` (as a traceback on stderr) instead of silently exiting, so CI
    failures are diagnosable.
    """
    (tmp_path / "demo.py").write_text("raise RuntimeError('boom from demo fixture')\n")
    (tmp_path / "test_demo.py").write_text(wrapper_path.read_text())

    result = subprocess.run(
        [sys.executable, "test_demo.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "boom from demo fixture" in result.stderr
    assert "Traceback" in result.stderr
