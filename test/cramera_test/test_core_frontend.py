"""
The shipped core viewer offers inspection and playback without robot editing.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% browser package boundary
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_core_viewer_scope() -> None:
    """
    Execute the browser package's passive-viewer contract through Node.
    """
    source = Path(__file__).parent / "js" / "test_core_viewer.js"
    result = subprocess.run(
        ["node", "--test", str(source)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_all_retained_browser_units() -> None:
    """
    Every retained browser regression participates in the pytest suite.
    """
    sources = sorted((Path(__file__).parent / "js").glob("test_*.js"))
    result = subprocess.run(
        ["node", "--experimental-test-coverage", "--test", *map(str, sources)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = result.stdout.partition("# start of coverage report")[2]
    if report:
        print(report)
