"""
Smoke tests running each underspecified tool-action demo end to end on the simulated
robot.

Every demo initializes ROS and builds its own world, so each one runs in its own
process. Each run is a full simulation taking minutes, hence the generous timeout.
"""

import subprocess
import sys

import pytest

from experiments.tool_based_actions.underspecified_demo import (
    demo_cutting,
    demo_mixing,
    demo_pouring,
    demo_wiping,
)

DEMO_MODULES = (demo_cutting, demo_mixing, demo_pouring, demo_wiping)
"""
The demo modules under test, each runnable as ``python -m <module>``.
"""

DEMO_TIMEOUT = 1200.0
"""
Wall-clock limit in seconds for one demo run.
"""

@pytest.mark.parametrize("demo_module", DEMO_MODULES, ids=lambda module: module.__name__)
def test_underspecified_demo_runs_to_completion(demo_module) -> None:
    """
    Run one underspecified demo in an isolated process and expect a clean exit.
    """
    completed_process = subprocess.run(
        [sys.executable, "-m", demo_module.__name__], timeout=DEMO_TIMEOUT
    )
    assert completed_process.returncode == 0
