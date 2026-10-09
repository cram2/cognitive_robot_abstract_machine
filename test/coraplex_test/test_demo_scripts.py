"""
Coverage for the demo scripts CI runs as regression tests.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from ament_index_python.packages import get_packages_with_prefixes

from coraplex.demonstrations import RobotDemonstrationRosSession
from semantic_digital_twin.robots.garmi import Garmi

DEMOS_ROOT = Path(__file__).resolve().parents[2] / "coraplex" / "demos"

WRAPPER_PATHS = [
    DEMOS_ROOT / "coraplex_bullet_world_demo" / "test_demo.py",
    DEMOS_ROOT / "coraplex_real_tracy" / "test_demo.py",
    DEMOS_ROOT / "coraplex_unitree_g1_warehouse_demo" / "test_demo.py",
]

GARMI_DEMO = "coraplex_garmi_demo"
"""
Directory of the demo whose draws have to repeat.
"""

GARMI_DEMO_RESOURCE_PACKAGES = ["iai_garmi_apartment", "garmi_description"]
"""
Workspace packages the garmi demo reads its scene and its robot description from.
"""


def load_demo_module(demo_directory: str) -> ModuleType:
    """
    Import the ``demo.py`` of the given demo, which is a script rather than a package.

    :param demo_directory: Name of the demo's directory under ``coraplex/demos``.
    """
    specification = importlib.util.spec_from_file_location(
        demo_directory, DEMOS_ROOT / demo_directory / "demo.py"
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[demo_directory] = module
    specification.loader.exec_module(module)
    return module


# %% wrapper failure reporting


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


BULLET_WORLD_WRAPPER = DEMOS_ROOT / "coraplex_bullet_world_demo" / "test_demo.py"
"""
The wrapper of the one demo that is imported first and run through its ``main``.
"""


@pytest.mark.parametrize(
    "main_body, exit_code",
    [("raise RuntimeError('boom from main')", 1), ("pass", 0)],
    ids=["main-fails", "main-succeeds"],
)
def test_wrapper_reports_whether_the_demos_main_succeeded(
    tmp_path, main_body, exit_code
):
    """
    A demo that runs from ``main`` rather than on import fails inside ``main``, and the
    wrapper has to surface that failure as it does a failing import.
    """
    (tmp_path / "demo.py").write_text(f"def main():\n    {main_body}\n")
    (tmp_path / "test_demo.py").write_text(BULLET_WORLD_WRAPPER.read_text())

    result = subprocess.run(
        [sys.executable, "test_demo.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == exit_code
    assert ("boom from main" in result.stderr) is (exit_code == 1)


# %% reproducible draws


def test_garmi_demonstration_pins_the_draws_its_locations_make(rclpy_node):
    """
    The garmi demonstration is kept as a regression test, so the poses its locations
    draw must be the ones the run before drew -- a plan that reached the drawer once
    reaches it again instead of standing somewhere new each run.
    """
    installed_packages = get_packages_with_prefixes()
    missing_packages = [
        package
        for package in GARMI_DEMO_RESOURCE_PACKAGES
        if package not in installed_packages
    ]
    if missing_packages:
        pytest.skip(
            f"GARMI demo resources not installed: {', '.join(missing_packages)}"
        )

    demo_module = load_demo_module(GARMI_DEMO)
    demonstration = demo_module.GarmiApartmentDemonstration(used_robot=Garmi)
    demonstration.ros_session = RobotDemonstrationRosSession.start(
        demonstration.ros_node_name
    )

    context = demonstration.build_context(demonstration.build_simulated_world())
    demonstration.ros_session.stop()

    assert context.sampling_seed == demo_module.SamplingSeed.REPEATABLE
