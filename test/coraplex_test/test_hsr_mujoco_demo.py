"""
Tests for the HSR carrying a milk carton from a counter to a table in MuJoCo, which is
held by friction alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk

from ..pytest_environment import runs_in_continuous_integration

pytestmark = [
    pytest.mark.skipif(
        not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
    ),
    pytest.mark.skipif(
        not HSRB.is_description_installed(),
        reason="The description of the HSR is not installed",
    ),
]

# %% the demo, loaded from its folder


@pytest.fixture(scope="module")
def hsr_demo() -> ModuleType:
    """
    The HSR demo's ``demo`` module, its folder put on the path for the modules it imports.
    """
    folder = (
        Path(__file__).parents[2] / "coraplex" / "demos" / "coraplex_hsr_mujoco_demo"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(folder))
        specification = importlib.util.spec_from_file_location(
            "hsr_mujoco_demo", folder / "demo.py"
        )
        module = importlib.util.module_from_spec(specification)
        patch.setitem(sys.modules, specification.name, module)
        specification.loader.exec_module(module)
    return module


# %% carrying the milk


def test_the_milk_ends_where_it_is_set_down_on_the_table(hsr_demo):
    demonstration = hsr_demo.MilkTransportDemonstration(headless=True)

    world = demonstration.run()

    [milk] = world.get_semantic_annotations_by_type(Milk)
    set_down_at = demonstration._milk_pose_on(
        world.get_body_by_name(hsr_demo.SceneBody.TABLE), milk
    )
    position = world.compute_forward_kinematics_np(world.root, milk.root)[:3, 3]
    assert list(position) == pytest.approx(
        [float(set_down_at.x), float(set_down_at.y), float(set_down_at.z)], abs=0.02
    )
