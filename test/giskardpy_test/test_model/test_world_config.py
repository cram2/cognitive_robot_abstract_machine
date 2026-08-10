from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from giskardpy.data_types.exceptions import WorldNotEmptyError
from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

_SIMPLE_ROBOT_URDF_PATH = (
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "urdf"
    / "simple_two_arm_robot.urdf"
)


@pytest.fixture
def simple_robot_urdf() -> str:
    """
    URDF of a small fixed-base robot whose root link is ``base_link``.
    """
    return _SIMPLE_ROBOT_URDF_PATH.read_text()


class TestWorldWithFixedRobot:
    def test_robot_root_becomes_world_root(self, simple_robot_urdf: str) -> None:
        """
        A rigidly mounted robot needs no map frame: its URDF root is the world root.
        """
        config = WorldWithFixedRobot(urdf=simple_robot_urdf)

        config.setup_world()

        assert config.world.root is config.robot_root
        assert config.world.root.name.name == "base_link"

    def test_setup_on_non_empty_world_raises(self, simple_robot_urdf: str) -> None:
        """
        Merging into a pre-populated world would demote the robot below the existing
        root.
        """
        config = WorldWithFixedRobot(urdf=simple_robot_urdf)
        with config.world.modify_world():
            config.world.add_body(Body(name=PrefixedName("pre_existing")))

        with pytest.raises(WorldNotEmptyError):
            config.setup_world()
