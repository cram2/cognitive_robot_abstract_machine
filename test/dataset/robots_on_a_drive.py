"""
Robots made of a single body whose mobile base is that body, for tests that move a robot
by its mobile base.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import Self

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import MobileBase, TGenericDrive
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% a mobile base that is the root of its robot


@dataclass(eq=False)
class RootBodyBase(MobileBase[TGenericDrive], ABC):
    """
    A mobile base that is the root body of its robot, facing along its x-axis.
    """

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    def setup_hardware_interfaces(self) -> None:
        return None

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(root=robot_root)


@dataclass(eq=False)
class RootBodyBaseOnADifferentialDrive(RootBodyBase[DifferentialDrive]):
    """
    A mobile base that is the root body of its robot, driven by a differential drive.
    """


@dataclass(eq=False)
class RootBodyBaseOnAnOmniDrive(RootBodyBase[OmniDrive]):
    """
    A mobile base that is the root body of its robot, driven by an omnidirectional
    drive.
    """


# %% robots whose root body is their mobile base


@dataclass(eq=False)
class RobotOnADifferentialDrive(
    MinimalRobot, HasMobileBase[RootBodyBaseOnADifferentialDrive]
):
    """
    A minimal robot whose root body is its mobile base, on a differential drive.
    """


@dataclass(eq=False)
class RobotOnAnOmniDrive(MinimalRobot, HasMobileBase[RootBodyBaseOnAnOmniDrive]):
    """
    A minimal robot whose root body is its mobile base, on an omnidirectional drive.
    """
