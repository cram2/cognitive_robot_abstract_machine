from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum

# %% where a robot part is read from


@dataclass
class InputSource(ABC):
    """
    Where a robot part is read from, either the world it stands in or the robot itself.
    """


class RobotTopic(StrEnum):
    """
    Topics a robot publishes the state of its parts on, as spelled by every robot that
    follows the ROS conventions.

    ..note:: A robot that names one of them differently spells it in its own topic enum.
    """

    JOINT_STATES = "joint_states"


# %% joint positions


@dataclass
class JointPositionSource(InputSource, ABC):
    """
    Where the positions of a robot part's joints come from.
    """


@dataclass
class SimulatedJointPositionSource(JointPositionSource):
    """
    A source that leaves the joint positions the world already holds, as the simulator
    integrates them.
    """


# %% base pose


@dataclass
class BasePoseSource(InputSource, ABC):
    """
    Where the pose of a mobile base comes from.
    """


@dataclass
class SimulatedBasePoseSource(BasePoseSource):
    """
    A source that leaves the drive origin the world already holds, as the simulator
    integrates it.
    """
