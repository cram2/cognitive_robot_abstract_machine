from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)


@dataclass
class ExecutionData:
    """
    All kinds of data that is relevant to the execution of an action designator.
    """

    execution_start_pose: Pose
    """
    Start of the robot at the start of execution of an action designator
    """

    execution_start_world_state: np.ndarray
    """
    The world state at the start of execution of an action designator.
    """

    execution_end_pose: Pose | None = None
    """
    The pose of the robot at the end of executing an action designator.
    """

    execution_end_world_state: np.ndarray | None = None
    """
    The world state at the end of executing an action designator.
    """

    added_world_modifications: list[WorldModelModificationBlock] = field(
        default_factory=list
    )
    """
    A list of World modification blocks that were added during the execution of the
    action designator.
    """
