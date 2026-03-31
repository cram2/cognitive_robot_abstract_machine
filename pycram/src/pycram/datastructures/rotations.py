from __future__ import annotations

import math
from typing import Dict, Optional, Union, List

from pycram.datastructures.enums import Grasp, ApproachDirection, VerticalAlignment


class Rotations(Dict[Optional[Union[Grasp, bool]], List[float]]):
    """
    A dictionary that defines standard quaternions for different grasps and orientations. This is mainly used
    to automatically calculate all grasp descriptions of a robot gripper for the robot description.

    SIDE_ROTATIONS: The quaternions for the different approach directions (front, back, left, right)
    VERTICAL_ROTATIONS: The quaternions for the different vertical alignments, in case the object requires for
    example a top grasp
    HORIZONTAL_ROTATIONS: The quaternions for the different horizontal alignments, in case the gripper needs to roll
    90°
    """

    SIDE_ROTATIONS = {
        ApproachDirection.FRONT: [0, 0, 0, 1],
        ApproachDirection.BACK: [0, 0, 1, 0],
        ApproachDirection.LEFT: [0, 0, -math.sqrt(2) / 2, math.sqrt(2) / 2],
        ApproachDirection.RIGHT: [0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2],
    }

    VERTICAL_ROTATIONS = {
        VerticalAlignment.NoAlignment: [0, 0, 0, 1],
        VerticalAlignment.TOP: [0, math.sqrt(2) / 2, 0, math.sqrt(2) / 2],
        VerticalAlignment.BOTTOM: [0, -math.sqrt(2) / 2, 0, math.sqrt(2) / 2],
    }

    HORIZONTAL_ROTATIONS = {
        False: [0, 0, 0, 1],
        True: [math.sqrt(2) / 2, 0, 0, math.sqrt(2) / 2],
    }
