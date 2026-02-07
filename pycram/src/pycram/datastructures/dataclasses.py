from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import (
    List,
    Optional,
    Any,
    TYPE_CHECKING, Dict, Union,
)

from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)
from .enums import ApproachDirection, VerticalAlignment, Grasp
from .pose import PoseStamped

if TYPE_CHECKING:
    from ..plan import Plan


@dataclass
class Context:
    """
    A dataclass for storing the context of a plan
    """

    world: World
    """
    The world in which the plan is executed
    """

    robot: AbstractRobot
    """
    The semantic robot annotation which should execute the plan
    """

    super_plan: Optional[Plan] = field(default=None)
    """
    The plan of which this plan/designator is a part of
    """

    ros_node: Optional[Any] = field(default=None)
    """
    A ROS node that should be used for communication in this plan
    """

    @classmethod
    def from_world(cls, world: World, super_plan: Optional[Plan] = None):
        """
        Create a context from a world by getting the first robot in the world. There is no super plan in this case.

        :param world: The world for which to create the context
        :param super_plan: An optional super plan
        :return: A context with the first robot in the world and no super plan
        """
        return cls(
            world=world,
            robot=world.get_semantic_annotations_by_type(AbstractRobot)[0],
            super_plan=super_plan,
        )

    @classmethod
    def from_plan(cls, plan: Plan):
        """
        Create a context from a plan by getting the context information from the plan and setting the super plan to
        the given plan.

        :param plan: Plan from which to create the context
        :return: A new context with the world and robot from the plan and the super plan set to the given plan
        """
        return cls(world=plan.world, robot=plan.robot, super_plan=plan)


@dataclass
class ExecutionData:
    """
    A dataclass for storing the information of an execution that is used for creating a robot description for that
    execution. An execution is a Robot with a virtual mobile base that can be used to move the robot in the environment.
    """

    execution_start_pose: PoseStamped
    """
    Start of the robot at the start of execution of an action designator
    """

    execution_start_world_state: np.ndarray
    """
    The world state at the start of execution of an action designator
    """

    execution_end_pose: Optional[PoseStamped] = None
    """
    The pose of the robot at the end of executing an action designator
    """

    execution_end_world_state: Optional[np.ndarray] = None
    """
    The world state at the end of executing an action designator
    """

    added_world_modifications: List[WorldModelModificationBlock] = field(
        default_factory=list
    )
    """
    A list of World modification blocks that were added during the execution of the action designator
    """

    manipulated_body_pose_start: Optional[PoseStamped] = None
    """
    Start pose of the manipulated Body if there was one
    """

    manipulated_body_pose_end: Optional[PoseStamped] = None
    """
    End pose of the manipulated Body if there was one
    """

    manipulated_body: Optional[Body] = None
    """
    Reference to the manipulated body 
    """

    manipulated_body_name: Optional[str] = None
    """
    Name of the manipulated body
    """



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
