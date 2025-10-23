from __future__ import division

from typing import Optional

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


@validated_dataclass
class Open(Goal):
    """
    Open a container in an environment.
    Only works with the environment was added as urdf.
    Assumes that a handle has already been grasped.
    Can only handle containers with 1 dof, e.g. drawers or doors.
    """

    tip_link: Body
    """end effector that is grasping the handle"""

    environment_link: Body
    """name of the handle that was grasped"""

    goal_joint_state: Optional[float] = None
    """goal state for the container. default is maximum joint state."""

    weight: float = WEIGHT_ABOVE_CA

    def __post_init__(self):
        self.connection = self.environment_link.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

        max_position = self.connection.dof.upper_limits.position
        if self.goal_joint_state is None:
            self.goal_joint_state = max_position
        else:
            self.goal_joint_state = min(max_position, self.goal_joint_state)

        goal_state = {self.connection: self.goal_joint_state}
        hinge_goal = JointPositionList(
            goal_state=goal_state, name=f"{self.name}/hinge", weight=self.weight
        )
        self.add_task(hinge_goal)

        handle_pose = cas.TransformationMatrix(
            reference_frame=self.tip_link, child_frame=self.tip_link
        )

        hold_handle = CartesianPose(
            root_link=self.environment_link,
            tip_link=self.tip_link,
            name=f"{self.name}/hold handle",
            goal_pose=handle_pose,
            weight=self.weight,
        )
        self.add_task(hold_handle)
        self.observation_expression = cas.logic_and(
            hinge_goal.observation_expression, hold_handle.observation_expression
        )


@validated_dataclass
class Close(Open):
    """
    Same as Open, but will use minimum value as default for goal_joint_state
    """

    def __post_init__(self):
        self.connection = self.environment_link.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        min_position = self.connection.dof.lower_limits.position
        if self.goal_joint_state is None:
            self.goal_joint_state = min_position
        else:
            self.goal_joint_state = max(min_position, self.goal_joint_state)
        super().__post_init__()
