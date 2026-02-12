from __future__ import division

from dataclasses import dataclass, field

from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    RotationMatrix,
)
from semantic_digital_twin.world_description.connections import DiffDrive
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from .templates import Sequence, Parallel
from ..binding_policy import GoalBindingPolicy
from ..context import BuildContext
from ..data_types import DefaultWeights
from ..exceptions import NodeInitializationError
from ..graph_node import Goal, MotionStatechartNode
from ..tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPositionStraight,
    CartesianPose,
)


@dataclass(eq=False, repr=False)
class DiffDriveBaseGoal(Sequence):
    """
    A sequence that moves the robot to a goal pose using a differential drive.
    1. Orient to goal position
    2. Drive to goal position
    3. Orient to goal orientation
    """

    diff_drive_connection: DiffDrive | None = field(kw_only=True, default=None)
    """Drive connection to use. If it is None and there is only one diff drive in the world, it will be used."""

    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    """Pose to reach."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_CA
    """Task priority relative to other tasks."""

    nodes: list[MotionStatechartNode] = field(default_factory=list, init=False)

    def expand(self, context: BuildContext) -> None:
        if self.diff_drive_connection is None:
            diff_drives = context.world.get_connections_by_type(DiffDrive)
            if len(diff_drives) == 0:
                raise NodeInitializationError(self, "No diff drives found in world.")
            if len(diff_drives) > 1:
                raise NodeInitializationError(
                    self, "More than one diff drive found in world."
                )
            self.diff_drive_connection = diff_drives[0]
        map = context.world.root
        tip = self.diff_drive_connection.child

        root_T_goal = context.world.transform(self.goal_pose, map)
        root_T_current = tip.global_pose
        root_V_current_to_goal = (
            root_T_goal.to_position() - root_T_current.to_position()
        )
        root_V_current_to_goal.scale(1)
        root_V_z = Vector3.Z(reference_frame=map)
        root_R_first_orientation = RotationMatrix.from_vectors(
            x=root_V_current_to_goal, z=root_V_z, reference_frame=map
        )

        root_T_goal2 = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=root_T_goal.to_position(), rotation_matrix=root_R_first_orientation
        )

        self.nodes = [
            CartesianOrientation(
                name=f"{self.name}/step1",
                root_link=map,
                tip_link=tip,
                goal_orientation=root_R_first_orientation,
                weight=self.weight,
            ),
            CartesianPose(
                name=f"{self.name}/step2",
                root_link=map,
                tip_link=tip,
                goal_pose=root_T_goal2,
                weight=self.weight,
            ),
            CartesianPose(
                name=f"{self.name}/step3",
                root_link=map,
                tip_link=tip,
                goal_pose=root_T_goal,
                weight=self.weight,
            ),
        ]
        super().expand(context)


@dataclass(eq=False, repr=False)
class CartesianPoseStraight(Parallel):
    """
    Like CartesianPose, but constrains the tip link to move in a straight line towards the goal.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Name of the root link of the kin chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Name of the tip link of the kin chain."""

    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    """The goal pose."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_CA
    """Task priority relative to other tasks."""

    binding_policy: GoalBindingPolicy = field(
        default=GoalBindingPolicy.Bind_at_build, kw_only=True
    )
    """Describes when the goal is computed. See GoalBindingPolicy for more information."""

    nodes: list[MotionStatechartNode] = field(default_factory=list, init=False)

    def expand(self, context: BuildContext) -> None:
        self.nodes = [
            CartesianPositionStraight(
                name=self.name + "/position",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_point=self.goal_pose.to_position(),
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
            CartesianOrientation(
                name=self.name + "/orientation",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_orientation=self.goal_pose.to_rotation_matrix(),
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
        ]
        super().expand(context)


@dataclass(eq=False, repr=False)
class RelativePositionSequence(Goal):
    goal1: HomogeneousTransformationMatrix = field(kw_only=True)
    goal2: HomogeneousTransformationMatrix = field(kw_only=True)
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)

    def __post_init__(self):
        """
        Only meant for testing.
        """
        name1 = f"{self.name}/goal1"
        name2 = f"{self.name}/goal2"
        task1 = CartesianPose(
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_pose=self.goal1,
            name=name1,
            absolute=True,
        )
        self.add_task(task1)
        task2 = CartesianPose(
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_pose=self.goal2,
            name=name2,
            absolute=True,
        )
        self.add_task(task2)
        task2.start_condition = task1
        task1.end_condition = task1
        self.observation_expression = task2.observation_expression
