from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any, Self

from cramph.composites import Parallel, Sequence
from cramph.node import (
    CompositeNode,
    FailsOnObservingFalse,
    NodeArtifacts,
    StatechartNode,
    SucceedsOnObservingTrue,
    expanded_child_field,
)
from giskardpy.motion_statechart.exceptions import RobotWithoutMobileBaseError
from giskardpy.motion_statechart.graph_node import BindsGoalReferenceFrame
from giskardpy.motion_statechart.mixins import (
    HasGoalPoseParameter,
    HasThresholdParameter,
    HasWeightParameter,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.tasks.drive_tasks import (
    BaseHeading,
    BaseOrientation,
    BasePosition,
    BaseTurnTowardsGoalPosition,
)
from krrood.symbolic_math.symbolic_math import Scalar, trinary_if_cases
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot, MobileBase
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world_description.connections import DifferentialDrive
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% putting the base of a robot at a pose


class MobileBasePoseGoal(ABC):
    """
    A node putting the base of a robot at a pose, built for any robot the same way, so
    which kind of node does it can be chosen without knowing what each one needs.
    """

    @classmethod
    @abstractmethod
    def moving_base_of(
        cls, robot: AbstractRobot, goal_pose: Pose, threshold: float
    ) -> Self:
        """
        :param robot: The robot whose base is put at `goal_pose`.
        :param goal_pose: Where the root of `robot` should end up.
        :param threshold: How far the base may be from `goal_pose`, in meters and in
            radians, to count as arrived.
        :return: A new node putting the base of `robot` at `goal_pose`.
        """


# %% driving the base along the floor


@dataclass(eq=False, repr=False, kw_only=True)
class DrivenBaseGoal(
    SucceedsOnObservingTrue,
    FailsOnObservingFalse,
    CompositeNode,
    MobileBasePoseGoal,
    BindsGoalReferenceFrame,
    HasGoalPoseParameter,
    HasWeightParameter,
    HasThresholdParameter,
    ABC,
):
    """
    Drives a base along the floor and turns it about the vertical until it stands below
    :attr:`goal_pose`, facing its way. Its threshold holds in meters and in radians.

    The base is moved relative to the body it is attached to, as it is whenever the
    tasks are built, so the goal follows a base moved under another parent while the
    statechart runs. Where the goal lies is read from its reference frame as
    :attr:`binding_policy` says.
    """

    mobile_base: MobileBase
    """
    The base moved, by the drive attaching its robot to its parent.
    """

    root_T_goal_reference_frame: HomogeneousTransformationMatrix | None = (
        expanded_child_field(default=None)
    )
    """
    Differs from :attr:`BindsGoalReferenceFrame.root_T_goal_reference_frame` in being
    created by :meth:`expand` and sent along with this goal once it expanded, since the
    tasks of :attr:`motion` are expressed over it.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_pose.reference_frame

    @property
    def driven_body(self) -> KinematicStructureEntity:
        """
        :return: The body the drive of :attr:`mobile_base` carries.
        """
        return self.mobile_base.drive.child

    @property
    @abstractmethod
    def motion(self) -> StatechartNode:
        """
        :return: The node running the tasks, once :meth:`_create_motion` created it.
        """

    @abstractmethod
    def _create_motion(self, task_parameters: dict[str, Any]) -> None:
        """
        Creates the node running the tasks that move :attr:`driven_body`.

        :param task_parameters: The parameters every task is created with.
        """

    @classmethod
    def moving_base_of(
        cls, robot: AbstractRobot, goal_pose: Pose, threshold: float
    ) -> Self:
        """
        Differs from :meth:`MobileBasePoseGoal.moving_base_of` in requiring `robot` to
        have a mobile base.

        :raises RobotWithoutMobileBaseError: If `robot` has no mobile base.
        """
        if not isinstance(robot, HasMobileBase) or robot.mobile_base is None:
            raise RobotWithoutMobileBaseError(robot=robot)
        return cls(
            mobile_base=robot.mobile_base, goal_pose=goal_pose, threshold=threshold
        )

    @property
    def _drive_parent(self) -> KinematicStructureEntity:
        """
        :return: The body :attr:`driven_body` is attached to now.
        """
        return self.driven_body.parent_kinematic_structure_entity

    @property
    def goal_binding_root(self) -> KinematicStructureEntity:
        return self._drive_parent

    def _root_T_goal_reference_frame_to_bind(self) -> HomogeneousTransformationMatrix:
        """
        Differs from
        :meth:`BindsGoalReferenceFrame._root_T_goal_reference_frame_to_bind` in binding
        the matrix :meth:`expand` created, which the tasks are expressed over.
        """
        return self.root_T_goal_reference_frame

    def expand(self) -> None:
        """
        Add the node running the tasks.
        """
        self.root_T_goal_reference_frame = (
            HomogeneousTransformationMatrix.create_with_variables(
                str(self._goal_reference_frame_name)
            )
        )
        self._create_motion(
            dict(
                driven_body=self.driven_body,
                goal_pose=self.goal_pose,
                parent_T_goal_reference_frame=self.root_T_goal_reference_frame,
                weight=self.weight,
                threshold=self.threshold,
            )
        )
        self._add_child_to_statechart(self.motion)
        self._succeed_at_its_goal(self.motion)

    def build_artifacts(self) -> NodeArtifacts:
        """
        Report the outcome of :attr:`motion`, and express
        :attr:`root_T_goal_reference_frame` relative to the new parent of the base if it
        was moved under another parent, since the tasks are rebuilt against it.

        Where the goal's reference frame was bound is kept.
        """
        if self._goal_reference_frame_binding.root is not self._drive_parent:
            self._goal_reference_frame_binding.change_root(
                self._drive_parent, self.context.world
            )
        return NodeArtifacts(
            observation=trinary_if_cases(
                cases=[
                    (self.motion.is_succeeded, Scalar.const_true()),
                    (self.motion.is_failed_or_interrupted, Scalar.const_false()),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


@dataclass(eq=False, repr=False, kw_only=True)
class HolonomicBaseGoal(DrivenBaseGoal):
    """
    Moves the base to the pose in whatever direction it lies, sideways included, while
    turning it.
    """

    parallel: Parallel = expanded_child_field()
    """
    The parallel running the position and the orientation task.
    """

    @property
    def motion(self) -> StatechartNode:
        return self.parallel

    def _create_motion(self, task_parameters: dict[str, Any]) -> None:
        self.parallel = Parallel(
            name=f"{self.name}/parallel",
            nodes=[
                BasePosition(name=f"{self.name}/position", **task_parameters),
                BaseOrientation(name=f"{self.name}/orientation", **task_parameters),
            ],
        )


@dataclass(eq=False, repr=False, kw_only=True)
class DifferentialDriveBaseGoal(DrivenBaseGoal):
    """
    Moves the robot to a goal pose using a differential drive, running these steps in
    one :class:`~cramph.composites.Sequence`:

    1. Turn towards the goal position, without driving
    2. Drive to goal position, steering towards it
    3. Orient to goal orientation
    """

    mobile_base: MobileBase[DifferentialDrive]
    """
    The base moved, by the differential drive attaching its robot to its parent.
    """

    sequence: Sequence = expanded_child_field()
    """
    The sequence running the three steps.
    """

    @property
    def motion(self) -> StatechartNode:
        return self.sequence

    def _create_motion(self, task_parameters: dict[str, Any]) -> None:
        steps = [
            BaseTurnTowardsGoalPosition(name=f"{self.name}/step1", **task_parameters),
            Parallel(
                name=f"{self.name}/step2",
                nodes=[
                    BasePosition(name=f"{self.name}/step2/position", **task_parameters),
                    BaseHeading(name=f"{self.name}/step2/heading", **task_parameters),
                ],
            ),
            Parallel(
                name=f"{self.name}/step3",
                nodes=[
                    BasePosition(name=f"{self.name}/step3/position", **task_parameters),
                    BaseOrientation(
                        name=f"{self.name}/step3/orientation", **task_parameters
                    ),
                ],
            ),
        ]
        self.sequence = Sequence(name=f"{self.name}/sequence", nodes=steps)
