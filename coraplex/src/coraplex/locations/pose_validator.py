from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import List, Optional, Self, TYPE_CHECKING

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.exceptions import NoProgressError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.progress_monitors import ProgressStalled
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from coraplex.plans.plan_node import ActionNode, MotionNode
from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.mixins import HasApproachesGraspPoses

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import EndEffector
from coraplex.exceptions import TipLinkDoesNotMatchAnyArm
from coraplex.locations.base import PoseValidator
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

logger = logging.getLogger("coraplex")


@dataclass
class IsVisibleBy(PoseValidator):
    """
    Validator for checking if either the given pose or body is visible for the robot.

    One has to be given, if both are provided the body is prefered
    """

    target_pose: Pose = field(default=None)
    """
    Pose for which visibility should be checked.
    """

    target_body: Body = field(default=None)
    """
    Body for which visibility should be checked.
    """

    def __call__(self, *args, **kwargs) -> bool:
        if not (self.target_pose or self.target_body):
            raise AttributeError("Either a pose or a body have to be given")
        return self.validate_body() if self.target_body else self.validate_pose()

    def validate_pose(self) -> bool:
        """
        Validates if the target_pose is visible for the robot by creating a temporary
        body at the pose and performing a ray test to see if there is a viewing axis
        between the robot and the target pose.

        :return: True if the target pose is visible for the robot, False otherwise
        """
        gen_body = Body(
            name=PrefixedName("visibility_test_obj", "coraplex"),
            collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        )
        with self.world.modify_world():
            self.world.add_connection(
                FixedConnection(
                    parent=self.world.root,
                    child=gen_body,
                    parent_T_connection_expression=self.target_pose.to_homogeneous_matrix(),
                )
            )

        result = self._ray_test(gen_body)

        if isinstance(self.target_pose, Pose):
            with self.world.modify_world():
                self.world.remove_connection(gen_body.parent_connection)
                self.world.remove_kinematic_structure_entity(gen_body)

        return result

    def validate_body(self) -> bool:
        return self._ray_test(self.target_body)

    def _ray_test(self, target_body: Body) -> bool:
        """
        Performs a ray test from the robot to check if the given body is visible, the
        check filters out bodies of the ' robot form the hit list of the ray test.

        :param target_body: The body for which the ray test is to be performed
        :return: True if the target body is visible for the robot, False otherwise
        """
        ray_tracer = self.world.ray_tracer
        camera = self.robot.get_default_camera()
        ray = ray_tracer.ray_test(
            camera.bodies[0].global_transform.to_position()[:3].to_np(),
            target_body.global_transform.to_position()[:3].to_np(),
            multiple_hits=True,
        )

        hit_bodies = [body for body in ray[2] if body not in self.robot.bodies]

        return hit_bodies[0] == target_body if len(hit_bodies) > 0 else False


@dataclass
class IsReachableBy(PoseValidator):
    """
    Validator that checks if a single pose is reachable with a link of the robot.
    """

    pose: Pose
    """
    Pose that should be reached with the tip_link.
    """

    tip_link: KinematicStructureEntity
    """
    Link that should be moved to the given pose.
    """

    def __call__(self) -> bool:
        return AreReachableBy(
            pose_sequence=[self.pose],
            tip_link=self.tip_link,
            context=self.context,
        ).__call__()


@dataclass
class AreReachableBy(PoseValidator, HasApproachesGraspPoses):
    """
    Validator that checks if a sequence of poses is reachable with the given robot link.

    Poses are addressed in the order they are given.
    """

    pose_sequence: List[Pose]
    """
    Sequence of poses that should be reached.
    """

    tip_link: KinematicStructureEntity
    """
    Link of the robot which should be used for reachability checking.

    ..note:: The poses are goals for this link itself, so a caller checking a grasp
        passes what
        :meth:`~coraplex.robot_plans.mixins.HasApproachesGraspPoses.grasp_pose_sequence`
        produced rather than the grasp frames it was built from.
    """

    @classmethod
    def for_grasp(
        cls,
        grasp_pose: Pose,
        end_effector: EndEffector,
        body_T_grasp: Optional[Pose],
        context: Context,
        **clearances,
    ) -> Self:
        """
        Build a validator for reaching a grasp, rather than for a ready-made sequence.

        Keeps the geometry with the validator instead of with every caller that wants to
        know whether a grasp is within reach.

        :param grasp_pose: The grasp frame to reach.
        :param end_effector: The end effector that is to reach it.
        :param body_T_grasp: The same grasp in the grasped body's frame, or ``None``.
        :param context: The context the check runs in.
        :param clearances: Overrides for :class:`HasApproachesGraspPoses`' distances.
        :return: A validator for the poses reaching that grasp.
        """
        approach = HasApproachesGraspPoses(**clearances)
        return cls(
            pose_sequence=approach.grasp_pose_sequence(
                grasp_pose, end_effector, body_T_grasp
            ),
            tip_link=end_effector.tool_frame,
            context=context,
            **clearances,
        )

    def _arm_reaching_with_the_tip(self) -> Optional[Arms]:
        """
        :return: The arm whose tool frame the sequence moves, or None when the tip is
            not a tool frame of this robot.
        """
        for arm in Arms:
            if (
                self.tip_link
                == ViewManager.get_end_effector_view(arm, self.robot).tool_frame
            ):
                return arm
        return None

    def _gripper_allowance_of_the_reach(self) -> List[UpdateTemporaryCollisionRules]:
        """
        :return: The rule freeing the manipulator that performs this reach, matching
            what the reach itself is executed with. Empty when the tip is not a tool
            frame, since nothing is being grasped with it then.

        A reach onto an object ends inside the buffer zone kept around that object, so a
        probe that does not free the manipulator never converges on the pose it is
        asked about.
        """
        arm = self._arm_reaching_with_the_tip()
        if arm is None:
            return []
        return [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AllowCollisionForEndEffector(
                        end_effector=ViewManager.get_end_effector_view(arm, self.robot)
                    )
                ]
            )
        ]

    def create_msc(self) -> MotionStatechart:
        """
        Creates the Motion state chart to reach the given pose sequence with the given
        tip link.

        Also takes into account if there are alternative motion mappings for moving the
        end effector to the given pose.
        """
        alternative_motion = AlternativeMotion.check_for_alternative(
            self.alternative_motion_mappings, self.robot, MoveToolCenterPointMotion
        )
        if alternative_motion:
            correct_arm = self._arm_reaching_with_the_tip()
            if correct_arm is None:
                raise TipLinkDoesNotMatchAnyArm(self.tip_link, self.robot)
            sequence = []
            for pose in self.pose_sequence:
                motion = alternative_motion(
                    pose,
                    correct_arm,
                    True,
                )
                node = MotionNode(designator=motion)
                # Imagine a plan for the motion node
                plan = Plan(
                    Context(
                        self.world,
                        self.robot,
                        alternative_motion_mappings=self.alternative_motion_mappings,
                    )
                )
                plan.add_node(node)
                motion.plan_node = node
                sequence.append(motion._motion_chart)

        else:
            root = (
                self.robot.root
                if not (
                    self.robot.mobile_base.full_body_controlled
                    if isinstance(self.robot, HasMobileBase)
                    else False
                )
                else self.world.root
            )

            sequence = self.pose_sequence

            tolerances = self.context.motion_tolerances
            sequence = [
                CartesianPose(
                    root_link=root,
                    tip_link=self.tip_link,
                    goal_pose=pose,
                    translation_threshold=tolerances.default_tcp_position_threshold,
                    orientation_threshold=tolerances.tool_orientation_threshold,
                )
                for pose in sequence
            ]

        msc = MotionStatechart()
        msc.add_node(sequence_node := Sequence(sequence))
        if GiskardExecutable.collision_avoidance:
            msc.add_node(ExternalCollisionAvoidance(cancel_if_collision_violated=False))
            msc.add_node(SelfCollisionAvoidance(cancel_if_collision_violated=False))
            msc.add_nodes(self._gripper_allowance_of_the_reach())
        msc.add_node(EndMotion.when_true(sequence_node))
        msc.add_node(stalled := ProgressStalled(monitored_node=sequence_node))
        msc.add_node(stalled.cancel_motion())

        return msc

    def __call__(self, *args, **kwargs) -> bool:
        logger.debug(
            f"Hash of input for pose_sequence_reachability_validator: {hash((*self.pose_sequence, self.tip_link, self.robot))}"
        )

        with (
            self.world.reset_state_context(),
            self.world.collision_manager.reset_temporary_rules_context(),
        ):

            msc = self.create_msc()

            executor = Executor(
                context=MotionStatechartContext(
                    world=self.world,
                    qp_controller_config=QPControllerConfig(
                        target_frequency=50, prediction_horizon=4, verbose=False
                    ),
                ),
            )
            executor.compile(msc)

            try:
                executor.tick_until_end(
                    timeout=len(self.pose_sequence) * GiskardExecutable.ticks_per_motion
                )
            except TimeoutError:
                logger.debug(
                    f"Timeout while executing pose sequence: {self.pose_sequence}"
                )
                return False
            except NoProgressError as no_progress:
                logger.debug(
                    f"Stopped approaching pose sequence {self.pose_sequence}: "
                    f"{no_progress.error_message()}"
                )
                return False
            return True


@dataclass
class IsObjectReachableBy(PoseValidator, HasApproachesGraspPoses):
    """
    Reachability check that is evaluated against a *fresh* copy of the world.

    Both the world copy and the grasp pose sequence are produced inside
    :meth:`__call__`, i.e. when the surrounding condition/monitor is evaluated,
    so the result reflects the current world state instead of the state at the
    time the plan was parsed. The actual reachability simulation is delegated to
    :class:`AreReachableBy` / :class:`IsReachableBy`, which run on the throwaway
    copy so the live world is left untouched.
    """

    arm: Arms
    """
    The arm whose end effector should reach the object.
    """

    object_designator: Body
    """
    The object that should be reachable.
    """

    grasp_pose: Optional[Pose] = field(default=None)
    """
    The grasp frame to reach on the object.

    ``None`` grasps the object at its own frame, which is what ``as_single_grasp``
    always does since it is never given one.
    """

    reverse: bool = field(default=False)
    """
    Whether the grasp pose sequence should be reversed.
    """

    as_single_grasp: bool = field(default=False)
    """
    If set, check reachability of a single grasp pose at the object (used for grasping
    handles of containers) instead of a full pick pose sequence.
    """

    def __call__(self, *args, **kwargs) -> bool:
        world = deepcopy(self.world)
        robot = world.get_semantic_annotation_by_id(self.robot.id)
        end_effector = ViewManager.get_end_effector_view(self.arm, robot)

        grasp_pose = self.grasp_pose
        if grasp_pose is None:
            grasp_pose = Pose(reference_frame=self.object_designator)

        if self.as_single_grasp:
            return IsReachableBy(
                context=Context(
                    world=world,
                    robot=robot,
                    alternative_motion_mappings=self.alternative_motion_mappings,
                ),
                pose=end_effector.tool_frame_goal(grasp_pose),
                tip_link=end_effector.tool_frame,
            ).__call__()

        pose_sequence = self.grasp_pose_sequence(
            grasp_pose,
            end_effector,
            self._grasp_in_body_frame(grasp_pose, self.object_designator),
            reverse=self.reverse,
        )

        return AreReachableBy(
            context=Context(
                world=world,
                robot=robot,
                alternative_motion_mappings=self.alternative_motion_mappings,
            ),
            pose_sequence=pose_sequence,
            tip_link=end_effector.tool_frame,
        ).__call__()
