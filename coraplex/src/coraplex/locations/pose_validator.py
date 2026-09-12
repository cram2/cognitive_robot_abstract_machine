from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field

from abc import ABC

from typing_extensions import List, Optional, Self, Tuple, TYPE_CHECKING

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.exceptions import NoProgressError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from coraplex.plans.plan_node import ActionNode, MotionNode
from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context

try:
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    VizMarkerPublisher = None
from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.mixins import HasApproachesGraspPoses

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import EndEffector
    from semantic_digital_twin.world import World
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
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPoses
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
        arm: Arms,
        *,
        body_T_grasp: Optional[Pose] = None,
        context: Context,
        reverse: bool = False,
        **clearances,
    ) -> Self:
        """
        Build a validator for reaching a grasp, rather than for a ready-made sequence.

        Keeps the geometry with the validator instead of with every caller that wants to
        know whether a grasp is within reach.

        :param grasp_pose: The grasp frame to reach.
        :param arm: The arm that is to reach it.
        :param body_T_grasp: The same grasp in the grasped body's frame, or ``None``.
        :param context: The context the check runs in.
        :param reverse: Whether the gripper withdraws from the grasp rather than moving
            onto it. Each pose of the sequence is where the next one is reached from, so
            a probe running them in the other order answers about a motion that is not
            the one executed.
        :param clearances: Overrides for :class:`HasApproachesGraspPoses`' distances.
        :return: A validator for the poses reaching that grasp.
        """
        end_effector = ViewManager.get_end_effector_view(arm, context.robot)
        approach = HasApproachesGraspPoses(**clearances)
        return cls(
            pose_sequence=approach.grasp_pose_sequence(
                grasp_pose, end_effector, body_T_grasp, reverse=reverse
            ),
            tip_link=end_effector.tool_frame,
            context=context,
            **clearances,
        )

    def _gripper_allowance_of_the_reach(self) -> List[UpdateTemporaryCollisionRules]:
        """
        :return: The rule freeing the manipulator that performs this reach, matching
            what the reach itself is executed with. Empty when the tip is not a tool
            frame, since nothing is being grasped with it then.

        A reach onto an object ends inside the buffer zone kept around that object, so a
        probe that does not free the manipulator never converges on the pose it is
        asked about.
        """
        arm = ViewManager.get_arm_by_tool_frame(self.tip_link, self.robot)
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
            correct_arm = ViewManager.get_arm_by_tool_frame(self.tip_link, self.robot)
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
        msc.add_node(
            still_progressing := StillProgressing(monitored_node=sequence_node)
        )
        msc.add_node(still_progressing.cancel_motion())

        return msc

    def create_executor(self, msc: MotionStatechart) -> Executor:
        """
        Creates the executor that runs a probe of this validator.

        :param msc: The motion statechart the executor is compiled against.
        """
        executor = Executor(
            context=MotionStatechartContext(
                world=self.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
        )
        executor.compile(msc)
        return executor

    def __call__(self, *args, **kwargs) -> bool:
        logger.debug(
            f"Hash of input for pose_sequence_reachability_validator: {hash((*self.pose_sequence, self.tip_link, self.robot))}"
        )

        collision_manager = self.world.collision_manager
        entered_temporary_rules = list(collision_manager.temporary_rules)
        with self.world.reset_state_context():
            try:
                executor = self.create_executor(self.create_msc())

                try:
                    executor.tick_until_end()
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
            finally:
                collision_manager.clear_temporary_rules()
                collision_manager.extend_temporary_rule(entered_temporary_rules)
                collision_manager.update_collision_matrix()


@dataclass
class ReachabilityProbeWorld:
    """
    A throwaway copy of a world, with the robot and gripper found again inside it.

    A reachability check drives the robot to see whether it arrives, which moves it. The
    copy is what gets moved, so the world the plan runs against is left as it was.
    """

    world: World
    """
    The copy itself.
    """

    robot: AbstractRobot
    """
    The robot of :attr:`world`, the same one the caller named.
    """

    end_effector: EndEffector
    """
    The end effector of :attr:`robot` that is to do the reaching.
    """

    source_context: Context
    """
    The context of the world that was copied, whose settings the reach is run with.
    """

    @property
    def context(self) -> Context:
        """
        :return: A context addressing this copy, for validators that run inside it.
        """
        return self.source_context.for_world(self.world, self.robot)


@dataclass
class GraspReachabilityValidator(PoseValidator, HasApproachesGraspPoses, ABC):
    """
    Base for validators answering whether a grasp can be reached from where the robot
    stands.

    The copy and the answer are both produced inside :meth:`__call__`, when the
    surrounding condition is evaluated, so they reflect the world as it is then rather
    than as it was when the plan was built.
    """

    arm: Arms
    """
    The arm whose end effector should do the reaching.
    """

    def _copied_world(self) -> ReachabilityProbeWorld:
        """
        :return: A copy of the world to try the reach in.

        Published to Rviz while debugging: the reach happens in the copy, so a run being
        watched would otherwise show the robot standing still through every grasp it
        judges.
        """
        world = deepcopy(self.world)
        robot = world.get_semantic_annotation_by_id(self.robot.id)
        if self.context.debug and VizMarkerPublisher is not None:
            VizMarkerPublisher(
                _world=world, node=self.context.ros_node
            ).with_collision_visualization()
        return ReachabilityProbeWorld(
            world=world,
            robot=robot,
            end_effector=ViewManager.get_end_effector_view(self.arm, robot),
            source_context=self.context,
        )

    def _reaches(
        self,
        grasp_pose: Pose,
        copied_world: ReachabilityProbeWorld,
        body: Optional[Body],
        reverse: bool = False,
    ) -> bool:
        """
        Whether the gripper can perform the approach onto a grasp and withdraw again.

        :param grasp_pose: The grasp frame to reach, in ``body``'s own frame when there
            is one and in ``copied_world``'s frames either way.
        :param copied_world: The copy to try it in.
        :param body: The body being grasped, that the approach must avoid.
        :param reverse: Whether to withdraw from the grasp rather than move onto it.
        :return: Whether the whole sequence was reached.
        """
        return AreReachableBy(
            context=copied_world.context,
            pose_sequence=self.grasp_pose_sequence(
                grasp_pose,
                copied_world.end_effector,
                grasp_pose if body is not None else None,
                reverse=reverse,
            ),
            tip_link=copied_world.end_effector.tool_frame,
            approach_clearance=self.approach_clearance,
            retreat_distance=self.retreat_distance,
        )()


@dataclass
class IsObjectReachableBy(GraspReachabilityValidator):
    """
    Validator that asks whether an object can be grasped from where the robot stands.

    The grasps are the ones the object itself offers, tried in the order the gripper
    ranks them, and the first that can be reached is kept in :attr:`reachable_grasp` so
    the caller that chose the standing pose also learns which grasp it was chosen for. A
    grasp is only ever reachable from somewhere, so settling on one before a pose is
    known is the wrong way round.
    """

    graspable: HasGraspPoses
    """
    The annotation of the object that should be grasped.
    """

    reachable_grasp: Optional[Pose] = field(default=None, init=False)
    """
    The first grasp found reachable, in the object's own frame.

    ``None`` until a call succeeds, and cleared by every call, so it always belongs to
    the pose the validator was last asked about.
    """

    def _against(self, grasp_pose: Pose, root: Body) -> Pose:
        """
        The same grasp, written against another copy of the body it belongs to.

        A grasp is expressed in its own root's frame, so the numbers naming it on one
        copy of a body name it on any other.

        :param grasp_pose: The grasp to rewrite.
        :param root: The body to write it against.
        :return: The rewritten grasp.
        """
        return Pose(
            position=grasp_pose.to_position(),
            orientation=grasp_pose.to_quaternion(),
            reference_frame=root,
        )

    def _candidates_in(
        self, copied_world: ReachabilityProbeWorld, graspable: HasGraspPoses
    ) -> List[Tuple[Pose, Pose]]:
        """
        The grasps to try, the ones the gripper is closest to first.

        :param copied_world: The copy the reaches are tried in.
        :param graspable: The object, as the copy holds it.
        :return: Pairs of the grasp to try, written against the copy, and the grasp to
            hand back for it, written against the object the caller holds.
        """
        return [
            (grasp_pose, self._against(grasp_pose, self.graspable.root))
            for grasp_pose in copied_world.end_effector.grasp_poses_by_distance(
                graspable
            )
        ]

    def __call__(self, *args, **kwargs) -> bool:
        self.reachable_grasp = None
        copied_world = self._copied_world()
        graspable = copied_world.world.get_semantic_annotation_by_id(self.graspable.id)

        for grasp_pose, reported in self._candidates_in(copied_world, graspable):
            if self._reaches(grasp_pose, copied_world, graspable.root):
                self.reachable_grasp = reported
                return True
        return False


@dataclass
class IsGraspReachableBy(GraspReachabilityValidator):
    """
    Validator that asks whether one named grasp can be reached from where the robot
    stands.

    A caller that has settled on a grasp is asking about that grasp, so no other is
    tried; :class:`IsObjectReachableBy` is the question to ask when any grasp will do.
    """

    grasp_pose: Pose
    """
    The grasp frame to reach, in :attr:`object_designator`'s frame.
    """

    object_designator: Optional[Body] = field(default=None)
    """
    The body being grasped, that the approach must avoid.

    ``None`` when no body is being reached around.
    """

    reverse: bool = field(default=False)
    """
    Whether the gripper withdraws from the grasp rather than moving onto it.
    """

    def __call__(self, *args, **kwargs) -> bool:
        return self._reaches(
            self.grasp_pose,
            self._copied_world(),
            self.object_designator,
            reverse=self.reverse,
        )
