import logging
from dataclasses import dataclass, field

from typing_extensions import List

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.base import PoseValidator
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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
    Validator for checking if either the given pose or body is visible for the robot. One has to be given, if both are
    provided the body is prefered
    """

    target_pose: Pose = field(default=None)
    """
    Pose for which visibility should be checked
    """

    target_body: Body = field(default=None)
    """
    Body for which visibility should be checked
    """

    def __call__(self, *args, **kwargs) -> bool:
        if not (self.target_pose or self.target_body):
            raise AttributeError("Either a pose or a body have to be given")
        return self.validate_body() if self.target_body else self.validate_pose()

    def validate_pose(self) -> bool:
        """
        Validates if the target_pose is visible for the robot by creating a temporary body at the pose and performing
        a ray test to see if there is a viewing axis between the robot and the target pose.

        :return: True if the target pose is visible for the robot, False otherwise
        """
        gen_body = Body(
            name=PrefixedName("vist_test_obj", "coraplex"),
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
        Performs a ray test from the robot to check if the given body is visible, the check filters out bodies of the '
        robot form the hit list of the ray test.

        :param target_body: The body for which the ray test is to be performed
        :return: True if the target body is visible for the robot, False otherwise
        """
        r_t = self.world.ray_tracer
        camera = self.robot.get_default_camera()
        ray = r_t.ray_test(
            camera.bodies[0].global_transform.to_position()[:3].to_np(),
            target_body.global_transform.to_position()[:3].to_np(),
            multiple_hits=True,
        )

        hit_bodies = [b for b in ray[2] if not b in self.robot.bodies]

        return hit_bodies[0] == target_body if len(hit_bodies) > 0 else False


@dataclass
class IsReachableBy(PoseValidator):
    """
    Validator that checks if a single pose is reachable with a link of the robot.
    """

    pose: Pose
    """
    Pose that should be reached with the tip_link
    """

    tip_link: KinematicStructureEntity
    """
    Link that should be moved to the given pose
    """

    grasp_description: GraspDescription = field(default=None)
    """
    The grasp description that should be used for validation
    """

    def __call__(self) -> bool:
        return AreReachableBy(
            pose_sequence=[self.pose],
            tip_link=self.tip_link,
            robot=self.robot,
            world=self.world,
            grasp_description=self.grasp_description,
        ).__call__()


@dataclass
class AreReachableBy(PoseValidator):
    """
    Validator that checks if a sequence of poses is reachable with the given robot link. Poses are addressed in the
    order they are given.
    """

    pose_sequence: List[Pose]
    """
    Sequence of poses that should be reached
    """

    tip_link: KinematicStructureEntity
    """
    Link of the robot which should be used for reachability checking
    """

    grasp_description: GraspDescription = field(default=None)
    """
    The grasp description that should be used for validation
    """

    def create_msc(self) -> MotionStatechart:
        """
        Creates the Motion state chart to reach the given pose sequence with the given tip link. Also takes into account
        if there are alternative motion mappings for moving the end effector to the given pose.
        """
        alternative_motion = AlternativeMotion.check_for_alternative(
            self.robot, MoveToolCenterPointMotion
        )
        if alternative_motion:
            correct_arm = None
            for arm in Arms:
                if (
                    self.tip_link
                    == ViewManager.get_end_effector_view(arm, self.robot).tool_frame
                ):
                    correct_arm = arm
            sequence = []
            for pose in self.pose_sequence:

                if self.grasp_description:
                    pose = self.grasp_description._pose_sequence(pose)[1]

                motion = alternative_motion(
                    pose,
                    correct_arm,
                    True,
                )
                node = PlanNode()
                # Imagine a plan for the motion node
                plan = Plan(Context(self.world, self.robot))
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

            sequence = (
                [
                    self.grasp_description._pose_sequence(pose)[1]
                    for pose in self.pose_sequence
                ]
                if self.grasp_description
                else self.pose_sequence
            )

            sequence = [
                CartesianPose(root_link=root, tip_link=self.tip_link, goal_pose=pose)
                for pose in sequence
            ]

        msc = MotionStatechart()
        msc.add_node(n := Sequence(sequence))
        msc.add_node(EndMotion.when_true(n))

        return msc

    def __call__(self, *args, **kwargs) -> bool:
        logger.debug(
            f"Hash of input for pose_sequence_reachability_validator: {hash((*self.pose_sequence, self.tip_link, self.robot))}"
        )

        with self.world.reset_state_context():

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
                executor.tick_until_end()
            except TimeoutError:
                logger.debug(
                    f"Timeout while executing pose sequence: {self.pose_sequence}"
                )
                return False
            return True
