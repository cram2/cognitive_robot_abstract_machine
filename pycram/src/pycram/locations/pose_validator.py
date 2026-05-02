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
from pycram.alternative_motion_mapping import AlternativeMotion
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.locations.base import PoseValidator
from pycram.plans.plan import Plan
from pycram.plans.plan_node import PlanNode
from pycram.robot_plans import MoveToolCenterPointMotion
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

logger = logging.getLogger("pycram")


@dataclass
class VisibilityValidator(PoseValidator):

    target_pose: Pose = field(default=None)

    target_body: Body = field(default=None)

    def __call__(self, *args, **kwargs) -> bool:
        assert self.target_pose or self.target_body
        return self.validate_body() if self.target_body else self.validate_pose()

    def validate_pose(self) -> bool:
        gen_body = Body(
            name=PrefixedName("vist_test_obj", "pycram"),
            collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        )
        with self.world.modify_world():
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=self.world.root, child=gen_body, world=self.world
                )
            )
        gen_body.parent_connection.origin = self.target_pose.to_homogeneous_matrix()

        result = self._ray_test(gen_body)

        if isinstance(self.target_pose, Pose):
            with self.world.modify_world():
                self.world.remove_connection(gen_body.parent_connection)
                self.world.remove_kinematic_structure_entity(gen_body)

        return result

    def validate_body(self) -> bool:
        return self._ray_test(self.target_body)

    def _ray_test(self, target_body: Body) -> bool:
        r_t = self.world.ray_tracer
        camera = self.robot.get_default_camera()
        ray = r_t.ray_test(
            camera.bodies[0].global_transform.to_position().to_np()[:3],
            target_body.global_transform.to_position().to_np()[:3],
            multiple_hits=True,
        )

        hit_bodies = [b for b in ray[2] if not b in self.robot.bodies]

        return hit_bodies[0] == target_body if len(hit_bodies) > 0 else False

@dataclass
class ReachabilityValidator(PoseValidator):

    pose: Pose

    tip_link: KinematicStructureEntity

    def __call__(self) -> bool:
        return ReachabilitySequenceValidator(
            pose_sequence=[self.pose],
            tip_link=self.tip_link,
            robot=self.robot,
            world=self.world,
        ).__call__()


@dataclass
class ReachabilitySequenceValidator(PoseValidator):

    pose_sequence: List[Pose]

    tip_link: KinematicStructureEntity

    def create_msc(self):
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
                motion = alternative_motion(pose, correct_arm, True)
                node = PlanNode()
                # Imagine a plan for the motion node
                plan = Plan(Context(self.world, self.robot))
                plan.add_node(node)
                motion.plan_node = node
                sequence.append(motion._motion_chart)

        else:
            root = (
                self.robot.root
                if not self.robot.full_body_controlled
                else self.world.root
            )
            sequence = [
                CartesianPose(root_link=root, tip_link=self.tip_link, goal_pose=pose)
                for pose in self.pose_sequence
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