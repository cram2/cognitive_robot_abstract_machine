"""
Plan cereal grasps with Simox and execute standard CoraPlex actions in simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from coraplex.datastructures.dataclasses import Context
from coraplex.demonstrations import RobotDemonstration, RobotDemonstrationRosSession
from coraplex.external_interfaces.simox_grasp_planner import (
    SimoxConfiguration,
    SimoxGraspPlanner,
    RosSimoxService,
)
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import PickAndPlaceAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from krrood.entity_query_language.factories import a, variable_from
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification

if TYPE_CHECKING:
    from coraplex.plans.plan_node import PlanNode
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World

# %% simulation scene and actions


@dataclass
class SimoxDemonstration(RobotDemonstration):
    """
    Try ranked Simox candidates through the shared manipulation pipeline.
    """

    used_robot: type[AbstractRobot] = PR2
    """
    Robot whose tool calibration is supplied to the planner.
    """

    planner: SimoxGraspPlanner = field(kw_only=True)
    """
    External grasp planner with an active transport.
    """

    resource_directory: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "resources"
    )
    """
    Directory containing the apartment and cereal meshes.
    """

    cereal_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.0487
        )
    )
    """
    Cereal placement in the simulated apartment, in metres.
    """

    robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(1.72, 1.8)
    )
    """
    Initial simulated robot base placement.
    """

    placement_offset: float = 0.2
    """
    Cereal displacement along the world's y axis, in metres.
    """

    torso_state: TorsoState = TorsoState.MID
    """
    Torso posture for approaching the simulated counter.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification.from_urdf(
            str(self.resource_directory / "worlds/apartment.urdf")
        )

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot, world_T_odom=self.robot_start)

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(Cereal))

    def populate_scene(self, world: World) -> None:
        SemanticAnnotationWithRootSpecification(
            name="cereal",
            semantic_annotation_type=Cereal,
            root_specification=BodySpecification.mesh(
                "cereal", str(self.resource_directory / "objects/breakfast_cereal.stl")
            ),
        ).spawn(world, parent_T_self=self.cereal_start)

    def build_context(self, world: World) -> Context:
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(PR2)[0],
            ros_node=self.ros_node,
            sampling_seed=0,
        )

    def build_plan(self, context: Context) -> PlanNode:
        robot = context.robot
        cereal = context.world.get_semantic_annotations_by_type(Cereal)[0]
        arm = robot.get_right_arm_if_specified()
        results = self.planner.plan_grasps_for_body(cereal, arm, robot.root)
        cereal_pose = cereal.root.global_pose
        target = Pose(
            position=Point3(
                float(cereal_pose.position.x),
                float(cereal_pose.position.y) + self.placement_offset,
                float(cereal_pose.position.z),
            ),
            orientation=cereal_pose.orientation,
            reference_frame=context.world.root,
        )
        pick_and_place = a(PickAndPlaceAction)(
            pick_up=a(PickUpAction)(
                grasp=variable_from([result.candidate for result in results]), arm=arm
            ),
            place=PlaceAction(object_designator=cereal, target_location=target),
        )
        return sequential(
            [
                ParkArmsAction(list(robot.all_arms)),
                MoveTorsoAction(self.torso_state),
                pick_and_place,
                ParkArmsAction(list(robot.all_arms)),
            ],
            context=context,
        )


# %% command line


def main() -> None:
    """
    Run the simulated PR2 with an independently running local Simox service.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--num-grasps", type=int, default=15)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    arguments = parser.parse_args()
    configuration = SimoxConfiguration.for_pr2(arguments.robot_xml.resolve())
    configuration.num_grasps_to_plan = arguments.num_grasps
    configuration.timeout_ms = arguments.timeout_ms
    with RobotDemonstrationRosSession.start("simox_demo_client") as session:
        planner = SimoxGraspPlanner(configuration, RosSimoxService(session.node))
        demonstration = SimoxDemonstration(planner=planner)
        try:
            demonstration.run()
        finally:
            demonstration.stop_visualization()


if __name__ == "__main__":
    main()
