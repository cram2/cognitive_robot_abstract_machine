"""
Stretch fetches a cereal box from a shelf, places it on a bedside table and puts it
back again.

Running with :class:`~coraplex.plans.executors.RealRobotPlanExecutor` drives the actual
robot and fetches the world from the running world server. The default runs the whole plan
in simulation against a world built from the Stretch URDF, so nothing on the network is
needed.

..warning:: A real run needs a perception pipeline answering
    :data:`~coraplex.perception.ROBOKUDO_QUERY_ACTION_NAME` as well as the world server:
    the plan detects the cereal before grasping it, and raises
    :class:`~coraplex.exceptions.PerceptionSourceUnavailable` rather than grasping at the
    pose it was spawned with. Simulated runs read the world model instead and need
    neither.

Start the controller and the pipeline separately, as on the robot. RoboKudo's entry point
takes its arguments with an underscore prefix::

    python3 -m robokudo.scripts.main _ae stretch_demo _headless

That engine localizes whatever stands on the dominant plane in front of the Stretch's
RealSense but does not recognize it, so it reports poses without class labels and the
detection is labelled with what the plan asked for. Two consequences: a second object on the
same shelf layer aborts the run with
:class:`~coraplex.exceptions.UnidentifiedDetections` rather than risking the wrong grasp,
and an empty scene fails the detection, which ends the run with
:class:`~coraplex.exceptions.MotionDidNotFinish`. Adding a
classifying annotator to that engine removes both without changing this demo.
"""

from dataclasses import dataclass, field

import numpy as np

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.plan_transformation import PlanRewriting
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from coraplex.datastructures.enums import DetectionTechnique
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.executors import (
    PlanExecutor,
    RealRobotPlanExecutor,
    KinematicallySimulatedPlanExecutor,
)
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.robot_plans.plan_transformations import DetectBeforeGrasp
from krrood.entity_query_language.factories import a
from semantic_digital_twin.specifications.connections import Connection6DoFSpecification
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.predefined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.semantic_annotations.semantic_annotations import CheezeIt
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from cramph.composites import Sequence, StatementNode

CEREAL_NAME = "cheeze_it.obj"
"""
Name of the transported body, which also marks whether the scene was already spawned.
"""

CEREAL_SHELF_LAYER_NAME = "shelf_layer2"
"""
Name of the shelf layer the cereal starts on.
"""

CEREAL_SHELF_LAYER_T_CEREAL = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=-0.1, y=0.0, z=0.115
)
"""
Where the cereal starts, relative to its shelf layer.

Only a prior: a detection before the pick-up overwrites it with a perceived pose.
"""

LIVE_DEMONSTRATION_REPETITIONS = 5
"""
How often running this module as a script transports the cereal there and back again.
"""


@dataclass
class StretchApartmentDemonstration(RobotDemonstration):
    """
    Stretch transports a cereal box between a shelf and a bedside table in the apartment.

    The plan ends where it started, so it can be repeated against the same scene.
    """

    ros_node_name: str = "stretch_demo_node"

    def scene_specification(self) -> WorldSpecification:
        """
        :return: A world holding nothing but a map body, which :meth:`populate_scene`
            furnishes.
        """
        return WorldSpecification(world_parser=None)

    def robot_specification(self) -> RobotSpecification:
        """
        :return: The Stretch on its drive, between the shelf and the bedside table.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(1, 1),
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(CEREAL_NAME)

    def populate_scene(self, world: World) -> None:
        """
        Spawn the apartment and the cereal box on its shelf.
        """
        ApartmentEnvironment().populate(world)

        # %% cereal

        # The cereal is the one body perception moves, and a pose can only be written to
        # a connection that has the degrees of freedom to carry it, so it hangs off a
        # 6DoF connection rather than the default fixed one.
        CheezeIt.get_annotation_specification(
            CEREAL_NAME,
            BodySpecification.mesh(
                CEREAL_NAME,
                ApartmentEnvironment.mesh_path(CEREAL_NAME),
                parent_T_self=CEREAL_SHELF_LAYER_T_CEREAL,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.get_body_by_name(CEREAL_SHELF_LAYER_NAME))

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        """
        Give the plan the Stretch in ``world``, detecting an object before grasping it.
        """
        return [
            RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0]),
            PlanRewriting(transformations=[DetectBeforeGrasp()]),
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Carry the cereal box from its shelf to the bedside table and back again.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot

        cereal = world.get_semantic_annotations_by_type(CheezeIt)[0]
        arm = robot.all_arms[0]
        shelf_layer_body = world.get_body_by_name(CEREAL_SHELF_LAYER_NAME)
        bedside_table_body = world.get_body_by_name("bedside_table.dae")
        CEREAL_SHELF_LAYER_T_CEREAL.reference_frame = shelf_layer_body

        plan = Sequence(
            nodes=[
                ParkArmsAction(arms=robot.all_arms),
                SetGripperAction(
                    configuration=arm.end_effector.default_configuration(
                        GripperState.CLOSE
                    )
                ),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        1.2, 1.2, 0, yaw=np.pi, reference_frame=world.root
                    )
                ),
                LookAtAction(
                    target=Pose.from_xyz_rpy(reference_frame=shelf_layer_body)
                ),
                StatementNode(
                    statement=a(NavigateAction)(
                        target_location=Pose.from_xyz_rpy(
                            0.8, 0.6, 0, yaw=-np.pi / 2, reference_frame=world.root
                        )
                    )
                ),
                LookAtAction(
                    target=Pose.from_xyz_rpy(reference_frame=shelf_layer_body)
                ),
                DetectAction(
                    technique=DetectionTechnique.TYPES,
                    object_sem_annotation=CheezeIt,
                    trust_detected_orientation=True,
                    accept_first_if_multiple=True,
                ),
                PickUpAction(
                    grasp=cereal.grasp_candidates()[0],
                    arm=arm,
                ),
                ParkArmsAction(arms=robot.all_arms),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        0.8, 0, 0, yaw=np.pi, reference_frame=bedside_table_body
                    )
                ),
                PlaceAction(
                    object_designator=cereal,
                    target_location=Pose.from_xyz_rpy(
                        x=0.1,
                        z=0.56,
                        yaw=np.pi,
                        reference_frame=bedside_table_body,
                    ),
                ),
                ParkArmsAction(arms=robot.all_arms),
                SetGripperAction(
                    configuration=arm.end_effector.default_configuration(
                        GripperState.CLOSE
                    )
                ),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        1.2, 1.2, 0, yaw=np.pi, reference_frame=world.root
                    )
                ),
                LookAtAction(
                    target=Pose.from_xyz_rpy(reference_frame=shelf_layer_body)
                ),
                StatementNode(
                    statement=a(NavigateAction)(
                        target_location=Pose.from_xyz_rpy(
                            0.8, 0, 0, yaw=np.pi, reference_frame=bedside_table_body
                        )
                    )
                ),
                LookAtAction(
                    target=Pose.from_xyz_rpy(reference_frame=bedside_table_body)
                ),
                DetectAction(
                    technique=DetectionTechnique.TYPES,
                    object_sem_annotation=CheezeIt,
                    trust_detected_orientation=False,
                    accept_first_if_multiple=True,
                ),
                StatementNode(
                    statement=a(PickUpAction)(
                        grasp=cereal.grasp_candidates()[0],
                        arm=arm,
                    )
                ),
                ParkArmsAction(arms=robot.all_arms),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        0.8, 0.6, 0, yaw=-np.pi / 2, reference_frame=world.root
                    )
                ),
                StatementNode(
                    statement=a(PlaceAction)(
                        object_designator=cereal,
                        target_location=CEREAL_SHELF_LAYER_T_CEREAL.pose,
                    )
                ),
                ParkArmsAction(arms=robot.all_arms),
                SetGripperAction(
                    configuration=arm.end_effector.default_configuration(
                        GripperState.CLOSE
                    )
                ),
            ]
        )

        statechart = Statechart(context=context)
        statechart.add_node(plan)
        return statechart


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
    repetitions: int = 1,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it
        drives the real robot or simulates it.
    :param repetitions: How often to transport the cereal there and back again.
    """
    # StretchApartmentDemonstration(used_robot=PR2, executor_type=executor_type).run()
    # StretchApartmentDemonstration(used_robot=HSRB, executor_type=executor_type).run()
    StretchApartmentDemonstration(
        used_robot=Stretch, executor_type=executor_type, repetitions=repetitions
    ).run()


if __name__ == "__main__":
    main(
        executor_type=RealRobotPlanExecutor, repetitions=LIVE_DEMONSTRATION_REPETITIONS
    )
