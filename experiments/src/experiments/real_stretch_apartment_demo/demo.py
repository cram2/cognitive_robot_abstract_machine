"""
Stretch fetches a cereal box from a shelf, places it on a bedside table and puts it
back again.

Running with :attr:`~coraplex.datastructures.enums.ExecutionType.REAL` drives the actual
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
and an empty scene aborts with :class:`~coraplex.exceptions.NothingDetected`. Adding a
classifying annotator to that engine removes both without changing this demo.
"""

from dataclasses import dataclass

import numpy as np
from typing_extensions import ClassVar

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    DetectionTechnique,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import a
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.semantic_annotations.semantic_annotations import CheezeIt
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

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

    ros_node_name: ClassVar[str] = "stretch_demo_node"

    def build_simulated_world(self) -> World:
        """
        Put the Stretch on its drive in a world holding nothing but a map body.
        """
        return WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(
                        1, 1
                    ),
                )
            ],
        ).to_domain_object()

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

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the Stretch in ``world``.

        ..note:: The ROS node has to be in the context for a real robot.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry the cereal box from its shelf to the bedside table and back again.
        """
        world = context.world

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_arm_view(Arms.LEFT, context.robot).end_effector,
        )

        cereal = world.get_semantic_annotations_by_type(CheezeIt)[0]
        cereal_body = cereal.root
        shelf_layer_body = world.get_body_by_name(CEREAL_SHELF_LAYER_NAME)
        bedside_table_body = world.get_body_by_name("bedside_table.dae")
        CEREAL_SHELF_LAYER_T_CEREAL.reference_frame = shelf_layer_body

        plan = sequential(
            [
                ParkArmsAction(Arms.BOTH),
                SetGripperAction(Arms.BOTH, motion=GripperState.CLOSE),
                NavigateAction(
                    Pose.from_xyz_rpy(
                        1.2, 1.2, 0, yaw=np.pi, reference_frame=world.root
                    )
                ),
                LookAtAction(Pose.from_xyz_rpy(reference_frame=shelf_layer_body)),
                a(NavigateAction)(
                    target_location=Pose.from_xyz_rpy(
                        0.8, 0.6, 0, yaw=-np.pi / 2, reference_frame=world.root
                    )
                ),
                LookAtAction(Pose.from_xyz_rpy(reference_frame=shelf_layer_body)),
                DetectAction(
                    DetectionTechnique.TYPES,
                    object_sem_annotation=CheezeIt,
                    trust_detected_orientation=True,
                    accept_first_if_multiple=True,
                ),
                PickUpAction(
                    cereal, Arms.LEFT, grasp_description, perceive_before_grasp=True
                ),
                ParkArmsAction(Arms.BOTH),
                NavigateAction(
                    Pose.from_xyz_rpy(
                        0.8, 0, 0, yaw=np.pi, reference_frame=bedside_table_body
                    )
                ),
                PlaceAction(
                    object_designator=cereal_body,
                    target_location=Pose.from_xyz_rpy(
                        x=0.1,
                        z=0.56,
                        yaw=np.pi,
                        reference_frame=bedside_table_body,
                    ),
                    arm=Arms.LEFT,
                ),
                ParkArmsAction(Arms.BOTH),
                SetGripperAction(Arms.BOTH, motion=GripperState.CLOSE),
                NavigateAction(
                    Pose.from_xyz_rpy(
                        1.2, 1.2, 0, yaw=np.pi, reference_frame=world.root
                    )
                ),
                LookAtAction(Pose.from_xyz_rpy(reference_frame=shelf_layer_body)),
                a(NavigateAction)(
                    target_location=Pose.from_xyz_rpy(
                        0.8, 0, 0, yaw=np.pi, reference_frame=bedside_table_body
                    )
                ),
                LookAtAction(Pose.from_xyz_rpy(reference_frame=bedside_table_body)),
                DetectAction(
                    DetectionTechnique.TYPES,
                    object_sem_annotation=CheezeIt,
                    trust_detected_orientation=False,
                    accept_first_if_multiple=True,
                ),
                a(PickUpAction)(
                    object_designator=cereal,
                    arm=Arms.LEFT,
                    grasp_description=grasp_description,
                    perceive_before_grasp=True,
                ),
                ParkArmsAction(Arms.BOTH),
                NavigateAction(
                    Pose.from_xyz_rpy(
                        0.8, 0.6, 0, yaw=-np.pi / 2, reference_frame=world.root
                    )
                ),
                a(PlaceAction)(
                    object_designator=cereal_body,
                    target_location=CEREAL_SHELF_LAYER_T_CEREAL.to_pose(),
                    arm=Arms.LEFT,
                ),
                ParkArmsAction(Arms.BOTH),
                SetGripperAction(Arms.BOTH, motion=GripperState.CLOSE),
            ],
            context=context,
        )

        return plan


def main(
    execution_type: ExecutionType = ExecutionType.SIMULATED, repetitions: int = 1
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    :param repetitions: How often to transport the cereal there and back again.
    """
    # StretchApartmentDemonstration(used_robot=PR2, execution_type=execution_type).run()
    # StretchApartmentDemonstration(used_robot=HSRB, execution_type=execution_type).run()
    StretchApartmentDemonstration(
        used_robot=Stretch, execution_type=execution_type, repetitions=repetitions
    ).run()


if __name__ == "__main__":
    main(execution_type=ExecutionType.REAL, repetitions=LIVE_DEMONSTRATION_REPETITIONS)
