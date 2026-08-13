import os

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import simulated_robot, simulated_robot_advanced
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from coraplex.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Spoon,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection

world = setup_world()

spoon = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "spoon.stl"
    )
).parse()


with world.modify_world():
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"),
        child=spoon.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.05, -0.05, 0
        ),
    )
    world.merge_world(spoon, connection)


try:
    import rclpy

    rclpy.init()
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    node = rclpy.create_node("viz_marker")
    v = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

robot = Garmi.from_world(world)
context = Context(world=world, robot=robot, _debug=False, ros_node=node)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()
    world.add_semantic_annotations(
        [

            Spoon(root=world.get_body_by_name("spoon.stl")),
        ]
    )
    world.add_semantic_annotation_recursively(
        Drawer(
            root=world.get_body_by_name("cabinet10_drawer_top"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
        )
    )

context.evaluate_conditions = False
context.teleport_to_navigate_in_simulation = True
plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
        TransportAction(
            world.get_body_by_name("milk.stl"),
            Pose.from_xyz_rpy(4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
        TransportAction(
            world.get_body_by_name("spoon.stl"),
            Pose.from_xyz_rpy(5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                robot.left_arm.end_effector,
            ),
        ),
    ],
    context=context,
).plan

with simulated_robot_advanced:
    plan.perform()
