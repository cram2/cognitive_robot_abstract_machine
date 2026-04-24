import os

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription

from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from pycram.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
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
bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1, reference_frame=world.root
        ),
    )
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

pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl")),
        ]
    )
    handle = Handle(root=world.get_body_by_name("handle_cab10_t"))
    world.add_semantic_annotations(
        [
            handle,
            Drawer(
                root=world.get_body_by_name("cabinet10_drawer_top"),
                handle=handle,
            ),
        ]
    )

context.evaluate_conditions = False

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
            world.get_body_by_name("bowl.stl"),
            Pose.from_xyz_rpy(5, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
        TransportAction(
            world.get_body_by_name("spoon.stl"),
            Pose.from_xyz_rpy(5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.TOP, pr2.left_arm.manipulator
            ),
        ),
    ],
    context=context,
).plan

with simulated_robot:
    plan.perform()
