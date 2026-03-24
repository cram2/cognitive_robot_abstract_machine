import os
import threading

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.designators.location_designator import CostmapLocation
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import (
    ParkArmsActionDescription,
    NavigateActionDescription,
    PickUpActionDescription,
    MoveTorsoActionDescription,
    PlaceActionDescription,
    TransportActionDescription,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    DiffDrive,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

import pycram.alternative_motion_mappings.stretch_motion_mapping  # type: ignore

world = World()

with world.modify_world():
    root = Body(name=PrefixedName("root"))
    world.add_body(root)

    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "resources",
        "robots",
        "stretch_description.urdf",
    )

    stretch_world = URDFParser.from_file(urdf_dir).parse()
    diff_drive = DiffDrive.create_with_dofs(world, root, stretch_world.root)
    world.merge_world(stretch_world, diff_drive)

    box1_geometry = ShapeCollection([Box(scale=Scale(1, 1, 1))])
    box1 = Body(
        name=PrefixedName("box1"), collision=box1_geometry, visual=box1_geometry
    )
    world.add_connection(
        FixedConnection(
            root,
            box1,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                1.5, 0, 0.5
            ),
        )
    )

    box2_geometry = ShapeCollection([Box(scale=Scale(1, 1, 1))])
    box2 = Body(
        name=PrefixedName("box2"), collision=box2_geometry, visual=box2_geometry
    )
    world.add_connection(
        FixedConnection(
            root,
            box2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -1.5, 0, 0.5
            ),
        )
    )

    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()

    world.merge_world(
        milk_world,
    )

    world.add_semantic_annotation(Milk(root=world.get_body_by_name("milk.stl")))
world.get_body_by_name("milk.stl").parent_connection.origin = (
    HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 0, 1.05)
)

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("stretch_demo")

    exec_filter = SingleThreadedExecutor()
    exec_filter.add_node(node)
    th = threading.Thread(target=exec_filter.spin, daemon=True)
    th.start()

    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError as e:
    node = None


context = Context(world, Stretch.from_world(world), ros_node=node)

place_pose = PoseStamped.from_list([-1.2, 0, 1.05], [0, 0, 1, 0], frame=root)

plan = SequentialPlan(
    context,
    MoveTorsoActionDescription(TorsoState.HIGH),
    NavigateActionDescription(
        CostmapLocation(
            target=world.get_body_by_name("milk.stl"),
            reachable_for=context.robot,
            reachable_arm=Arms.LEFT,
        )
    ),
    PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            context.robot.arm.manipulator,
        ),
    ),
    NavigateActionDescription(
        CostmapLocation(
            target=place_pose,
            reachable_for=context.robot,
            reachable_arm=Arms.LEFT,
        )
    ),
    PlaceActionDescription(world.get_body_by_name("milk.stl"), place_pose, Arms.LEFT),
    ParkArmsActionDescription(Arms.LEFT),
    TransportActionDescription(
        world.get_body_by_name("milk.stl"),
        PoseStamped.from_list([1.2, 0, 1.05], frame=root),
        Arms.LEFT,
    ),
)

with simulated_robot:
    plan.perform()
