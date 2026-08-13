import os

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot, simulated_robot_advanced
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import OmniDrive

# %% World setup
lab_world = MJCFParser(
    CompositePathResolver().resolve("package://iai_garmi_apartment/mjcf/scene-bodies.xml")
).parse()

# lab_world = URDFParser.from_file("package://iai_kit_mobile_lab/urdf/R007.urdf").parse()
garmi_world = URDFParser.from_file(
    "package://garmi_description/urdf/garmi.urdf"
).parse()
milk_world = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "milk.stl"
    )
).parse()

with lab_world.modify_world():
    root_connection = OmniDrive.create_with_dofs(
        parent=lab_world.root, child=garmi_world.root, world=lab_world
    )
    lab_world.merge_world(garmi_world, root_connection)
    root_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(0, 5, 0)
    lab_world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.8, 6.4, 1.1, yaw=180, reference_frame=lab_world.root
        ),
    )

world = lab_world

# %% Visualization

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    rclpy.init()
    node = rclpy.create_node("viz_marker")
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

# %% Demo
print(world.root)
garmi = Garmi.from_world(world)
context = Context(world=world, robot=garmi, _debug=False, ros_node=node)
context.evaluate_conditions = False
context.teleport_to_navigate_in_simulation = True

plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
        TransportAction(
            world.get_body_by_name("milk.stl"),
            Pose.from_xyz_rpy(0.256, 2.72, 0.3, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
    ],
    context=context,
).plan

with simulated_robot_advanced:
    plan.perform()
