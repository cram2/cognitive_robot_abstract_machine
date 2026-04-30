import os
import time
import multiprocessing
import threading
import rclpy

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.motion_executor import simulated_robot
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.testing import setup_world

from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from semantic_digital_twin.spatial_types.spatial_types import Pose

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive


def hide_robot(world_ref: World, prefix: str):
    """
    Hides a specific robot within the simulated physics world by renaming its associated
    bodies and connections using unique IDs to prevent name collisions in the underlying math solvers.

    param world_ref: The reference to the simulated physics world containing the entities.
    param prefix: The string prefix identifying the robot to be hidden (e.g., "hsrb" or "pr2")
    """
    for b in world_ref.bodies:
        if prefix in str(b.name):
            b.name = PrefixedName(name=f"hidden_b_{id(b)}")
    for c in world_ref.connections:
        if prefix in str(c.name):
            c.name = PrefixedName(name=f"hidden_c_{id(c)}")


def run_robot_process(active_robot: str):
    """
    Initializes an isolated environment to set up the simulation world and execute a
    plan for a designated robot.

    param active_robot: The string identifier of the robot that should execute its
                            plan in this specific process.
    """
    # Setup World (Each process gets its own clean physics world)
    world = setup_world()

    hsrb_urdf = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "robots", "hsrb.urdf")
    hsrb_world = URDFParser.from_file(hsrb_urdf).parse()

    with world.modify_world():
        for c in world.connections:
            if c.parent == world.root and "pr2" in str(c.child.name) and "base_footprint" in str(c.child.name):
                c.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 1.65, 0.0)
                break

        hsrb_root = hsrb_world.get_body_by_name("base_footprint")
        hsrb_drive = OmniDrive.create_with_dofs(parent=world.root, child=hsrb_root, world=world)
        world.merge_world(hsrb_world, hsrb_drive)
        hsrb_drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 4.0, 0)

        missing_joints = ["torso_lift_joint", "hand_l_proximal_joint", "hand_r_proximal_joint", "hand_motor_joint"]
        for c in world.connections:
            if getattr(c.name, 'prefix', '') == 'hsrb' and getattr(c.name, 'name', '') in missing_joints:
                c.has_hardware_interface = True

        world.update_forward_kinematics()
        WorldReasoner(world).reason()
        world.add_semantic_annotations([
            Bowl(root=world.get_body_by_name("milk.stl")),
            Spoon(root=world.get_body_by_name("breakfast_cereal.stl")),
        ])

    # Setup ROS Visualization
    try:
        rclpy.init()
        viz_node = rclpy.create_node(f"viz_marker_{active_robot}")
        v = VizMarkerPublisher(_world=world, node=viz_node)
        v.with_tf_publisher()

        threading.Thread(target=rclpy.spin, args=(viz_node,), daemon=True).start()
    except ImportError:
        pass

    # Spawn Robots & Execute Plans
    if active_robot == "pr2":
        hide_robot(world, "hsrb")
        pr2 = PR2.from_world(world)

        pr2_plan = sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                TransportAction(
                    world.get_body_by_name("breakfast_cereal.stl"),
                    Pose.from_xyz_rpy(5.0, 3.3, 0.8, reference_frame=world.root),
                    Arms.LEFT,
                ),
            ],
            context=Context(world, pr2, None)
        ).plan

        with simulated_robot:
            print(f"--- PR2 is starting its turn (Process ID: {os.getpid()}) ---")
            pr2_plan.perform()

    elif active_robot == "hsrb":
        hide_robot(world, "pr2")
        hsrb = HSRB.from_world(world)

        hsrb_plan = sequential(
            [
                ParkArmsAction(Arms.BOTH),
                NavigateAction(Pose.from_xyz_rpy(1.65, 2.4, 0.0, reference_frame=world.root)),
                MoveTorsoAction(TorsoState.HIGH),
                TransportAction(
                    world.get_body_by_name("milk.stl"),
                    Pose.from_xyz_rpy(2.4, 4.0, 1.0, reference_frame=world.root),
                    Arms.LEFT
                )
            ],
            context=Context(world, hsrb, None)
        ).plan

        with simulated_robot:
            hsrb_plan.perform()

    time.sleep(5)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    print("--- Starting Demo ---")

    p_pr2 = multiprocessing.Process(target=run_robot_process, args=("pr2",))
    p_hsrb = multiprocessing.Process(target=run_robot_process, args=("hsrb",))

    p_pr2.start()
    p_hsrb.start()

    p_pr2.join()
    p_hsrb.join()

    print("--- Demo Finished! ---")