import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    ExecutionType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from panda_assets import PandaMeshAssets
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoBody
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import Pose

STACK_HEIGHT_OFFSET = 0.05
"""
Vertical offset (in meters) above a target cube's center at which a placed
cube should end up.

A cube is 0.04 tall, so stacking one centred on another needs exactly that
much centre-to-centre. The remaining 10mm is release clearance, and has to
stay above the arm's own vertical positioning error: any less and the arm
drives the carried cube into the one below and pushes against a contact it
cannot resolve instead of letting go above it.
"""

rclpy.init()
node = rclpy.create_node("panda_stacking_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()

SCENE = Path(__file__).parent / "stacking_scene.xml"

# The Panda meshes are tens of megabytes, so they are fetched from
# mujoco_menagerie on first run instead of being kept in the repository.
PandaMeshAssets(scene=SCENE).download_if_missing()

world = MJCFParser(str(SCENE)).parse()
Panda.from_world(world)
VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

# It is important to have the ros_node in the context for a real robot
context = Context(
    world=world,
    robot=world.get_semantic_annotations_by_type(Panda)[0],
    ros_node=node,
    evaluate_conditions=False,
    update_world_model_attachment=False,
)

cube0 = world.get_body_by_name("cube0")
cube1 = world.get_body_by_name("cube1")
cube2 = world.get_body_by_name("cube2")
cube3 = world.get_body_by_name("cube3")

arm = context.robot.get_arms()[0]
gripper = arm.end_effector
physically_simulated_dofs = {c.raw_dof for c in gripper.active_connections} | {
    c.raw_dof for c in arm.active_connections
}

# The arm's actuator gains assume gravity is cancelled separately rather than
# held against by the gains alone; without this each joint settles with enough
# gravity sag to never register as converged.
for connection in arm.active_connections:
    connection.child.simulator_additional_properties.append(
        MujocoBody(gravitation_compensation_factor=1.0)
    )

multi_sim = MujocoSim(
    world=world,
    headless=False,
    step_size=0.0001,
    real_time_factor=1,
    physically_simulated_dofs=physically_simulated_dofs,
    sync_rate_hz=100,
    mirror_attachments=False,
)


# Pace motion execution against MuJoCo's own clock: this simulation runs at
# roughly 0.7x real time, so wall-clock pacing would let the controller
# command the arm faster than the simulated arm can actually move.
context.simulation_clock = lambda: multi_sim.simulator.current_simulation_time

full_plan = sequential(
    [
        ParkArmsAction(Arms.RIGHT),
        PickUpAction(
            cube1,
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                gripper,
            ),
        ),
        PlaceAction(
            cube1,
            Pose.from_xyz_rpy(
                x=cube0.global_pose.x,
                y=cube0.global_pose.y,
                z=cube0.global_pose.z + STACK_HEIGHT_OFFSET,
                reference_frame=world.root,
            ),
            Arms.RIGHT,
        ),
        ParkArmsAction(Arms.RIGHT),
        PickUpAction(
            cube2,
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                gripper,
            ),
        ),
        PlaceAction(
            cube2,
            Pose.from_xyz_rpy(
                x=cube3.global_pose.x,
                y=cube3.global_pose.y,
                z=cube3.global_pose.z + STACK_HEIGHT_OFFSET,
                reference_frame=world.root,
            ),
            Arms.RIGHT,
        ),
        ParkArmsAction(Arms.RIGHT),
    ],
    context=context,
)

multi_sim.start_simulation()

with ExecutionEnvironment(
    execution_type=ExecutionType.SIMULATED,
    collision_avoidance=True,
    real_time_pacing=True,
    max_ticks_per_motion_mapping=1000,
):
    full_plan.perform()

while multi_sim.simulator.renderer.is_running():
    time.sleep(0.1)
