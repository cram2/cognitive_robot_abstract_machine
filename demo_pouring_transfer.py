import threading
import time
from dataclasses import dataclass
from pathlib import Path

import math
import rclpy
from rclpy.executors import SingleThreadedExecutor

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.feature_functions import HeightGoal
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
)
from giskardpy.qp.constraint import LargeNumber
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.joint_state import JointState
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from importlib.resources import files

# ------ Constants ----
_JEROEN_CUP_STL = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)
_JEROEN_CUP_SCALE = Scale(1, 1, 1)
_TABLE_SURFACE_Z = 0.9
IS_SIM = False

# START_FILL = 0.8
# GOAL_FILL_CONST = 0.7
# START_YAW = 0.8
START_FILL = 1.0
GOAL_FILL_CONST = 0.7
START_YAW = 0.1


def _spawn_jeroen_cup_body(name: str) -> Body:
    """Create a Body with the Jeroen cup mesh geometry."""
    mesh = Mesh(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
        filename=_JEROEN_CUP_STL,
        scale=_JEROEN_CUP_SCALE,
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([mesh]), name=PrefixedName(name)
    )


# --------------------------
rclpy.init()

# World with real robot:
rclpy_node = rclpy.create_node("tracy_transfer_demo")

executor = SingleThreadedExecutor()
executor.add_node(rclpy_node)
print(f"Executor started")
thread = threading.Thread(
    target=executor.spin,
    daemon=True,
    name="rclpy-executor",
)
thread.start()


giskard = GiskardWrapper(rclpy_node)

world = giskard.world
print(f"world root: {world.root.name.name}")

# ----- Park arms -----
_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]
_LEFT_ARM_PARK_POSITIONS = [2.62, -1.035, 1.13, -0.966, -0.88, 2.07]

_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]
_RIGHT_ARM_PARK_POSITIONS = [3.72, -2.07, -1.17, 4.0, 0.82, 0.75]

park_state = JointState.from_mapping(
    mapping={
        world.get_connection_by_name(name): value
        for name, value in zip(
            _LEFT_ARM_JOINT_NAMES + _RIGHT_ARM_JOINT_NAMES,
            _LEFT_ARM_PARK_POSITIONS + _RIGHT_ARM_PARK_POSITIONS,
        )
    }
)
msc_park = MotionStatechart()
park_task = JointPositionList(goal_state=park_state)
msc_park.add_node(park_task)
msc_park.add_node(EndMotion.when_true(park_task))
giskard.execute(msc_park)

# ------ Move the left gripper to the upright carry pose ----
left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")

root_T_upright_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
    pos_x=1,
    pos_y=0.3,
    pos_z=_TABLE_SURFACE_Z + 0.15,
    quat_z=0.5,
    quat_x=0.5,
    quat_y=0.5,
    quat_w=0.5,
    reference_frame=world.root,
)
upright_pose_T_rotated = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=START_YAW)
upright_pose = (root_T_upright_pose @ upright_pose_T_rotated).to_pose()

msc_cartesian = MotionStatechart()
cartesian_task = CartesianPose(
    root_link=world.root,
    tip_link=left_tool_frame,
    goal_pose=upright_pose,
)
msc_cartesian.add_node(cartesian_task)
msc_cartesian.add_node(min_reached := LocalMinimumReached())
msc_cartesian.add_node(EndMotion.when_true(min_reached))

giskard.execute(msc_cartesian)

# ----- Set up the cups on the first run, reuse them on subsequent runs -----
# Re-running the demo against an already-running Giskard process fetches a world that may still
# contain the cups from a previous run. Re-adding them would fail on duplicate bodies, so on reuse
# the cups are kept and only their fill levels are reset. This avoids restarting Giskard each time.
cups_already_present = bool(world.get_semantic_annotations_by_name("source_cup"))

if cups_already_present:
    print("Cups already present; reusing them and resetting fill levels.")
    source_cup = world.get_semantic_annotation_by_name("source_cup")
    receiving_cup = world.get_semantic_annotation_by_name("receiving_cup")
    # The fetched annotations reference detached copies of their fill connections; re-resolve the
    # world-resident ones so their fill state can be read and written.
    for cup in (source_cup, receiving_cup):
        cup.fill_connection = world.get_connection(
            cup.fill_connection.parent, cup.fill_connection.child
        )
    with world.modify_world():
        world.set_positions_1DOF_connection(
            {source_cup.fill_connection: START_FILL, receiving_cup.fill_connection: 0.0}
        )
    # Give the reset fill state time to propagate to the Giskard process before the transfer goal.
    time.sleep(0.5)
else:
    # ----- Attach the source cup to the gripper -----
    old_number_bodies = len(world.bodies)

    source_cup_body = _spawn_jeroen_cup_body("source_cup")
    with world.modify_world():
        world.add_body(source_cup_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=left_tool_frame,
                child=source_cup_body,
                name=PrefixedName("l_gripper_T_source_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0, y=-0.0
                ),
            )
        )

    assert len(world.bodies) > old_number_bodies

    source_cup = HasFillLevel(name=PrefixedName("source_cup"), root=source_cup_body)
    with world.modify_world():
        world.add_semantic_annotation(source_cup)
    source_cup.initialize_fill_level(
        world=world, initial_fill=START_FILL, outflow_rate_constant=0.8
    )

    # ----- Place the receiving cup on the table -----
    receiving_cup_body = _spawn_jeroen_cup_body("receiving_cup")
    with world.modify_world():
        world.add_body(receiving_cup_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world,
                world.root,
                receiving_cup_body,
                name=PrefixedName("table_T_receiving_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 0.1, _TABLE_SURFACE_Z
                ),
            )
        )
    receiving_cup = HasFillLevel(
        name=PrefixedName("receiving_cup"), root=receiving_cup_body
    )
    with world.modify_world():
        world.add_semantic_annotation(receiving_cup)
    receiving_cup.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )

    # ----- Couple the receiver's inflow to the source's gated outflow -----
    receiving_cup.receive_outflow_from(source=source_cup, world=world)

time.sleep(0.2)

assert source_cup.fill_level == START_FILL
assert receiving_cup.fill_level == 0.0

# ----- Transfer -----
goal_fill = GOAL_FILL_CONST
tolerance = 0.05

transfer_task = FillByTransferTask(
    receiver=receiving_cup,
    goal_value=goal_fill,
    fill_level_tolerance=tolerance,
    reference_velocity=0.03,
)
# Keep the liquid's projectile landing in the receiver so the optimizer repositions the gripper
# upstream as the source tilts and the arc reaches forward.
no_spill = KeepProjectileInReceiver(receiver=receiving_cup, source=source_cup)
# Keep the source cup above the receiver so the optimizer never lowers it into the receiver.
minimum_clearance = 0.2
keep_above = HeightGoal(
    root_link=world.root,
    tip_link=source_cup.root,
    tip_point=Point3(reference_frame=source_cup.root),
    reference_point=Point3(reference_frame=receiving_cup.root),
    lower_limit=minimum_clearance,
    upper_limit=minimum_clearance + 0.02,
    weight=DefaultWeights.WEIGHT_ABOVE_CA,
)
keep_plane = AlignPlanes(
    root_link=world.root,
    tip_link=left_tool_frame,
    goal_normal=Vector3.X(reference_frame=world.root),
    tip_normal=Vector3.Z(reference_frame=left_tool_frame),
)
motion = Parallel([transfer_task, no_spill, keep_above, keep_plane])
msc_transfer = MotionStatechart()
msc_transfer.add_node(motion)
msc_transfer.add_node(EndMotion.when_true(motion))

print("Start transfer.")

giskard.execute(msc_transfer)

# ``giskard.execute`` returns on the action result, but the final fill levels are synchronized
# back from the Giskard process asynchronously. Wait until the receiver's fill reflects the
# completed transfer and settles, so the printed values are the transferred amounts rather than
# a stale pre-transfer belief.
settle_deadline = time.time() + 10.0
previous_fill = None
while time.time() < settle_deadline:
    current_fill = receiving_cup.fill_level
    transfer_reflected = current_fill > tolerance
    settled = previous_fill is not None and abs(current_fill - previous_fill) < 1e-4
    if transfer_reflected and settled:
        break
    previous_fill = current_fill
    time.sleep(0.1)

print(f"receiving cup fill level: {receiving_cup.fill_level}")
print(f"source cup fill level: {source_cup.fill_level}")


msc_cartesian = MotionStatechart()
cartesian_task = CartesianPose(
    root_link=world.root,
    tip_link=left_tool_frame,
    goal_pose=upright_pose,
)
msc_cartesian.add_node(cartesian_task)
msc_cartesian.add_node(min_reached := LocalMinimumReached())
msc_cartesian.add_node(EndMotion.when_true(min_reached))

giskard.execute(msc_cartesian)
