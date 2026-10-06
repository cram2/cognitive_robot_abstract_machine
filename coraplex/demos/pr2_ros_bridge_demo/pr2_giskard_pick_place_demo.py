#!/usr/bin/env python3
"""
pr2_giskard_pick_place_demo.py
══════════════════════════════════════════════════════════════════════════════
PR2 Pick-and-Place Demo (Giskard collision-aware planning)

Complete sequence:
  1. Park both arms + Torso up + Open right gripper
  2. Move robot base closer to the object
  3. Approach object (ReachAction with Giskard)
  4. Close gripper
  5. Lift & move object to place location (left of original)
  6. Open gripper (release)
  7. Move base back
  8. Close gripper + Park arms

Execution modes:
  [1] Simulation  -- watch in RViz2 (no robot needed)
  [2] Gazebo sim  -- via docker sim_bridge container
  [3] Real robot  -- via docker bridge container

HOW TO RUN:
    Terminal 1: source /opt/ros/jazzy/setup.bash && rviz2
    Terminal 2: source /opt/ros/jazzy/setup.bash && workon cram-env && python <this file>

    In RViz2 add:
        TF           (Fixed Frame = printed by demo)
        MarkerArray  topic=/semworld/viz_marker  Durability=TRANSIENT_LOCAL
"""

import os, threading, time, logging, math
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hardware Calibration ---
# The physical PR2 might have a 90-degree offset on its wrist roll compared to the URDF.
# 1.57 rad = 90 degrees. Change to -1.57 if it rotates the wrong way.
HARDWARE_WRIST_ROLL_OFFSET = 1.57

# ---- ROS 2 node (needed for TF + MarkerArray) --------------------------------
import rclpy
from rclpy.executors import MultiThreadedExecutor
if not rclpy.ok():
    rclpy.init()
_ros_node = rclpy.create_node('pr2_pick_place_demo')
_executor = MultiThreadedExecutor()
_executor.add_node(_ros_node)
threading.Thread(target=_executor.spin, daemon=True).start()
print(f'[demo] ROS 2 node ready  (ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID","0")})')

# ---- CoraPlex + Giskard imports ----------------------------------------------
from coraplex.testing import setup_world
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription

from coraplex.plans.factories import sequential, execute_single
from coraplex.motion_executor import simulated_robot
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction, MoveManipulatorAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.robot_plans.actions.core.pick_up import ReachAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState, GripperState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix, Pose
import coraplex.alternative_motion_mappings.pr2_motion_mapping  # noqa — registers PR2 handlers

print('\n══════════════════════════════════════════════════════════')
print(' PR2 Pick-and-Place Demo (Giskard Collision-Aware)')
print('══════════════════════════════════════════════════════════\n')

# ---- World setup -------------------------------------------------------------
print('[demo] Loading apartment world...')
world  = setup_world()
pr2    = PR2.from_world(world)
cereal = world.get_body_by_name('breakfast_cereal.stl')
with world.modify_world():
    world.add_semantic_annotation(Cereal(root=cereal))
context = Context.from_world(world)
print('[demo] World ready.\n')

# ---- RViz2 publishers --------------------------------------------------------
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
_viz = VizMarkerPublisher(_world=world, node=_ros_node)
_viz.with_tf_publisher()
print('[demo] TF + MarkerArray publishers active.')

# ---- Print world info --------------------------------------------------------
root_frame = str(world.root.name) if world.root else 'unknown'
cereal_pos = cereal.global_pose.to_position()
cereal_xyz = [round(float(v), 3) for v in cereal_pos[:3]]
print(f'\n[demo] ================================================')
print(f'[demo]  RViz2 Fixed Frame  ->  "{root_frame}"')
print(f'[demo]  MarkerArray topic  ->  /semworld/viz_marker')
print(f'[demo]  Durability         ->  TRANSIENT_LOCAL')
print(f'[demo] ================================================')
print(f'[demo]  Cereal at: {cereal_xyz}')
print(f'[demo]  PR2 base:  [1.5, 2.5, 0.0]')
print(f'[demo] ================================================\n')

# ---- Grasp planning ----------------------------------------------------------
manipulator = ViewManager.get_end_effector_view(Arms.RIGHT, pr2)
grasp_desc  = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, manipulator)
pre_pose, grasp_pose, lift_pose = grasp_desc.grasp_pose_sequence(cereal)

LIFT_HEIGHT = 0.10  # metres to raise before sweeping to place location

# Place: 0.3 m to the left (y +) of original cereal position
place_pose = Pose.from_xyz_quaternion(
    cereal_xyz[0], cereal_xyz[1] + 0.3, cereal_xyz[2] ,
    quat_x=grasp_pose.orientation.x,
    quat_y=grasp_pose.orientation.y,
    quat_z=grasp_pose.orientation.z,
    quat_w=grasp_pose.orientation.w,
    reference_frame=world.root
)

# Intermediate lift pose: straight up from grasp, before sweeping to place
# We use lift_pose from the grasp sequence (already computed above)
# and also define a pose directly above the place location for a clean descent.
place_approach_pose = Pose.from_xyz_quaternion(
    cereal_xyz[0], cereal_xyz[1] + 0.3, cereal_xyz[2] + 0.1 + LIFT_HEIGHT,
    quat_x=grasp_pose.orientation.x,
    quat_y=grasp_pose.orientation.y,
    quat_z=grasp_pose.orientation.z,
    quat_w=grasp_pose.orientation.w,
    reference_frame=world.root
)

print(f'[demo] Pick  location: {cereal_xyz}')
print(f'[demo] Place location: [{cereal_xyz[0]:.2f}, {cereal_xyz[1]+0.3:.2f}, {cereal_xyz[2]:.2f}]')
print(f'[demo] Lift height:    {LIFT_HEIGHT} m\n')

# ============================================================
#  Mode selection
# ============================================================
print('[demo] ================================================')
print('[demo]  Choose execution mode:')
print('[demo]  [1] Simulation  -- watch in RViz2 (no robot needed)')
print('[demo]  [2] Gazebo sim  -- via docker sim_bridge container')
print('[demo]  [3] Real robot  -- via docker bridge container')
print('[demo] ================================================')
mode = input('[demo] Enter 1, 2 or 3: ').strip()

# ============================================================
# HELPERS (used by all modes)
# ============================================================
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MoveJointsAction(ActionDescription):
    """Wrapper so MoveJointsMotion triggers execute_msc() via the plan tree."""
    names: List[str]
    positions: List[float]

    def execute(self):
        self.add_subplan(
            execute_single(MoveJointsMotion(self.names, self.positions))
        ).perform()


def sim_action(description):
    """Execute an ActionDescription (ParkArms, MoveTorso, Navigate, ReachAction)."""
    with simulated_robot:
        execute_single(description, context=context).plan.perform()


def sim_gripper(gripper_state, arm, gap=None):
    """Execute a gripper motion. If gap (metres) is provided, set exact width."""
    with simulated_robot:
        if gap is not None:
            sim_action(MoveJointsAction(
                names=[f'{arm.name.lower()[0]}_gripper_l_finger_joint',
                       f'{arm.name.lower()[0]}_gripper_r_finger_joint'],
                positions=[gap/2, gap/2]
            ))
        else:
            execute_single(MoveGripperMotion(gripper_state, arm), context=context).plan.perform()


def sim_tcp(pose, arm, **kwargs):
    """Execute a TCP motion via ActionDescription so Giskard plans it."""
    with simulated_robot:
        execute_single(
            MoveManipulatorAction(
                target_pose=pose,
                end_effector=ViewManager.get_end_effector_view(arm, pr2),
                allow_gripper_collision=kwargs.get('allow_gripper_collision', False)
            ),
            context=context
        ).plan.perform()


def _nav_target_towards_object(desired_dist=0.8):
    """Compute a base navigation target that brings the PR2 within desired_dist of the cereal."""
    c_pos = cereal.global_pose.to_position()
    b_pos = pr2.root.global_pose.to_position()
    dx = float(c_pos[0]) - float(b_pos[0])
    dy = float(c_pos[1]) - float(b_pos[1])
    dist = math.hypot(dx, dy)
    if dist > desired_dist + 0.05:
        move  = dist - desired_dist
        nx    = float(b_pos[0]) + dx / dist * move
        ny    = float(b_pos[1]) + dy / dist * move
        return (dx / dist * move, dy / dist * move), Pose.from_xyz_rpy(nx, ny, 0.0, 0, 0, 0, reference_frame=world.root)
    return None, None  # already close enough


# ============================================================
#  SIMULATION MODE
# ============================================================
if mode == '1':
    input('\n[demo] Configure RViz2 then press Enter to start...\n')
    try:
        # Step 1 ── Park + Torso + Open gripper
        print('\n[demo] == Step 1: Park Arms + Torso Up + Open Gripper ==')
        sim_action(ParkArmsAction(Arms.BOTH))
        time.sleep(0.5)
        sim_action(MoveTorsoAction(TorsoState.HIGH))
        time.sleep(0.5)
        sim_gripper(GripperState.OPEN, Arms.RIGHT)
        time.sleep(1)

        # Step 2 ── Move base closer
        print('\n[demo] == Step 2: Move Base Closer to Object ==')
        delta, nav_target = _nav_target_towards_object(desired_dist=0.8)
        if nav_target:
            print(f'[demo] Moving base {[round(float(v),3) for v in delta]} m towards object')
            with world.modify_world():
                pr2.root.parent_connection.origin = nav_target.to_homogeneous_matrix()
        else:
            print('[demo] Base already close enough')
        time.sleep(1)

        # Step 3 ── Approach & reach grasp pose
        print('\n[demo] == Step 3: Approach Object ==')
        sim_action(ReachAction(
            target_pose=grasp_pose, arm=Arms.RIGHT,
            grasp_description=grasp_desc, object_designator=cereal
        ))
        time.sleep(1.5)

        # Step 4 ── Close gripper (grasp)
        print('\n[demo] == Step 4: Close Gripper (Grasp to 4 cm) ==')
        sim_gripper(GripperState.CLOSE, Arms.RIGHT, gap=0.04)
        time.sleep(1)

        # Attach cereal to gripper in simulation
        end_eff = ViewManager.get_end_effector_view(Arms.RIGHT, pr2)
        with world.modify_world():
            world.move_branch_with_fixed_connection(cereal, end_eff.tool_frame)

        # Step 5 ── Lift & carry to place location
        print('\n[demo] == Step 5a: Lift Up ==')
        sim_tcp(lift_pose, Arms.RIGHT, allow_gripper_collision=True)
        time.sleep(1.0)

        print('\n[demo] == Step 5b: Sweep to Above Place Location ==')
        sim_tcp(place_approach_pose, Arms.RIGHT, allow_gripper_collision=True)
        time.sleep(1.0)

        print('\n[demo] == Step 5c: Lower to Place Pose ==')
        sim_tcp(place_pose, Arms.RIGHT, allow_gripper_collision=True)
        time.sleep(1.5)

        # Step 6 ── Open gripper (release)
        print('\n[demo] == Step 6: Open Gripper (Release) ==')
        sim_gripper(GripperState.OPEN, Arms.RIGHT)
        time.sleep(1)

        # Detach cereal
        with world.modify_world():
            world.move_branch_with_fixed_connection(cereal, world.root)

        # Step 6.5 ── Move base straight back along X only
        if delta:
            print('\n[demo] == Step 6.5: Move Base Back (X only) ==')
            b_pos = pr2.root.global_pose.to_position()
            nx = float(b_pos[0]) - delta[0]   # reverse X displacement only
            ny = float(b_pos[1])               # keep Y unchanged — straight back
            back_target = Pose.from_xyz_rpy(nx, ny, 0.0, 0, 0, 0, reference_frame=world.root)
            with world.modify_world():
                pr2.root.parent_connection.origin = back_target.to_homogeneous_matrix()
            time.sleep(1)

        # Step 7 ── Close gripper + Park arms
        print('\n[demo] == Step 7: Close Gripper (4 cm) + Park Arms ==')
        sim_gripper(GripperState.CLOSE, Arms.RIGHT, gap=0.04)
        time.sleep(0.5)
        sim_action(ParkArmsAction(Arms.BOTH))
        time.sleep(1)

        print('\n[demo] Simulation complete — pick-and-place finished in RViz2.')
    except Exception as e:
        logger.exception('[demo] Simulation failed: %s', e)
        raise
    finally:
        input('\n[demo] Press Enter to shut down...')
        rclpy.shutdown()

# ============================================================
#  BRIDGE MODE (Gazebo or Real Robot)
# ============================================================
elif mode in ('2', '3'):
    import subprocess, json
    from coraplex.alternative_motion_mappings.pr2_motion_mapping import (
        PR2ROS1TrajectoryTask, _find_bridge_container
    )

    container = _find_bridge_container()
    if not container:
        print('[demo] No bridge container found!')
        print('[demo] Run: docker compose up --no-deps bridge')
        rclpy.shutdown()
        exit(1)
    print(f'[demo] Using container: {container}\n')

    WAIT_SEC = 3
    ARM_DUR  = 5.0
    TORSO_DUR = 4.0

    R_ARM = ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
             'r_elbow_flex_joint',   'r_forearm_roll_joint',  'r_wrist_flex_joint', 'r_wrist_roll_joint']
    L_ARM = ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint',
             'l_elbow_flex_joint',   'l_forearm_roll_joint',  'l_wrist_flex_joint', 'l_wrist_roll_joint']

    def extract(joints):
        positions = []
        for n in joints:
            pos = float(world.state[world.get_connection_by_name(n).dof.id].position)
            # Apply hardware calibration offset for wrist roll
            if n in ('r_wrist_roll_joint', 'l_wrist_roll_joint'):
                pos += HARDWARE_WRIST_ROLL_OFFSET
            positions.append(pos)
        return positions

    def bridge_send(joints, positions, dur, label='', blocking=True):
        def _send():
            print(f'[bridge] -> {label}: {[round(p,3) for p in positions]}')
            t = PR2ROS1TrajectoryTask(joint_names=joints, positions=positions,
                                       duration_sec=dur, timeout_sec=dur + 15)
            t.build(); t.on_start()
        if blocking:
            _send()
        else:
            th = threading.Thread(target=_send, daemon=True)
            th.start()
            return th

    def bridge_arms_parallel(r_pos, l_pos, dur, label=''):
        th_r = bridge_send(R_ARM, r_pos, dur, f'r_arm {label}', blocking=False)
        th_l = bridge_send(L_ARM, l_pos, dur, f'l_arm {label}', blocking=False)
        th_r.join(); th_l.join()

    def bridge_gripper(side, open_gripper, label='', gap=None):
        state = 'OPEN' if open_gripper else 'CLOSE'
        print(f'[bridge] -> {label}: gripper {state}' + (f' (gap={gap}m)' if gap is not None else ''))
        params_dict = {'side': side, 'open': open_gripper}
        if gap is not None:
            params_dict['gap'] = gap
        params = json.dumps(params_dict)
        cmd = (
            f"source /opt/ros/noetic/setup.bash && "
            f"source /catkin_ws/install/setup.bash && "
            f"python3 /workspace/pr2_publish_gripper.py '{params}'"
        )
        try:
            subprocess.run(['docker', 'exec', container, 'bash', '-c', cmd], timeout=20)
            print(f'[bridge] Gripper {state} done')
        except Exception as _e:
            print(f'[bridge] Gripper {state} failed (continuing): {_e}')

    def bridge_base(dx, dy, speed=0.15, label=''):
        print(f'[bridge] -> {label} (dx={dx:.3f}, dy={dy:.3f})')
        params = json.dumps({'dx': dx, 'dy': dy, 'speed': speed})
        cmd = (
            f"source /opt/ros/noetic/setup.bash && "
            f"source /catkin_ws/install/setup.bash && "
            f"python3 /workspace/base_cmd_vel.py '{params}'"
        )
        timeout = ((dx**2 + dy**2)**0.5) / speed + 5.0
        try:
            subprocess.run(['docker', 'exec', container, 'bash', '-c', cmd], timeout=timeout)
        except subprocess.TimeoutExpired:
            print('[bridge] Base movement timed out (continuing)')
        except Exception as _be:
            print(f'[bridge] Base movement failed (continuing): {_be}')

    # ── Step 1: Park + Torso + Open gripper ──────────────────────────────────
    print('\n[demo] == Step 1: Park Arms + Torso Up + Open Gripper ==')
    sim_action(ParkArmsAction(Arms.BOTH))
    sim_action(MoveTorsoAction(TorsoState.HIGH))
    sim_gripper(GripperState.OPEN, Arms.RIGHT)

    bridge_arms_parallel(extract(R_ARM), extract(L_ARM), ARM_DUR, 'park')
    bridge_send(['torso_lift_joint'], extract(['torso_lift_joint']), TORSO_DUR, 'torso high')
    bridge_gripper('r', True, 'open right gripper')
    time.sleep(WAIT_SEC)

    # ── Step 2: Move base closer ──────────────────────────────────────────────
    print('\n[demo] == Step 2: Move Base Closer to Object ==')
    delta, nav_target = _nav_target_towards_object(desired_dist=0.8)
    if delta:
        dx_move, dy_move = delta
        bridge_base(dx_move, dy_move, speed=0.15, label='move closer to object')
        # Sync simulated world with real base movement
        b_pos = pr2.root.global_pose.to_position()
        with world.modify_world():
            new_x = float(b_pos[0]) + dx_move
            new_y = float(b_pos[1]) + dy_move
            pr2.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                new_x, new_y, 0, reference_frame=world.root
            )
    else:
        print('[demo] Base already close enough')
    time.sleep(WAIT_SEC)

    # ── Step 3: Approach object ───────────────────────────────────────────────
    print('\n[demo] == Step 3: Approach Object ==')
    sim_action(ReachAction(
        target_pose=grasp_pose, arm=Arms.RIGHT,
        grasp_description=grasp_desc, object_designator=cereal
    ))
    bridge_send(R_ARM, extract(R_ARM), ARM_DUR, 'r_arm approach')
    time.sleep(WAIT_SEC)

    # ── Step 4: Close gripper (grasp) ─────────────────────────────────────────
    print('\n[demo] == Step 4: Close Gripper (Grasp to 4 cm) ==')
    sim_gripper(GripperState.CLOSE, Arms.RIGHT, gap=0.05)
    bridge_gripper('r', False, 'close right gripper', gap=0.05)
    time.sleep(WAIT_SEC)

    # Attach cereal to gripper in simulation
    end_eff = ViewManager.get_end_effector_view(Arms.RIGHT, pr2)
    with world.modify_world():
        world.move_branch_with_fixed_connection(cereal, end_eff.tool_frame)

    # ── Step 5: Lift → Sweep → Lower to place ────────────────────────────────
    print('\n[demo] == Step 5a: Lift Up ==')
    sim_tcp(lift_pose, Arms.RIGHT, allow_gripper_collision=True)
    bridge_send(R_ARM, extract(R_ARM), ARM_DUR, 'r_arm lift')
    time.sleep(WAIT_SEC)

    print('\n[demo] == Step 5b: Sweep to Above Place Location ==')
    sim_tcp(place_approach_pose, Arms.RIGHT, allow_gripper_collision=True)
    bridge_send(R_ARM, extract(R_ARM), ARM_DUR, 'r_arm sweep to place')
    time.sleep(WAIT_SEC)

    print('\n[demo] == Step 5c: Lower to Place Pose ==')
    sim_tcp(place_pose, Arms.RIGHT, allow_gripper_collision=True)
    bridge_send(R_ARM, extract(R_ARM), ARM_DUR, 'r_arm lower to place')
    time.sleep(WAIT_SEC)

    # ── Step 6: Open gripper (release) ───────────────────────────────────────
    print('\n[demo] == Step 6: Open Gripper (Release) ==')
    sim_gripper(GripperState.OPEN, Arms.RIGHT)
    bridge_gripper('r', True, 'open right gripper (place)')
    time.sleep(WAIT_SEC)

    # Detach cereal
    with world.modify_world():
        world.move_branch_with_fixed_connection(cereal, world.root)

    # ── Step 6.5: Move base straight back along X only ────────────────────────
    if delta:
        print('\n[demo] == Step 6.5: Move Base Back (X only) ==')
        dx_move, dy_move = delta
        bridge_base(-dx_move, 0.0, speed=0.15, label='move back (X only)')
        b_pos = pr2.root.global_pose.to_position()
        with world.modify_world():
            new_x = float(b_pos[0]) - dx_move
            new_y = float(b_pos[1])            # keep Y unchanged
            pr2.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                new_x, new_y, 0, reference_frame=world.root
            )
        time.sleep(WAIT_SEC)

    # ── Step 7: Close gripper + Park arms ─────────────────────────────────────
    print('\n[demo] == Step 7: Close Gripper (4 cm) + Park Arms ==')
    sim_gripper(GripperState.CLOSE, Arms.RIGHT, gap=0.02)
    sim_action(ParkArmsAction(Arms.BOTH))
    bridge_gripper('r', False, 'close right gripper', gap=0.02)
    bridge_arms_parallel(extract(R_ARM), extract(L_ARM), ARM_DUR, 'park')

    print('\n[demo] Pick-and-Place complete!')
    input('\n[demo] Press Enter to shut down...')
    rclpy.shutdown()

else:
    print('[demo] Invalid choice. Exiting.')
    rclpy.shutdown()
    exit(1)
