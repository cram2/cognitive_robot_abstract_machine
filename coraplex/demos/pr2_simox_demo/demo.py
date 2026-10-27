#!/usr/bin/env python3
"""
pr2_simox_demo/demo.py
══════════════════════════════════════════════════════════════════════════════ PR2 Simox
Grasp Planner + CoraPlex + Giskard + ROS 1/2 Bridge Pick-and-Place Demo.

Complete sequence:
  1. Park both arms + Torso up + Open right gripper
  2. Move PR2 base closer to the kitchen counter
  3. Call Simox (/plan_grasp) to generate physics-based force-closure grasps
  4. Approach & grasp breakfast_cereal using the best reachable Simox GraspPose
  5. Lift cereal up, move PR2 a bit to the left (+0.25 m along counter), and lower to place
  6. Open gripper (release cereal), retract arm
  7. Move base back + Close gripper + Park both arms

Execution modes:
  [1] Simulation  -- watch in RViz2 (no robot/docker needed)
  [2] Real robot  -- via docker bridge container (ros1_2_bridge)

HOW TO RUN:
  Terminal 1 (Simox service):
    ros2 run grasp_planner_service grasp_planner_service_node

  Terminal 2 (RViz2):
    source /opt/ros/jazzy/setup.bash && rviz2
    (Add TF with Fixed Frame='world', and MarkerArray on '/semworld/viz_marker' with Durability=TRANSIENT_LOCAL)

  Terminal 3 (Demo):
    source /opt/ros/jazzy/setup.bash
    python3 coraplex/demos/pr2_simox_demo/demo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.executors import MultiThreadedExecutor

try:
    import coraplex.alternative_motion_mappings.pr2_motion_mapping  # noqa: F401 — registers PR2 bridge handlers
except ImportError:
    pass

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, MovementType
from coraplex.execution_environment import simulated_robot
from coraplex.external_interfaces.simox_grasp_planner import (
    plan_grasps_for_body,
    translate_pose_along_local_axis,
)
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
)
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.testing import setup_world
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pr2_simox_demo")

# %% Bowl Tuning Parameters (Easy Control)
BOWL_TILT_DEG = 25.0  # Slight angle in degrees matching apartment_bowl.stl's conical wall (e.g. 25.0°)
BOWL_EXTRA_FORWARD_M = (
    0.065  # Move robot base closer to kitchen counter (increase to move more forward)
)
BOWL_PICK_Z_OFFSET = 0.05  # Height of gripper above bowl center when picking (0.05m = top rim of apartment_bowl.stl)
BOWL_PREGRASP_HEIGHT_M = 0.10  # How far back along the tilted wall angle the gripper starts before moving onto the rim
BOWL_LIFT_HEIGHT_M = 0.14  # How much the arm lifts the bowl up after grasping (14 cm)
BOWL_PLACE_DESCENT_M = (
    0.15  # How much the arm goes down when placing (decrease so arm stays higher)
)

# %% Paths
SIMOX_WORKSPACE = os.environ.get(
    "SIMOX_WORKSPACE_DIR", str(Path.home() / "grasp_planner_service")
)
REPO_ROOT = Path(SIMOX_WORKSPACE)
OBJECTS_DIR = REPO_ROOT / "grasp_test_files" / "resources" / "objects"
ROBOT_XML = str(REPO_ROOT / "grasp_test_files" / "resources" / "robots" / "pr2.xml")


@dataclass
class MoveJointsAction(ActionDescription):
    """
    Wrapper so MoveJointsMotion runs through the plan tree.
    """

    names: List[str]
    positions: List[float]

    @property
    def _action_plan(self):
        return execute_single(MoveJointsMotion(self.names, self.positions))


def main():
    parser = argparse.ArgumentParser(
        description="PR2 Simox Grasp Planner + CoraPlex + Giskard + Bridge Pick-and-Place Demo"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["1", "2"],
        help="Execution mode: 1=RViz2 Simulation, 2=Real PR2 via bridge",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="cereal",
        choices=["cereal", "breakfast_cereal", "milk", "bowl", "apartment_bowl"],
        help="Target object to pick and place (default: 'cereal').",
    )

    parser.add_argument(
        "--approach",
        type=str,
        default="right",
        choices=["front", "right", "left", "top", "back", "any"],
        help="Preferred Simox grasp approach direction (default: 'right').",
    )

    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Arm to use (default: 'right').",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.01,
        help="Minimum Simox grasp quality threshold (default: 0.01).",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=15,
        help="Number of grasp candidates to request from Simox (default: 15).",
    )

    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Run immediately without interactive Enter prompts.",
    )
    args, _ = parser.parse_known_args()

    print("\n══════════════════════════════════════════════════════════════════════")
    print(" PR2 Simox Grasp Planner + CoraPlex + Giskard + ROS 1/2 Bridge Demo")
    print("══════════════════════════════════════════════════════════════════════\n")

    # 1. ROS 2 Node & Executor
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("pr2_simox_pick_place_demo")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    print(
        f'[demo] ROS 2 node ready (ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID", "0")})'
    )

    # 2. World Setup
    print("[demo] Loading apartment world...")
    world = setup_world()
    pr2 = PR2.from_world(world)

    is_milk = args.object == "milk"
    is_bowl = args.object in ("bowl", "apartment_bowl")
    if is_milk:
        target_name = "milk.stl"
    elif is_bowl:
        target_name = "apartment_bowl.stl"
        from semantic_digital_twin.adapters.mesh import STLParser

        coraplex_objects_dir = (
            Path(__file__).resolve().parents[2] / "resources" / "objects"
        )
        bowl_stl_path = coraplex_objects_dir / "apartment_bowl.stl"
        bowl_world = STLParser(str(bowl_stl_path)).parse()
        world.merge_world(bowl_world)
    else:
        target_name = "breakfast_cereal.stl"

    target_obj = world.get_body_by_name(target_name)
    all_objs = ["breakfast_cereal.stl", "milk.stl"] + ([target_name] if is_bowl else [])
    other_objs = [world.get_body_by_name(n) for n in all_objs if n != target_name]

    arm_enum = Arms.RIGHT if args.arm == "right" else Arms.LEFT
    arm_obj = pr2.right_arm if args.arm == "right" else pr2.left_arm
    end_effector = arm_obj.end_effector
    eef_name = "r_gripper" if args.arm == "right" else "l_gripper"
    chain_name = "RightArm" if args.arm == "right" else "LeftArm"
    gripper_side = "r" if args.arm == "right" else "l"

    # Place target object on the kitchen counter
    # Note: apartment_bowl.stl has z_min=0.0m, z_max=0.10m, so pick_z=0.95 places its base on the counter ( rim at z=1.05m )
    pick_x = 2.5
    pick_y = 2.1 if arm_enum == Arms.RIGHT else 2.65
    pick_z = 0.98 if is_bowl else 1.05
    left_offset_y = 0.85  # Move left (+Y) to place

    target_pose = Pose.from_xyz_quaternion(
        pos_x=pick_x,
        pos_y=pick_y,
        pos_z=pick_z,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
        reference_frame=world.root,
    )
    with world.modify_world():
        target_obj.parent_connection.origin = target_pose.homogeneous_matrix
        for idx_o, o_body in enumerate(other_objs):
            aside_pose = Pose.from_xyz_quaternion(
                pos_x=2.37,
                pos_y=1.75 + idx_o * 0.20,
                pos_z=1.05,
                quat_x=0.0,
                quat_y=0.0,
                quat_z=0.0,
                quat_w=1.0,
                reference_frame=world.root,
            )
            o_body.parent_connection.origin = aside_pose.homogeneous_matrix
        if not is_milk and not is_bowl:
            world.add_semantic_annotation(Cereal(root=target_obj))

    context = Context(world=world, robot=pr2, ros_node=node)

    # 3. RViz2 Visualization
    _viz = VizMarkerPublisher(_world=world, node=node)
    root_frame = str(world.root.name) if world.root else "world"
    print("\n[demo] ================================================")
    print(f'[demo]  RViz2 Fixed Frame  ->  "{root_frame}" (or "world")')
    print("[demo]  MarkerArray topic  ->  /semworld/viz_marker")
    print("[demo]  Durability         ->  TRANSIENT_LOCAL")
    print("[demo] ================================================")
    print(f"[demo]  Pick  location: [{pick_x:.2f}, {pick_y:.2f}, {pick_z:.2f}]")
    print(
        f"[demo]  Place location: [{pick_x:.2f}, {pick_y + left_offset_y:.2f}, {pick_z:.2f}] (+{left_offset_y:.2f}m left)"
    )
    print("[demo]  PR2 start base: [1.50, 2.50, 0.00]")
    print("[demo] ================================================\n")

    # 4. Mode Selection
    mode = args.mode
    if mode is None:
        print("[demo] ================================================")
        print("[demo]  Choose execution mode:")
        print("[demo]  [1] Simulation  -- watch in RViz2 (no robot needed)")
        print("[demo]  [2] Real robot  -- via docker bridge container")
        print("[demo] ================================================")
        mode = input("[demo] Enter 1 or 2 [default=1]: ").strip() or "1"

    use_bridge = mode == "2"
    container = None
    if use_bridge:
        try:
            from coraplex.alternative_motion_mappings.pr2_motion_mapping import (
                PR2ROS1TrajectoryTask,
                _find_bridge_container,
            )
        except ImportError:
            print("[demo] ERROR: PR2 bridge module not found.")
            print(
                "[demo] Real robot mode (mode 2) requires 'coraplex.alternative_motion_mappings.pr2_motion_mapping'"
            )
            print("[demo] which is in the PR2 bridge branch (feat/pr2-ros-bridge).")
            print("[demo] Please run in simulation mode (mode 1) on this branch.")
            rclpy.shutdown()
            return

        container = _find_bridge_container()
        if not container:
            print("[demo] ERROR: No bridge container found!")
            print(
                "[demo] Run in the ros1_2_bridge directory: bash start_bridge_to_robot.sh"
            )
            rclpy.shutdown()
            return
        print(f"[demo] Using bridge container: {container}\n")

    # %% Helpers (Simulation + Bridge Sync)

    R_ARM = [
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_elbow_flex_joint",
        "r_forearm_roll_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint",
    ]
    L_ARM = [
        "l_shoulder_pan_joint",
        "l_shoulder_lift_joint",
        "l_upper_arm_roll_joint",
        "l_elbow_flex_joint",
        "l_forearm_roll_joint",
        "l_wrist_flex_joint",
        "l_wrist_roll_joint",
    ]
    ACTIVE_ARM_JOINTS = R_ARM if arm_enum == Arms.RIGHT else L_ARM
    ARM_DUR = 5.0
    TORSO_DUR = 4.0
    WAIT_SEC = 2.0 if use_bridge else 0.5

    def extract_joints(joints: List[str]) -> List[float]:
        return [
            float(world.state[world.get_connection_by_name(n).dof.id].position)
            for n in joints
        ]

    def bridge_send(
        joints: List[str],
        positions: List[float],
        dur: float,
        label: str = "",
        blocking: bool = True,
    ):
        if not use_bridge:
            return None

        def _send():
            print(
                f"[bridge] -> {label}: {[round(p, 3) for p in positions]}", flush=True
            )
            t = PR2ROS1TrajectoryTask(
                joint_names=joints,
                positions=positions,
                duration_sec=dur,
                timeout_sec=dur + 15.0,
            )
            t.build()
            t.on_start()

        if blocking:
            _send()
            return None
        th = threading.Thread(target=_send, daemon=True)
        th.start()
        return th

    def bridge_arms_parallel(
        r_pos: List[float], l_pos: List[float], dur: float, label: str = ""
    ):
        if not use_bridge:
            return
        th_r = bridge_send(R_ARM, r_pos, dur, f"r_arm {label}", blocking=False)
        th_l = bridge_send(L_ARM, l_pos, dur, f"l_arm {label}", blocking=False)
        if th_r:
            th_r.join()
        if th_l:
            th_l.join()

    def bridge_gripper(
        side: str, open_gripper: bool, label: str = "", gap: Optional[float] = None
    ):
        if not use_bridge:
            return
        state = "OPEN" if open_gripper else "CLOSE"
        print(
            f"[bridge] -> {label}: gripper {state}"
            + (f" (gap={gap}m)" if gap is not None else ""),
            flush=True,
        )
        params_dict = {"side": side, "open": open_gripper}
        if gap is not None:
            params_dict["gap"] = gap
        params = json.dumps(params_dict)
        cmd = (
            f"source /opt/ros/noetic/setup.bash && "
            "([ -f /catkin_ws/install/setup.bash ] && source /catkin_ws/install/setup.bash || true) && "
            f"python3 /workspace/pr2_publish_gripper.py '{params}'"
        )
        try:
            subprocess.run(["docker", "exec", container, "bash", "-c", cmd], timeout=20)
        except Exception as exc:
            print(f"[bridge] Gripper {state} warning: {exc}", flush=True)

    def bridge_base(dx: float, dy: float, speed: float = 0.15, label: str = ""):
        if not use_bridge:
            return
        print(f"[bridge] -> {label} (dx={dx:.3f}, dy={dy:.3f})", flush=True)
        params = json.dumps({"dx": dx, "dy": dy, "speed": speed})
        cmd = (
            f"source /opt/ros/noetic/setup.bash && "
            "([ -f /catkin_ws/install/setup.bash ] && source /catkin_ws/install/setup.bash || true) && "
            f"python3 /workspace/base_cmd_vel.py '{params}'"
        )
        timeout = ((dx**2 + dy**2) ** 0.5) / speed + 5.0
        try:
            subprocess.run(
                ["docker", "exec", container, "bash", "-c", cmd], timeout=timeout
            )
        except Exception as exc:
            print(f"[bridge] Base movement warning: {exc}", flush=True)

    def sim_action(description: ActionDescription):
        with simulated_robot:
            execute_single(description, context=context).plan.perform()

    def sim_gripper(
        gripper_state: GripperState, arm: Any = None, gap: Optional[float] = None
    ):
        with simulated_robot:
            if gap is not None:
                prefix = gripper_side
                sim_action(
                    MoveJointsAction(
                        names=[
                            f"{prefix}_gripper_l_finger_joint",
                            f"{prefix}_gripper_r_finger_joint",
                        ],
                        positions=[gap / 2.0, gap / 2.0],
                    )
                )
            else:
                execute_single(
                    MoveGripperMotion(gripper_state, end_effector), context=context
                ).plan.perform()

    def move_robot_base_by(dx: float, dy: float, label: str):
        """
        Move the PR2 base by (dx, dy) in world frame using the ros1_2_bridge
        base_cmd_vel controller when in bridge mode, and update the digital twin world
        state (falling back to Giskard NavigateAction if needed).
        """
        if use_bridge:
            bridge_base(dx, dy, speed=0.15, label=label)

        b_pos = pr2.root.global_pose.position
        nx = float(b_pos[0]) + dx
        ny = float(b_pos[1]) + dy
        target_nav_pose = Pose.from_xyz_rpy(
            nx, ny, 0.0, 0.0, 0.0, 0.0, reference_frame=world.root
        )
        try:
            with world.modify_world():
                pr2.root.parent_connection.origin = (
                    HomogeneousTransformationMatrix.from_xyz_rpy(
                        nx, ny, 0.0, reference_frame=world.root
                    )
                )
        except Exception:
            sim_action(NavigateAction(target_location=target_nav_pose))

    if mode == "1" and not args.no_prompt:
        input("\n[demo] Configure RViz2 then press Enter to start...\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1: Park Both Arms + Raise Torso + Open Gripper
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[demo] ══ Step 1: Park Arms + Raise Torso + Open Gripper ══")
    sim_action(ParkArmsAction([pr2.right_arm, pr2.left_arm]))
    sim_action(MoveTorsoAction(TorsoState.HIGH))
    sim_gripper(GripperState.OPEN, arm_enum)

    bridge_arms_parallel(extract_joints(R_ARM), extract_joints(L_ARM), ARM_DUR, "park")
    bridge_send(
        ["torso_lift_joint"],
        extract_joints(["torso_lift_joint"]),
        TORSO_DUR,
        "torso high",
    )
    bridge_gripper(gripper_side, True, "open gripper")
    time.sleep(WAIT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Move PR2 Base Closer to the Kitchen Counter
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[demo] ══ Step 2: Move Base Closer to Kitchen Counter ══")
    # Move PR2 from x=1.50, y=2.50 closer to the counter (+BOWL_EXTRA_FORWARD_M for bowl)
    b_pos = pr2.root.global_pose.position
    extra_fwd = BOWL_EXTRA_FORWARD_M if is_bowl else 0.0
    nav_dx = (pick_x - 0.65 + extra_fwd) - float(b_pos[0])
    nav_dy = (
        (pick_y + 0.11) - float(b_pos[1])
        if arm_enum == Arms.RIGHT
        else (pick_y - 0.11) - float(b_pos[1])
    )
    print(
        f"[demo] Navigating base closer to counter by dx={nav_dx:.3f}m, dy={nav_dy:.3f}m"
    )
    move_robot_base_by(nav_dx, nav_dy, label="move closer to kitchen counter")
    time.sleep(WAIT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: Call Simox Grasp Planner & Reach Best Grasp Pose via Giskard
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[demo] ══ Step 3: Query Simox /plan_grasp & Reach Object ══")
    grasp_dict = plan_grasps_for_body(
        body=target_obj,
        arm=arm_enum,
        end_effector_name=eef_name,
        kinematic_chain_name=chain_name,
        world=world,
        robot_xml=ROBOT_XML,
        object_dir=OBJECTS_DIR,
        num_grasps_to_plan=args.num_grasps,
        quality_threshold=args.quality_threshold,
        timeout_ms=90000,
    )

    if not grasp_dict:
        raise RuntimeError(
            f"Simox returned no valid grasps for '{target_obj.name}'. "
            "Make sure the Simox service is running (bash scripts/launch_simox_service.sh)."
        )

    if is_bowl:
        preferred = ["top"]
    elif args.approach.lower() == "any":
        preferred = None
    else:
        preferred = [args.approach.lower()]
    if preferred is not None:
        directions = [d for d in preferred if d in grasp_dict]
        if not directions:
            logger.warning(
                "Preferred approach %s not in Simox results %s — trying all available directions",
                preferred,
                list(grasp_dict.keys()),
            )
            directions = sorted(
                grasp_dict.keys(),
                key=lambda d: grasp_dict[d][0].quality if grasp_dict[d] else 0.0,
                reverse=True,
            )
    else:
        directions = sorted(
            grasp_dict.keys(),
            key=lambda d: grasp_dict[d][0].quality if grasp_dict[d] else 0.0,
            reverse=True,
        )

    end_effector = (
        pr2.right_arm.end_effector
        if arm_enum == Arms.RIGHT
        else pr2.left_arm.end_effector
    )
    selected_grasp_pose = None
    _pre_grasp_pose = None

    with simulated_robot:
        for direction in directions:
            for idx, candidate in enumerate(grasp_dict[direction], start=1):
                if is_bowl:
                    # Grasp apartment_bowl.stl on the near rim at a slight angle (BOWL_TILT_DEG)
                    # aligned with the bowl's slanted conical wall (~25°-30° from vertical):
                    # - Near rim radius of apartment_bowl.stl at top (z=0.10m) is 0.0815m.
                    # - Tilt angle `tilt` tilts the gripper fingers (+X) to match the sloped cone wall
                    #   so one finger slides along the inside wall and one along the outside wall.
                    from coraplex.tf_transformations import quaternion_from_matrix
                    import numpy as np

                    rim_radius = 0.0815
                    rim_sign = -1.0 if arm_enum == Arms.RIGHT else 1.0
                    tilt = math.radians(BOWL_TILT_DEG) * (
                        -rim_sign
                    )  # +25° for right arm, -25° for left arm

                    # Position along the slanted rim wall at height pick_z + BOWL_PICK_Z_OFFSET
                    depth_from_top = max(0.0, 0.050 - BOWL_PICK_Z_OFFSET)
                    rim_x = pick_x
                    rim_y = pick_y + rim_sign * (
                        rim_radius - depth_from_top * math.tan(abs(tilt))
                    )
                    rim_z = pick_z + BOWL_PICK_Z_OFFSET

                    T_vert = np.eye(4)
                    # Tool +X: tilted by BOWL_TILT_DEG parallel to the conical bowl wall
                    T_vert[:3, 0] = [0.0, math.sin(tilt), -math.cos(tilt)]
                    # Tool +Y: perpendicular to the conical bowl wall (finger closing axis across rim)
                    T_vert[:3, 1] = [0.0, math.cos(tilt), math.sin(tilt)]
                    # Tool +Z: (+X) x (+Y) = [1, 0, 0] (palm forward, zero wrist twist)
                    T_vert[:3, 2] = [1.0, 0.0, 0.0]
                    q_vert = quaternion_from_matrix(T_vert)

                    candidate = Pose.from_xyz_quaternion(
                        pos_x=rim_x,
                        pos_y=rim_y,
                        pos_z=rim_z,
                        quat_x=float(q_vert[0]),
                        quat_y=float(q_vert[1]),
                        quat_z=float(q_vert[2]),
                        quat_w=float(q_vert[3]),
                        reference_frame=world.root,
                    )
                    candidate.quality = grasp_dict[direction][idx - 1].quality
                    candidate.approach = "top"

                    # Pre-grasp backs off along the tilted cone-wall axis (-Tool +X) by BOWL_PREGRASP_HEIGHT_M
                    # so the open gripper slides straight down along the slanted wall angle onto the rim.
                    candidate_pre = translate_pose_along_local_axis(
                        candidate, [1.0, 0.0, 0.0], -BOWL_PREGRASP_HEIGHT_M
                    )
                else:
                    # 10 cm pre-grasp backoff along the Simox approach axis (-X of CoraPlex tool frame)
                    candidate_pre = translate_pose_along_local_axis(
                        candidate, [1.0, 0.0, 0.0], -0.10
                    )

                print(
                    f"[demo] Trying Simox grasp [{direction}] #{idx} "
                    f"(quality={candidate.quality:.4f}, pos=[{float(candidate.x):.3f}, {float(candidate.y):.3f}, {float(candidate.z):.3f}])"
                )
                try:
                    # 3a. Move to pre-grasp pose (directly above the bowl rim for bowl)
                    execute_single(
                        MoveManipulatorAction(
                            target_pose=candidate_pre,
                            end_effector=end_effector,
                            allow_gripper_collision=False,
                        ),
                        context=context,
                    ).plan.perform()
                    bridge_send(
                        ACTIVE_ARM_JOINTS,
                        extract_joints(ACTIVE_ARM_JOINTS),
                        ARM_DUR,
                        f"{args.arm}_arm pre-grasp ({direction})",
                    )

                    # 3b. Final straight-line Cartesian descent to grasp pose
                    if is_bowl:
                        execute_single(
                            MoveToolCenterPointMotion(
                                target=candidate,
                                arm=arm_obj,
                                allow_gripper_collision=True,
                                movement_type=MovementType.TRANSLATION,
                            ),
                            context=context,
                        ).plan.perform()
                    else:
                        execute_single(
                            MoveManipulatorAction(
                                target_pose=candidate,
                                end_effector=end_effector,
                                allow_gripper_collision=True,
                            ),
                            context=context,
                        ).plan.perform()
                    bridge_send(
                        ACTIVE_ARM_JOINTS,
                        extract_joints(ACTIVE_ARM_JOINTS),
                        ARM_DUR,
                        f"{args.arm}_arm final grasp ({direction})",
                    )

                    selected_grasp_pose = candidate
                    _pre_grasp_pose = candidate_pre
                    print(
                        f"[demo] ✔ Reached Simox grasp [{direction}] #{idx} "
                        f"(quality={candidate.quality:.4f})"
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Candidate [%s] #%d unreachable: %s", direction, idx, exc
                    )
                    continue
            if selected_grasp_pose is not None:
                break

    if selected_grasp_pose is None:
        raise RuntimeError("All Simox grasp candidates failed Giskard reachability.")

    time.sleep(WAIT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: Close Gripper & Attach Object
    # ══════════════════════════════════════════════════════════════════════════
    grasp_gap = 0.0 if is_bowl else 0.04
    print(
        f"\n[demo] ══ Step 4: Close Gripper (gap={grasp_gap:.3f}m, full close={is_bowl}) & Attach {target_obj.name} ══"
    )
    sim_gripper(GripperState.CLOSE, arm_enum, gap=grasp_gap)
    bridge_gripper(gripper_side, False, "close gripper (grasp)", gap=grasp_gap)

    time.sleep(WAIT_SEC)

    with world.modify_world():
        world.move_branch_with_fixed_connection(target_obj, end_effector.tool_frame)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5: Lift Up -> Move Left Along Counter -> Lower to Safe Place Pose
    # ══════════════════════════════════════════════════════════════════════════
    lift_height = BOWL_LIFT_HEIGHT_M if is_bowl else 0.12
    descent_height = BOWL_PLACE_DESCENT_M if is_bowl else 0.10
    place_safety_clearance = lift_height - descent_height
    print(f"\n[demo] ══ Step 5a: Lift {target_obj.name} Up by {lift_height:.2f}m ══")
    current_tcp = end_effector.tool_frame.global_pose
    lift_pose = Pose.from_xyz_quaternion(
        pos_x=float(current_tcp.x),
        pos_y=float(current_tcp.y),
        pos_z=float(current_tcp.z) + lift_height,
        quat_x=float(current_tcp.orientation.x),
        quat_y=float(current_tcp.orientation.y),
        quat_z=float(current_tcp.orientation.z),
        quat_w=float(current_tcp.orientation.w),
        reference_frame=world.root,
    )
    with simulated_robot:
        execute_single(
            MoveToolCenterPointMotion(
                target=lift_pose,
                arm=arm_obj,
                allow_gripper_collision=True,
                movement_type=MovementType.TRANSLATION,
            ),
            context=context,
        ).plan.perform()
    bridge_send(
        ACTIVE_ARM_JOINTS, extract_joints(ACTIVE_ARM_JOINTS), ARM_DUR, "lift object"
    )
    time.sleep(WAIT_SEC)

    print(
        f"\n[demo] ══ Step 5b: Move PR2 Base Left (+{left_offset_y:.2f}m along counter) ══"
    )
    move_robot_base_by(0.0, left_offset_y, label="move left to place location")
    time.sleep(WAIT_SEC)

    print(
        f"\n[demo] ══ Step 5c: Lower {target_obj.name} by {descent_height:.2f}m "
        f"(keeping +{place_safety_clearance:.2f}m counter clearance) ══"
    )
    current_tcp_after_move = end_effector.tool_frame.global_pose
    place_tcp_pose = Pose.from_xyz_quaternion(
        pos_x=float(current_tcp_after_move.x),
        pos_y=float(current_tcp_after_move.y),
        pos_z=float(current_tcp_after_move.z) - descent_height,
        quat_x=float(current_tcp_after_move.orientation.x),
        quat_y=float(current_tcp_after_move.orientation.y),
        quat_z=float(current_tcp_after_move.orientation.z),
        quat_w=float(current_tcp_after_move.orientation.w),
        reference_frame=world.root,
    )

    with simulated_robot:
        execute_single(
            MoveToolCenterPointMotion(
                target=place_tcp_pose,
                arm=arm_obj,
                allow_gripper_collision=True,
                movement_type=MovementType.TRANSLATION,
            ),
            context=context,
        ).plan.perform()
    bridge_send(
        ACTIVE_ARM_JOINTS,
        extract_joints(ACTIVE_ARM_JOINTS),
        ARM_DUR,
        "lower to place pose",
    )
    time.sleep(WAIT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6: Open Gripper (Release) & Retract Arm
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[demo] ══ Step 6: Open Gripper (Release) & Retract Arm ══")
    sim_gripper(GripperState.OPEN, arm_enum)
    bridge_gripper(gripper_side, True, "open gripper (release)")
    time.sleep(WAIT_SEC)

    with world.modify_world():
        world.move_branch_with_fixed_connection(target_obj, world.root)

    # Retract straight up by lift_height before parking
    retract_pose = Pose.from_xyz_quaternion(
        pos_x=float(place_tcp_pose.x),
        pos_y=float(place_tcp_pose.y),
        pos_z=float(place_tcp_pose.z) + lift_height,
        quat_x=float(place_tcp_pose.orientation.x),
        quat_y=float(place_tcp_pose.orientation.y),
        quat_z=float(place_tcp_pose.orientation.z),
        quat_w=float(place_tcp_pose.orientation.w),
        reference_frame=world.root,
    )
    with simulated_robot:
        execute_single(
            MoveToolCenterPointMotion(
                target=retract_pose,
                arm=arm_obj,
                allow_gripper_collision=True,
                movement_type=MovementType.TRANSLATION,
            ),
            context=context,
        ).plan.perform()
    bridge_send(
        ACTIVE_ARM_JOINTS, extract_joints(ACTIVE_ARM_JOINTS), ARM_DUR, "retract arm up"
    )
    time.sleep(WAIT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 7: Move Base Back + Close Gripper + Park Both Arms
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[demo] ══ Step 7: Move Base Back + Park Both Arms ══")
    move_robot_base_by(-nav_dx, 0.0, label="move base back from counter")
    time.sleep(WAIT_SEC)

    sim_gripper(GripperState.CLOSE, arm_enum, gap=0.02)
    sim_action(ParkArmsAction([pr2.right_arm, pr2.left_arm]))
    bridge_gripper(gripper_side, False, "close gripper", gap=0.02)
    bridge_arms_parallel(extract_joints(R_ARM), extract_joints(L_ARM), ARM_DUR, "park")

    print("\n══════════════════════════════════════════════════════════════════════")
    print(
        f" ✔ Demo Complete! {target_obj.name} picked via Simox and placed on the left!"
    )
    print(
        f" Final {target_obj.name} world pose: "
        f"x={float(target_obj.global_pose.x):.3f}, "
        f"y={float(target_obj.global_pose.y):.3f}, "
        f"z={float(target_obj.global_pose.z):.3f}"
    )
    print("══════════════════════════════════════════════════════════════════════\n")

    if not args.no_prompt:
        input("\n[demo] Press Enter to shut down...")
    else:
        time.sleep(3.0)

    try:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
