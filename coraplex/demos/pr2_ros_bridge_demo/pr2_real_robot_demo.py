#!/usr/bin/env python3
"""
PR2 Real Robot Hardware & State Demo ====================================

Demonstrates reading PR2 state and commanding motions via CoraPlex and standard
ROS interfaces.

Phases:
  1. State Inspection: Reads PR2 joint states via /joint_states and logs a summary
     grouped by robot kinematic chains (torso, arms, head, grippers).
  2. Safe Torso Command: Executes a safe torso motion using CoraPlex standard
     MoveJointsMotion, respecting joint limits derived dynamically from the
     PR2 semantic digital twin model.

Execution Modes:
  --mode sim  : Executes against local simulated digital twin (default).
  --mode real : Connects to live ROS joint states and delegates execution
                to Giskard.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import (
    ExecutionEnvironment,
    real_robot,
    simulated_robot,
)
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pr2_real_robot_demo")


class PR2StateMonitor(Node):
    """
    ROS 2 node that subscribes to /joint_states to monitor PR2 state.
    """

    def __init__(self, node_name: str = "pr2_state_monitor") -> None:
        super().__init__(node_name)
        self._joint_states: Dict[str, float] = {}
        self._msg_count = 0
        self._subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        self._msg_count += 1
        for name, pos in zip(msg.name, msg.position):
            self._joint_states[name] = pos

    def get_joint(self, name: str) -> Optional[float]:
        return self._joint_states.get(name)

    def wait_for_joint_states(self, timeout_sec: float = 10.0) -> bool:
        """
        Spin until at least one /joint_states message is received.
        """
        logger.info("Waiting for /joint_states from the PR2...")
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._joint_states:
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.1)
                return True
        return False

    def log_joint_summary(self) -> None:
        """
        Log all received PR2 joints grouped by body component.
        """
        if not self._joint_states:
            logger.warning("No joint state data received yet.")
            return

        groups: Dict[str, List[Tuple[str, float]]] = {
            "HEAD": [],
            "LEFT ARM": [],
            "RIGHT ARM": [],
            "TORSO": [],
            "GRIPPERS": [],
        }
        arm_keywords = ("shoulder", "upper_arm", "forearm", "elbow", "wrist")

        for name, pos in sorted(self._joint_states.items()):
            if "head" in name:
                groups["HEAD"].append((name, pos))
            elif name.startswith("l_") and any(k in name for k in arm_keywords):
                groups["LEFT ARM"].append((name, pos))
            elif name.startswith("r_") and any(k in name for k in arm_keywords):
                groups["RIGHT ARM"].append((name, pos))
            elif "torso" in name:
                groups["TORSO"].append((name, pos))
            elif "gripper" in name:
                groups["GRIPPERS"].append((name, pos))

        logger.info("══════════════════════════════════════════════════════════")
        logger.info("  PR2 Joint States Summary (/joint_states)")
        logger.info("══════════════════════════════════════════════════════════")
        for section, joints in groups.items():
            if joints:
                logger.info("[%s]", section)
                for jname, pos in joints:
                    deg = pos * 57.2958
                    logger.info("  %-35s : %+0.4f rad (%+0.1f deg)", jname, pos, deg)
        logger.info(
            "Total joints tracked: %d (messages: %d)",
            len(self._joint_states),
            self._msg_count,
        )
        logger.info("══════════════════════════════════════════════════════════")


def get_safe_torso_limits(pr2: PR2) -> Tuple[str, float, float]:
    """
    Extract torso joint name and safe limits dynamically from the PR2 model.
    """
    torso_conn = pr2.torso.active_connections[0]
    torso_name = torso_conn.name.name
    lower_limit = float(torso_conn.dof.limits.lower.position)
    upper_limit = float(torso_conn.dof.limits.upper.position)
    return torso_name, lower_limit, upper_limit


def execute_torso_motion(
    context: Context,
    pr2: PR2,
    target_height: float,
    execution_env: ExecutionEnvironment,
) -> None:
    """
    Execute a standard CoraPlex MoveJointsMotion for the torso.
    """
    torso_name, lower_limit, upper_limit = get_safe_torso_limits(pr2)
    if not (lower_limit <= target_height <= upper_limit):
        raise ValueError(
            f"Target height {target_height:.4f}m outside safe limits [{lower_limit:.4f}, {upper_limit:.4f}]m"
        )

    logger.info("Creating MoveJointsMotion for %s -> %.4f m", torso_name, target_height)
    motion = MoveJointsMotion(names=[torso_name], positions=[target_height])
    plan = execute_single(motion, context=context).plan

    logger.info(
        "Executing plan with execution environment: %s",
        execution_env.execution_type.name,
    )
    with execution_env:
        plan.perform()
    logger.info("Torso motion completed successfully.")


def run_demo(mode: str, auto_confirm: bool = False) -> None:
    """
    Run the PR2 real robot / simulation state and torso demo.
    """
    world: World = setup_world()
    pr2 = PR2.from_world(world)
    context = Context.from_world(world)
    torso_name, lower_limit, upper_limit = get_safe_torso_limits(pr2)

    logger.info(
        "PR2 model loaded. Torso joint '%s' limits: [%.4f, %.4f] m",
        torso_name,
        lower_limit,
        upper_limit,
    )

    if mode == "real":
        rclpy.init()
        monitor = PR2StateMonitor()
        try:
            logger.info("Phase 1: Reading joint states...")
            if not monitor.wait_for_joint_states(timeout_sec=10.0):
                logger.error(
                    "No /joint_states received within 10s. Verify bridge and robot network."
                )
                return

            monitor.log_joint_summary()

            current_pos = monitor.get_joint(torso_name)
            if current_pos is None:
                current_pos = lower_limit

            target = current_pos + 0.05
            if target > upper_limit - 0.02:
                target = current_pos - 0.05

            logger.info(
                "Current torso: %.4f m, Proposed target: %.4f m", current_pos, target
            )
            if not auto_confirm:
                response = (
                    input("Send torso motion command to the robot? [y/N]: ")
                    .strip()
                    .lower()
                )
                if response != "y":
                    logger.info("Execution skipped by user.")
                    return

            execute_torso_motion(context, pr2, target, real_robot)

            time.sleep(1.0)
            for _ in range(5):
                rclpy.spin_once(monitor, timeout_sec=0.1)
            final_pos = monitor.get_joint(torso_name)
            logger.info(
                "Final torso position: %s m",
                f"{final_pos:.4f}" if final_pos is not None else "unknown",
            )
        finally:
            monitor.destroy_node()
            rclpy.shutdown()
    else:
        logger.info("Phase 1: Simulated environment ready.")
        target = min(0.20, upper_limit)
        logger.info("Phase 2: Executing simulated torso motion to %.4f m...", target)
        execute_torso_motion(context, pr2, target, simulated_robot)
        logger.info("Simulated demo completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PR2 Real Robot Hardware & State Demo")
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="Execution mode: 'sim' for local digital twin simulation, 'real' for live robot execution (default: sim)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically confirm hardware movement commands without prompting",
    )
    args = parser.parse_args()
    run_demo(mode=args.mode, auto_confirm=args.yes)


if __name__ == "__main__":
    main()
