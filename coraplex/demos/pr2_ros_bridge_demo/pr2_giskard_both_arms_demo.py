"""
PR2 Giskard Both Arms Demo
==========================

Demonstrates dual-arm planning and execution with Giskard and CoraPlex
for the PR2 robot.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Tuple

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import (
    ExecutionEnvironment,
    real_robot,
    simulated_robot,
)
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_demo_context() -> Tuple[World, PR2, Context]:
    """
    Load the world and construct the CoraPlex planning context for the PR2.
    """
    logger.info("Setting up world and PR2 model...")
    world = setup_world()
    pr2 = PR2.from_world(world)
    context = Context.from_world(world)
    return world, pr2, context


def extract_arm_joints(pr2: PR2) -> Tuple[List[str], List[str]]:
    """
    Extract right and left arm joint names dynamically from semantic annotations.
    """
    r_arm_joints = [c.name.name for c in pr2.right_arm.active_connections]
    l_arm_joints = [c.name.name for c in pr2.left_arm.active_connections]
    return r_arm_joints, l_arm_joints


def run_both_arms_motion(
    context: Context,
    pr2: PR2,
    execution_env: ExecutionEnvironment = simulated_robot,
) -> None:
    """
    Plan and execute dual-arm parking for the PR2 using Giskard.
    """
    r_joints, l_joints = extract_arm_joints(pr2)
    logger.info("Right arm joints (%d): %s", len(r_joints), r_joints)
    logger.info("Left arm joints (%d): %s", len(l_joints), l_joints)

    action = ParkArmsAction(arms=list(pr2.all_arms))
    logger.info("Creating plan for: ParkArmsAction(arms=pr2.all_arms)...")
    plan = execute_single(action, context=context).plan

    logger.info(
        "Executing plan with execution environment: %s...",
        execution_env.execution_type.name,
    )
    with execution_env:
        plan.perform()
    logger.info("Both arms parking motion completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PR2 Giskard Both Arms Demo")
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="Execution mode: 'sim' for local simulation, 'real' for Giskard hardware execution (default: sim)",
    )
    args = parser.parse_args()

    execution_env = real_robot if args.mode == "real" else simulated_robot
    logger.info("Starting PR2 Both Arms Demo in '%s' mode...", args.mode)

    _, pr2, context = setup_demo_context()
    run_both_arms_motion(context=context, pr2=pr2, execution_env=execution_env)


if __name__ == "__main__":
    main()
