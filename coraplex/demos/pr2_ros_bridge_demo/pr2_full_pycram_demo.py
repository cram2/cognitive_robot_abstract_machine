#!/usr/bin/env python3
"""
PR2 Full PyCRAM Demo — Real PyCRAM API via Bridge
══════════════════════════════════════════════════════════════════

Demonstrates the FULL PyCRAM execution chain controlling the PR2 in
Gazebo simulation (or real robot) via the ROS 1/2 bridge.

This is REAL PyCRAM code:
  - Uses PyCRAM Action designators  (MoveTorsoAction, ParkArmsAction)
  - Uses PyCRAM plans               (sequential, execute_single)
  - Uses PyCRAM execution context   (bridge_robot)
  - Routes through AlternativeMotion → PR2ROS1TrajectoryTask → bridge

Full execution chain:
    MoveTorsoAction(TorsoState.HIGH).perform()
        ↓
    MoveJointsMotion(['torso_lift_joint'], [0.30])
        ↓
    AlternativeMotion → PR2MoveJointsMotion (BRIDGE, robot=PR2)
        ↓
    PR2ROS1TrajectoryTask.build() → on_start() → on_tick()
        ↓
    ROS 2 JointTrajectory → dynamic_bridge → PR2 torso controller

══════════════════════════════════════════════════════════════════
  HOW TO RUN
══════════════════════════════════════════════════════════════════

  Terminal 1 — start simulation + bridge:
    docker compose up --no-deps pr2_sim sim_bridge
    (wait ~30s, open http://localhost:6080)

  Terminal 2 — set ROS_DOMAIN_ID to match sim_bridge container, then run:
    export ROS_DOMAIN_ID=5
    source /opt/ros/foxy/setup.bash
    python3 cognitive_robot_abstract_machine/coraplex/demos/pr2_full_pycram_demo.py

  ⚠ IMPORTANT — DDS connectivity:
    The host machine needs to reach the sim_bridge container's ROS 2 topics.
    If topics are not visible (ros2 topic list shows nothing), the bridge
    container must expose host networking:
      SIM_DOMAIN=5 docker compose up --no-deps pr2_sim sim_bridge
    Or restart sim_bridge with network_mode:host (see docker-compose note).
"""

import os
import sys
import logging

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('pr2_full_pycram_demo')

# ── ROS 2 init (must happen before any rclpy imports) ────────────────────────
import rclpy
rclpy.init()

# ── PyCRAM / SDT imports ─────────────────────────────────────────────────────
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.world_description.connections import OmniDrive

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.motion_executor import bridge_robot
from coraplex.plans.factories import sequential, execute_single
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.datastructures.enums import Arms

# ── Register PR2 bridge alternatives (must import AFTER coraplex) ──────────────
# This import registers PR2MoveJointsMotion as an AlternativeMotion subclass.
from coraplex.alternative_motion_mappings.pr2_motion_mapping import (
    PR2MoveJointsMotion,
    PR2LookingMotion,
)


# ══════════════════════════════════════════════════════════════════════════════
#  World setup — load PR2 URDF into SDT world
# ══════════════════════════════════════════════════════════════════════════════

def setup_pr2_world():
    """
    Load the PR2 URDF into a minimal SDT World.

    Tries multiple URDF paths:
      1. Local resources/robots/pr2.urdf in the monorepo
      2. package://iai_pr2_description/... (ROS 1 env)
    """
    # Path 1: local URDF inside the monorepo resources
    local_urdf = os.path.join(
        os.path.dirname(__file__),
        '..', 'resources', 'robots', 'pr2.urdf'
    )
    local_urdf = os.path.realpath(local_urdf)

    if os.path.isfile(local_urdf):
        logger.info(f'Loading PR2 URDF from: {local_urdf}')
        pr2_world = URDFParser.from_file(local_urdf).parse()
    else:
        logger.info('Local URDF not found, trying package://iai_pr2_description/...')
        pr2_world = URDFParser.from_file(
            'package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro'
        ).parse()

    logger.info(f'PR2 world loaded: {len(list(pr2_world.connections))} connections')
    return pr2_world


# ══════════════════════════════════════════════════════════════════════════════
#  Plans — REAL PyCRAM plans using standard Actions
# ══════════════════════════════════════════════════════════════════════════════

def plan_torso_up_down(context: Context):
    """
    PyCRAM plan: move the torso HIGH then LOW.

    Uses the standard PyCRAM action API — MoveTorsoAction internally creates
    MoveJointsMotion which is redirected to PR2MoveJointsMotion via
    AlternativeMotion when bridge_robot context is active.
    """
    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            MoveTorsoAction(TorsoState.LOW),
        ],
        context=context,
    ).plan

    print('\n' + '═' * 62)
    print('  Plan: Torso HIGH → LOW  (real PyCRAM → bridge → PR2)')
    print('═' * 62)
    confirm = input('  Execute? [y/N]: ').strip().lower()
    if confirm != 'y':
        print('  Skipped.')
        return

    with bridge_robot:
        plan.perform()

    print('  ✔ Torso plan complete!')


def plan_move_joints_direct(context: Context):
    """
    PyCRAM plan: directly use MoveJointsMotion (lower-level API).

    Shows that the AlternativeMotion intercept works at the motion level too.
    """
    plan = execute_single(
        MoveJointsMotion(
            names=['torso_lift_joint'],
            positions=[0.20],
        ),
        context=context,
    ).plan

    print('\n' + '═' * 62)
    print('  Plan: MoveJointsMotion(torso → 0.20m)  (motion-level API)')
    print('═' * 62)
    confirm = input('  Execute? [y/N]: ').strip().lower()
    if confirm != 'y':
        print('  Skipped.')
        return

    with bridge_robot:
        plan.perform()

    print('  ✔ MoveJointsMotion plan complete!')


# ══════════════════════════════════════════════════════════════════════════════
#  Verification — confirm AlternativeMotion dispatch works
# ══════════════════════════════════════════════════════════════════════════════

def verify_dispatch(pr2: PR2):
    """
    Verify that MoveJointsMotion correctly dispatches to PR2MoveJointsMotion
    when bridge_robot context is active.
    """
    from coraplex.motion_executor import MotionExecutor
    from coraplex.alternative_motion_mapping import AlternativeMotion

    print('\n' + '─' * 62)
    print('  Verifying AlternativeMotion dispatch ...')

    # Temporarily set BRIDGE execution type
    prev = MotionExecutor.execution_type
    MotionExecutor.execution_type = ExecutionType.BRIDGE

    alternative = AlternativeMotion.check_for_alternative(pr2, MoveJointsMotion)

    MotionExecutor.execution_type = prev

    if alternative is PR2MoveJointsMotion:
        print('  ✔ Dispatch OK: MoveJointsMotion → PR2MoveJointsMotion')
    else:
        print(f'  ✗ Dispatch FAILED: got {alternative}')
        print('    Make sure PR2MoveJointsMotion is imported before this check.')
        return False

    # Check the task type
    m = PR2MoveJointsMotion(names=['torso_lift_joint'], positions=[0.30])
    task = m._motion_chart
    if isinstance(task, __import__(
        'coraplex.alternative_motion_mappings.pr2_motion_mapping',
        fromlist=['PR2ROS1TrajectoryTask']
    ).PR2ROS1TrajectoryTask):
        print('  ✔ Task type OK: PR2ROS1TrajectoryTask')
    print('─' * 62)
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

PLANS = {
    '1': ('Torso HIGH → LOW  (MoveTorsoAction)', plan_torso_up_down),
    '2': ('MoveJointsMotion direct  (motion-level API)', plan_move_joints_direct),
}


def main():
    print('\n' + '═' * 62)
    print('  PR2 Full PyCRAM Demo — Loading robot model...')
    print('═' * 62)

    # Load PR2 world and create context
    world = setup_pr2_world()
    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2)
    logger.info(f'PR2 loaded: {pr2.name}')

    # Verify dispatch
    if not verify_dispatch(pr2):
        print('\n  Aborting — dispatch verification failed.')
        return

    # Menu
    print('\n' + '═' * 62)
    print('  Select a plan:')
    print('═' * 62)
    for key, (desc, _) in PLANS.items():
        print(f'  [{key}] {desc}')
    print('  [q] Quit')

    choice = input('\n  Choose: ').strip().lower()

    if choice == 'q':
        print('  Bye.')
    elif choice in PLANS:
        _, plan_fn = PLANS[choice]
        try:
            plan_fn(context)
        except KeyboardInterrupt:
            print('\n  Interrupted.')
        except Exception as e:
            logger.exception(f'Plan failed: {e}')
    else:
        print('  Invalid choice.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
