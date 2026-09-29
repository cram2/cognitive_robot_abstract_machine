#!/usr/bin/env python3

import logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from coraplex.testing import setup_world
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential, execute_single
from coraplex.motion_executor import simulated_robot
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.robots.pr2 import PR2

import coraplex.alternative_motion_mappings.pr2_motion_mapping  # noqa — register bridge handlers

print(' PR2 Giskard Both Arms Demo (Plan Locally → Execute via Bridge)')


# ── 1. Setup world ────────────────────────────────────────────────────────────
print('[demo] Loading world...')
world = setup_world()
pr2   = PR2.from_world(world)
context = Context.from_world(world)
base_footprint = world.get_kinematic_structure_entity_by_name("base_footprint")
print('[demo] World ready.\n')

# ── 2. Define arm targets ─────────────────────────────────────────────────────
steps = [
    ("Park Both Arms", ParkArmsAction(Arms.BOTH))
]

# Right arm joint names for PR2
r_arm_joints = [
    'r_shoulder_pan_joint',
    'r_shoulder_lift_joint',
    'r_upper_arm_roll_joint',
    'r_elbow_flex_joint',
    'r_forearm_roll_joint',
    'r_wrist_flex_joint',
    'r_wrist_roll_joint'
]

# Left arm joint names for PR2
l_arm_joints = [
    'l_shoulder_pan_joint',
    'l_shoulder_lift_joint',
    'l_upper_arm_roll_joint',
    'l_elbow_flex_joint',
    'l_forearm_roll_joint',
    'l_wrist_flex_joint',
    'l_wrist_roll_joint'
]

# ── 3. Ask before executing on robot via bridge ──────────────────────────────
print('[demo]   ROBOT EXECUTION MODE')
print('[demo]    The PR2 BOTH arms will move to their parked positions.')
print('[demo]    If the sim_bridge is running, this will move the robot in RViz/Gazebo.')
print('[demo]    If the real bridge is running, this will move the physical robot.')
answer = input('[demo]    Type "yes" to execute, or anything else to skip: ').strip().lower()

if answer == 'yes':
    from coraplex.alternative_motion_mappings.pr2_motion_mapping import PR2ROS1TrajectoryTask, _find_bridge_container
    container = _find_bridge_container()

    if not container:
        print('[demo]  No bridge container found! Aborting.')
    else:
        print(f'\n[demo] Sending planned joints via bridge container ({container})...')

        for i, (name, action_or_motion) in enumerate(steps):
            print(f'\n[demo] ── Step {i+1}: {name} ──')

            # Wrap the action/motion in a plan
            plan = execute_single(action_or_motion, context=context).plan

            # Plan in simulation — giskardpy solves arm IK locally
            print('[demo] Planning with local giskardpy...')
            try:
                with simulated_robot:
                    plan.perform()
                print('[demo]  Planning succeeded!')
            except Exception as e:
                print(f'[demo]  Planning failed: {e}')
                break

            # Extract planned right arm joint positions from world state
            r_positions = []
            for j_name in r_arm_joints:
                conn = world.get_connection_by_name(j_name)
                pos = float(world.state[conn.dof.id].position)
                r_positions.append(pos)

            # Extract planned left arm joint positions from world state
            l_positions = []
            for j_name in l_arm_joints:
                conn = world.get_connection_by_name(j_name)
                pos = float(world.state[conn.dof.id].position)
                l_positions.append(pos)

            print(f'[demo]   Planned joints for right arm step {i+1}:')
            for j_name, pos in zip(r_arm_joints, r_positions):
                print(f'[demo]     {j_name}: {pos:.4f} rad')

            print(f'[demo]   Planned joints for left arm step {i+1}:')
            for j_name, pos in zip(l_arm_joints, l_positions):
                print(f'[demo]     {j_name}: {pos:.4f} rad')

            # Send commands to real robot via bridge (right then left for safety)
            print('\n[demo] Executing right arm trajectory...')
            r_task = PR2ROS1TrajectoryTask(
                joint_names=r_arm_joints,
                positions=r_positions,
                duration_sec=4.0,
                timeout_sec=20.0,
            )
            r_task.build()
            r_task.on_start()

            print('\n[demo] Executing left arm trajectory...')
            l_task = PR2ROS1TrajectoryTask(
                joint_names=l_arm_joints,
                positions=l_positions,
                duration_sec=4.0,
                timeout_sec=20.0,
            )
            l_task.build()
            l_task.on_start()

            if i < len(steps) - 1:
                print('[demo]  Reached target. Waiting 5 seconds before next movement...')
                time.sleep(5)
            else:
                print('[demo]  Sequence finished!')
else:
    print('[demo] Skipped real robot execution.')

print('\n══ Demo complete ══')
