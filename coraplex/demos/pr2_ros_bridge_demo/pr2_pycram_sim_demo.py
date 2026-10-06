#!/usr/bin/env python3
"""
PR2 PyCRAM Simulation Demo
══════════════════════════════════════════════════════════════════

Demonstrates the full PyCRAM designator pipeline controlling the PR2
in Gazebo simulation through the ROS 1 ↔ ROS 2 bridge.

PyCRAM pattern used:
    with pr2_bridge:                            ← execution context
        PR2MoveTorsoMotion(height=0.30).perform()   ← designator → bridge

This is the simulation equivalent of the full monorepo PyCRAM pattern:
    with real_robot:
        MoveJointsMotion(names=['torso_lift_joint'], positions=[0.30]).perform()

══════════════════════════════════════════════════════════════════
  HOW TO RUN (simulation)
══════════════════════════════════════════════════════════════════

  Terminal 1 — start simulation + bridge:
    docker compose up --no-deps pr2_sim sim_bridge

  Wait ~30s for Gazebo to boot, then open http://localhost:6080

  Terminal 2 — run this PyCRAM demo:
    docker compose exec sim_bridge bash -c \\
      "source /opt/ros/foxy/setup.bash && \\
       python3 /workspace/cognitive_robot_abstract_machine/coraplex/demos/pr2_pycram_sim_demo.py"

══════════════════════════════════════════════════════════════════
"""

import sys
import os

# Make pycram_pr2_bridge importable
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), '..', '..',
        'pycram_pr2_bridge', 'src'
    )
)

import rclpy

from pr2_motion_mapping import (
    pr2_bridge,
    PR2MoveTorsoMotion,
    PR2MoveHeadMotion,
    PR2MoveJointsMotion,
    get_pr2_interface,
)


# ══════════════════════════════════════════════════════════════
#  PyCRAM Plans  (the WHAT — sequences of designators)
# ══════════════════════════════════════════════════════════════

def torso_up_down_plan():
    """
    PyCRAM plan: move the torso up and then down using the designator API.

    This is the canonical PyCRAM pattern:
        with <execution_context>:
            <Designator>.perform()
    """
    print('\n' + '═' * 62)
    print('  PR2 PyCRAM Simulation Demo')
    print('  Designator → Bridge → Gazebo')
    print('═' * 62)

    # Enter the bridge context FIRST — this waits for /joint_states to arrive
    with pr2_bridge:
        interface = get_pr2_interface()
        current = interface.get_joint_position('torso_lift_joint')

        print(f'\n  Current torso : {current:.4f} m')

        if current < 0.15:
            target = 0.30
            direction = 'UP ↑'
        else:
            target = 0.05
            direction = 'DOWN ↓'

        print(f'  Decision      : move {direction} → {target:.2f} m')

        confirm = input('\n  Execute? [y/N]: ').strip().lower()
        if confirm != 'y':
            print('  Aborted — no commands sent.')
            return

        # ── PyCRAM execution ─────────────────────────────────────
        #
        #   Equivalent to the full monorepo call:
        #       with real_robot:
        #           MoveJointsMotion(
        #               names=['torso_lift_joint'],
        #               positions=[target]
        #           ).perform()
        #
        print(f'\n  [PyCRAM] PR2MoveTorsoMotion(height={target}).perform()')
        PR2MoveTorsoMotion(height=target, duration=3.0).perform()

        final = interface.get_joint_position('torso_lift_joint')
        print(f'\n  Final torso   : {final:.4f} m')

    print('\n  ✔ Plan complete!')


def multi_joint_plan():
    """
    PyCRAM plan: demonstrate PR2MoveJointsMotion with arbitrary joints.

    Shows how to move any combination of joints using the generic designator.
    """
    print('\n' + '═' * 62)
    print('  PR2 Multi-Joint PyCRAM Demo')
    print('═' * 62)

    with pr2_bridge:
        # Move torso using the generic interface
        print('\n  [PyCRAM] PR2MoveJointsMotion(torso_lift_joint → 0.20).perform()')
        PR2MoveJointsMotion(
            names=['torso_lift_joint'],
            positions=[0.20],
            duration=3.0,
        ).perform()

        # Move head
        print('\n  [PyCRAM] PR2MoveHeadMotion(pan=0.5, tilt=-0.2).perform()')
        PR2MoveHeadMotion(pan=0.5, tilt=-0.2, duration=2.0).perform()

        # Return head to forward
        print('\n  [PyCRAM] PR2MoveHeadMotion(pan=0.0, tilt=0.0).perform()')
        PR2MoveHeadMotion(pan=0.0, tilt=0.0, duration=2.0).perform()

    print('\n  ✔ Multi-joint plan complete!')


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

PLANS = {
    '1': ('Torso up/down demo   (PR2MoveTorsoMotion)', torso_up_down_plan),
    '2': ('Multi-joint demo     (PR2MoveJointsMotion + PR2MoveHeadMotion)', multi_joint_plan),
}


def main():
    rclpy.init()

    print('\n' + '═' * 62)
    print('  PR2 PyCRAM Bridge Demo — Select a plan:')
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
            plan_fn()
        except KeyboardInterrupt:
            print('\n  Interrupted.')
        except Exception as e:
            print(f'\n  ERROR: {e}')
            raise
    else:
        print('  Invalid choice.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
