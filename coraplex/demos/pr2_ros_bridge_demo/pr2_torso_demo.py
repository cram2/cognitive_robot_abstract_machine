#!/usr/bin/env python3
"""
PR2 Torso Up/Down — PyCRAM Bridge Demo
═══════════════════════════════════════════════════════════════

Demonstrates PyCRAM-style robot control of the PR2 via the
ROS 1 ↔ ROS 2 bridge.

This demo uses the EXACT same publish pattern that works in
demo_pr2_roundtrip.py — loop-publish for 3 seconds at 10 Hz.

Run inside the sim_bridge container:
  docker compose exec sim_bridge bash -c \\
    "source /opt/ros/foxy/setup.bash && \\
     python3 /workspace/cognitive_robot_abstract_machine/coraplex/demos/pr2_torso_demo.py"
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration


# ══════════════════════════════════════════════════════════════
#  PyCRAM Designators  (WHAT — intent)
# ══════════════════════════════════════════════════════════════

class TorsoDesignator:
    """
    CRAM-style designator: describes the INTENT to move the torso.
    Does not contain any ROS code — purely symbolic.
    """
    def __init__(self, height: float, duration: float = 3.0):
        if not (0.0 <= height <= 0.33):
            raise ValueError(f"Torso height {height:.3f} out of safe range [0.0, 0.33] m")
        self.height = height
        self.duration = duration

    def __repr__(self):
        return f"TorsoDesignator(height={self.height:.3f} m, duration={self.duration:.1f} s)"


# ══════════════════════════════════════════════════════════════
#  ROS 2 Node  (HOW — exactly mirrors demo_pr2_roundtrip.py)
# ══════════════════════════════════════════════════════════════

class PR2BridgeNode(Node):
    """
    Minimal ROS 2 node that:
      - subscribes to /joint_states  (read from robot)
      - publishes to /torso_controller/command  (write to robot)

    Uses the identical publish pattern as demo_pr2_roundtrip.py
    (loop for 3 s at 10 Hz, update stamp each iteration).
    """

    PUBLISH_DURATION = 3.0   # seconds — same as round-trip demo
    PUBLISH_RATE     = 10    # Hz      — same as round-trip demo

    def __init__(self):
        super().__init__('pr2_pycram_torso_demo')

        self._joint_states = {}

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self._torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/command', 10)

        self.get_logger().info('PR2BridgeNode ready.')

    def _js_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._joint_states[name] = pos

    def get_joint(self, name: str):
        return self._joint_states.get(name)

    def wait_for_joint_states(self, timeout: float = 10.0) -> bool:
        self.get_logger().info('Waiting for /joint_states …')
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._joint_states:
                self.get_logger().info(
                    f'Got {len(self._joint_states)} joints. '
                    f'Torso at {self.get_joint("torso_lift_joint"):.4f} m'
                )
                return True
        self.get_logger().error('Timed out waiting for /joint_states')
        return False

    # ── The one action: ground a TorsoDesignator to ROS 2 messages ──

    def execute(self, designator: TorsoDesignator):
        """
        Ground a TorsoDesignator to a JointTrajectory and publish it.
        Identical publish loop to demo_pr2_roundtrip.py's send_torso_command().
        """
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.frame_id = 'base_link'
        msg.joint_names = ['torso_lift_joint']

        point = JointTrajectoryPoint()
        point.positions = [designator.height]
        point.velocities = [0.0]
        point.accelerations = [0.0]
        point.time_from_start = Duration(
            sec=int(designator.duration),
            nanosec=int((designator.duration % 1) * 1e9)
        )
        msg.points = [point]

        print(f'\n  [EXECUTING] {designator}')
        print(f'  Publishing to /torso_controller/command for {self.PUBLISH_DURATION}s …')
        print(f'  (stamp=0 → start immediately, works with Gazebo simulated time)')

        start = time.time()
        count = 0
        while time.time() - start < self.PUBLISH_DURATION:
            # Use stamp=0 so the controller starts the trajectory immediately.
            # Wall-clock time (~1.7 billion seconds) would appear "already expired"
            # to the Gazebo controller which uses simulated time starting near 0.
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            self._torso_pub.publish(msg)
            count += 1
            rclpy.spin_once(self, timeout_sec=1.0 / self.PUBLISH_RATE)

        print(f'  Sent {count} messages. Waiting for movement to complete …')

        # Wait for the movement to physically complete
        deadline = time.time() + designator.duration
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)

        final = self.get_joint('torso_lift_joint')
        print(f'  Done. Torso now at: {final:.4f} m')


# ══════════════════════════════════════════════════════════════
#  PyCRAM Plan  (WHAT sequence to execute)
# ══════════════════════════════════════════════════════════════

def torso_up_down_plan(node: PR2BridgeNode):
    """
    PyCRAM plan: decide direction from current position, then move.

      current < 0.15 m  →  raise to 0.30 m
      current >= 0.15 m →  lower to 0.05 m
    """
    current = node.get_joint('torso_lift_joint')

    print('\n' + '═' * 60)
    print('  PR2 Torso Demo — PyCRAM Bridge Style')
    print('═' * 60)
    print(f'\n  Current torso position : {current:.4f} m')

    if current < 0.15:
        target = 0.30
        direction = 'UP ↑'
    else:
        target = 0.05
        direction = 'DOWN ↓'

    designator = TorsoDesignator(height=target, duration=3.0)

    print(f'  Decision : torso is {"below" if current < 0.15 else "at/above"} 0.15 m')
    print(f'  Action   : move {direction} → {designator}')

    confirm = input('\n  Execute on the robot? [y/N]: ').strip().lower()
    if confirm != 'y':
        print('\n  Aborted — no commands sent.')
        return

    # Ground designator → ROS 2 → bridge → robot
    node.execute(designator)

    print('\n' + '═' * 60)
    print('  Plan complete!')
    print(f'  ✔  {designator}')
    print('═' * 60 + '\n')


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = PR2BridgeNode()

    try:
        if not node.wait_for_joint_states(timeout=10.0):
            print('\n  ERROR: No /joint_states received.')
            print('  Make sure pr2_sim and sim_bridge are running.')
            return

        torso_up_down_plan(node)

    except KeyboardInterrupt:
        print('\n  Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
