#!/usr/bin/env python3
"""
PR2 Real Robot PyCRAM Demo
══════════════════════════════════════════════════════════════════

PyCRAM-style demo for the real PR2, mirroring demo_pr2_roundtrip.py.

  Phase 1 (READ):   Subscribe to /joint_states, print full joint summary.
                    ROS1 robot → bridge → this ROS2 script

  Phase 2 (WRITE):  Move torso ±0.05 m via a dense JointTrajectory.
                    This ROS2 script → bridge → ROS1 robot

PyCRAM layer:
  TorsoDesignator  — symbolic intent (WHAT to do)
  PR2RealNode       — grounds to ROS 2 dense trajectory (HOW to do it)
  real_robot_plan() — the plan (SEQUENCE of actions)

══════════════════════════════════════════════════════════════════
  HOW TO START
══════════════════════════════════════════════════════════════════

  Terminal 1 — start the bridge:
    ROS_MASTER_URI=http://192.168.102.60:11311 \\
    ROS_IP=192.168.101.108 \\
    ROS_DOMAIN_ID=5 \\
    docker compose up --no-deps bridge

  Terminal 2 — run this demo:
    docker compose exec bridge bash -c \\
      "source /opt/ros/foxy/setup.bash && \\
       python3 /workspace/cognitive_robot_abstract_machine/coraplex/demos/pr2_real_robot_demo.py"

  IMPORTANT:
    - Use Docker Engine (docker-ce), NOT Docker Desktop.
    - Update ROS_MASTER_URI and ROS_IP to match your lab network.
    - The robot will NOT move until you type 'y' and press Enter.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ══════════════════════════════════════════════════════════════
#  PyCRAM Designator  (WHAT — symbolic intent, no ROS code)
# ══════════════════════════════════════════════════════════════

class TorsoDesignator:
    """
    Describes the INTENT to move the torso to a target height.
    Contains no ROS code — purely symbolic.

    The designator is grounded to a dense JointTrajectory by
    PR2RealNode.execute(), matching the real PR2's safety requirements.
    """

    JOINT_LIMITS = (0.0, 0.33)  # metres

    def __init__(self, target_height: float, start_height: float, duration: float = 3.0):
        lo, hi = self.JOINT_LIMITS
        if not (lo <= target_height <= hi):
            raise ValueError(
                f"Target height {target_height:.3f} m outside safe range "
                f"[{lo}, {hi}] m"
            )
        self.target_height = target_height
        self.start_height  = start_height
        self.duration      = duration
        delta = target_height - start_height
        self.direction = "UP ↑" if delta >= 0 else "DOWN ↓"

    def __repr__(self):
        return (
            f"TorsoDesignator("
            f"start={self.start_height:.4f}m → target={self.target_height:.4f}m, "
            f"duration={self.duration:.1f}s, {self.direction})"
        )


# ══════════════════════════════════════════════════════════════
#  ROS 2 Node  (HOW — identical mechanics to demo_pr2_roundtrip)
# ══════════════════════════════════════════════════════════════

class PR2RealNode(Node):
    """
    ROS 2 node that:
      - Subscribes to /joint_states  (read from real PR2)
      - Publishes dense JointTrajectory to /torso_controller/command

    The dense trajectory (10 pts/sec) is required because the real PR2
    controller rejects sparse trajectories for safety.  Same logic as
    demo_pr2_roundtrip.py's send_torso_command().
    """

    POINTS_PER_SECOND = 10  # density of the trajectory — must stay ≥ 10

    def __init__(self):
        super().__init__('pr2_real_pycram_demo')

        self._joint_states: dict = {}
        self._msg_count = 0
        self._raw_msg = None

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_callback, 10)

        self._torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/command', 10)

        self.get_logger().info('PR2RealNode ready.')

    # ── Subscriber ──────────────────────────────────────────

    def _js_callback(self, msg: JointState):
        self._raw_msg = msg
        self._msg_count += 1
        for name, pos in zip(msg.name, msg.position):
            self._joint_states[name] = pos

    def get_joint(self, name: str):
        return self._joint_states.get(name)

    def wait_for_joint_states(self, timeout: float = 10.0) -> bool:
        """Spin until at least one /joint_states message arrives."""
        self.get_logger().info('Waiting for /joint_states from the PR2 …')
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._joint_states:
                # Collect a few more for stability
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.2)
                return True
        return False

    # ── Print summary (mirrors demo_pr2_roundtrip exactly) ──

    def print_joint_summary(self):
        """Print all PR2 joints grouped by body part."""
        if self._raw_msg is None:
            print('  No joint data received.')
            return

        groups = {
            'HEAD':      [],
            'LEFT ARM':  [],
            'RIGHT ARM': [],
            'TORSO':     [],
            'GRIPPERS':  [],
        }
        arm_kw = ('shoulder', 'upper_arm', 'forearm', 'elbow', 'wrist')

        for name, pos in self._joint_states.items():
            if 'head' in name:
                groups['HEAD'].append((name, pos))
            elif name.startswith('l_') and any(k in name for k in arm_kw):
                groups['LEFT ARM'].append((name, pos))
            elif name.startswith('r_') and any(k in name for k in arm_kw):
                groups['RIGHT ARM'].append((name, pos))
            elif 'torso' in name:
                groups['TORSO'].append((name, pos))
            elif 'gripper' in name:
                groups['GRIPPERS'].append((name, pos))

        print('\n' + '=' * 60)
        print('  PR2 Joint States (received via ROS 2 through bridge)')
        print('=' * 60)
        for title, joints in groups.items():
            if joints:
                print(f'\n  {title}:')
                for jname, pos in joints:
                    deg = pos * 57.2958
                    print(f'    {jname:40s}  {pos:+.4f} rad  ({deg:+.1f}°)')
        print(f'\n  Total joints received:  {len(self._joint_states)}')
        print(f'  Messages received:      {self._msg_count}')
        print('=' * 60)

    # ── Executor: ground TorsoDesignator → dense ROS2 message ──

    def execute(self, designator: TorsoDesignator):
        """
        Ground a TorsoDesignator to a dense JointTrajectory and send it.

        Identical dense-trajectory logic to demo_pr2_roundtrip.py:
          - num_points = duration_sec × POINTS_PER_SECOND
          - Linear interpolation from start_height to target_height
          - stamp = 0 (start immediately)
          - Wait for subscriber count > 0 before publishing
          - Publish 3×, then wait for movement to complete
        """
        print(f'\n  [EXECUTING] {designator}')

        n_pts = int(designator.duration * self.POINTS_PER_SECOND)
        start = designator.start_height
        end   = designator.target_height

        msg = JointTrajectory()
        msg.header.stamp.sec    = 0   # start immediately (real-time controller)
        msg.header.stamp.nanosec = 0
        msg.joint_names = ['torso_lift_joint']

        for i in range(1, n_pts + 1):
            frac = i / float(n_pts)
            height_i = start + (end - start) * frac
            t_sec    = (designator.duration / n_pts) * i

            pt = JointTrajectoryPoint()
            pt.positions = [height_i]
            pt.velocities = [0.0]
            pt.time_from_start = Duration(
                sec=int(t_sec),
                nanosec=int((t_sec % 1) * 1e9),
            )
            msg.points.append(pt)

        print(f'  Dense trajectory: {n_pts} points over {designator.duration:.1f}s')

        # Wait for the bridge to subscribe (same pattern as round-trip demo)
        print('  Waiting for ros1_bridge to subscribe to our publisher …')
        wait_start = time.time()
        while self._torso_pub.get_subscription_count() == 0:
            if time.time() - wait_start > 5.0:
                self.get_logger().warn('Bridge did not subscribe within 5s! Publishing anyway.')
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        sub_count = self._torso_pub.get_subscription_count()
        print(f'  Bridge subscribers: {sub_count}  — sending trajectory …')

        # Publish 3× to be safe
        for _ in range(3):
            self._torso_pub.publish(msg)
            time.sleep(0.1)

        # Spin for the movement duration
        print(f'  Waiting {designator.duration:.1f}s for movement to complete …')
        deadline = time.time() + designator.duration + 1.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)

        final = self.get_joint('torso_lift_joint')
        if final is not None:
            print(f'  Final torso position: {final:.4f} m')


# ══════════════════════════════════════════════════════════════
#  PyCRAM Plan  (WHAT sequence to execute)
# ══════════════════════════════════════════════════════════════

def real_robot_plan(node: PR2RealNode):
    """
    PyCRAM plan for the real PR2:

      Phase 1 — Read joint states and print full joint summary.
      Phase 2 — Move torso +0.05 m (or -0.05 m if near the top).
    """

    # ── Phase 1: Read ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  PHASE 1: Reading PR2 joint states via bridge')
    print('  (ROS 1 robot → bridge → this ROS 2 script)')
    print('=' * 60)

    if not node.wait_for_joint_states(timeout=10.0):
        print('\n  ERROR: No /joint_states received after 10s.')
        print('  Is the bridge running and connected to the PR2?')
        print('  Check: ROS_MASTER_URI and ROS_IP are set correctly.')
        return

    node.print_joint_summary()
    print('\n  PHASE 1 PASSED — Successfully reading PR2 data through the bridge!')

    # ── Phase 2: Move ─────────────────────────────────────────────
    current = node.get_joint('torso_lift_joint')
    if current is None:
        print('\n  Could not find torso_lift_joint. Skipping Phase 2.')
        return

    print('\n' + '=' * 60)
    print('  PHASE 2: Send a torso movement command')
    print('  (this ROS 2 script → bridge → ROS 1 robot → PR2 moves)')
    print('=' * 60)

    print(f'\n  Current torso position:')
    print(f'    torso_lift_joint = {current:+.4f} m')

    # Safe ±0.05 m movement (same as round-trip demo)
    target = current + 0.05
    if target > 0.30:
        target = current - 0.05

    print(f'\n  Proposed movement (small and safe):')
    print(f'    torso_lift_joint  {current:+.4f} m  →  {target:+.4f} m')
    print(f'    Duration: 3.0 seconds')
    print(f'\n    ⚠  This will physically move the robot\'s torso!')

    response = input('\n  Send this command to the PR2? [y/N]: ').strip().lower()

    if response != 'y':
        print('\n  Skipped. No commands were sent to the robot.')
        print('  Read-only demo complete! Phase 1 verified successfully.')
        return

    # Build PyCRAM designator and ground it
    designator = TorsoDesignator(
        target_height=target,
        start_height=current,
        duration=3.0,
    )
    node.execute(designator)

    # Updated joint summary after movement
    print('\n  Collecting updated joint positions …')
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.2)
    node.print_joint_summary()

    print('\n  PHASE 2 PASSED — PyCRAM round-trip demo complete!')
    print('  ROS2 TorsoDesignator → bridge → PR2 torso moved → joint_states back!')


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = PR2RealNode()
    try:
        real_robot_plan(node)
    except KeyboardInterrupt:
        print('\n  Interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
