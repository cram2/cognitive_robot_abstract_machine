"""
PR2 Motion Mapping for ROS1 Bridge Integration
==============================================

Registers PR2-specific AlternativeMotion handlers that route CoraPlex designators
through the ROS 1/2 bridge instead of Giskard motion planning.

Architecture
------------
The host (cram-env) runs CoraPlex planning ONLY.  Actual ROS 2 publishing is
delegated to the bridge container via ``docker exec``, avoiding all
FastDDS v2 (Foxy) / v3 (Jazzy) incompatibility issues.

Full execution chain::

    MoveTorsoAction(TorsoState.HIGH).perform()  — host, pure Python
        ↓
    MoveJointsMotion(['torso_lift_joint'], [0.30])
        ↓   AlternativeMotion.check_for_alternative()
    PR2MoveJointsMotion._motion_chart  (ExecutionType.BRIDGE, robot=PR2)
        ↓   MotionExecutor._execute_for_bridge()
    PR2ROS1TrajectoryTask.on_start()
        ↓   subprocess: docker exec <container> python3 /workspace/pr2_publish_trajectory.py
    Foxy rclpy inside container  →  dynamic_bridge  →  PR2 controller
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from giskardpy.motion_statechart.data_types import ObservationStateValues

from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans import MoveJointsMotion, LookingMotion
from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.robot_plans.motions.navigation import MoveMotion
from semantic_digital_twin.robots.pr2 import PR2

logger = logging.getLogger(__name__)


# %% Container detection helpers

# Known container names in priority order.
# docker compose names containers as <project>-<service>-<index>
_CANDIDATE_CONTAINERS = [
    "ros1_2_bridge-sim_bridge-1",  # simulation
    "ros1_2_bridge-bridge-1",  # real robot
]

_DEFAULT_JOINT_POSITIONS = {
    "torso_lift_joint": 0.0,
    "head_pan_joint": 0.0,
    "head_tilt_joint": 0.0,
    # arms default to 0 — override if needed
}


def _find_bridge_container() -> Optional[str]:
    """
    Return the first running bridge container name, or None.
    """
    for name in _CANDIDATE_CONTAINERS:
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and "true" in result.stdout:
                logger.debug(f"Found running container: {name}")
                return name
        except (subprocess.SubprocessError, OSError):
            pass
    return None


def _controller_topic(joint_names: List[str]) -> str:
    """
    Map joint names to their ROS 1 controller command topic.
    """
    first = joint_names[0].lower()
    if "head" in first:
        return "/head_traj_controller/command"
    elif "torso" in first:
        return "/torso_controller/command"
    elif "gripper" in first:
        # Must check gripper BEFORE arm (r_gripper_joint starts with r_)
        if first.startswith("l_"):
            return "/l_gripper_controller/command"
        else:
            return "/r_gripper_controller/command"
    elif first.startswith("l_"):
        return "/l_arm_controller/command"
    elif first.startswith("r_"):
        return "/r_arm_controller/command"
    else:
        raise ValueError(f"Cannot determine controller for joint: {first}")


# %% PR2ROS1TrajectoryTask — duck-typed Task (no giskardpy.graph_node.Task base)


@dataclass
class PR2ROS1TrajectoryTask:
    """
    Sends joint trajectories to the PR2 via docker exec into the bridge container.

    Lifecycle (called by MotionExecutor._execute_for_bridge):
        1. ``build(None)``    — find the bridge container, detect current joint pos
        2. ``on_start(None)`` — run publisher script inside the container
        3. ``on_tick(None)``  — wait for duration to elapse, return TRUE when done
    """

    joint_names: List[str]
    positions: List[float]
    duration_sec: float = 5.0
    timeout_sec: float = 15.0

    # %% Runtime state
    _container: Optional[str] = field(init=False, default=None)
    _start_positions: List[float] = field(init=False, default_factory=list)
    _start_time: Optional[float] = field(init=False, default=None)
    _done: bool = field(init=False, default=False)

    # %% Phase 1: build

    def build(self, context=None):
        """
        Find the bridge container and determine start positions.
        """
        self._container = _find_bridge_container()
        if self._container:
            logger.info(f"[bridge] Using container: {self._container}")
        else:
            logger.warning("[bridge] No bridge container found — is docker compose up?")

        # We will attempt to read the real joint positions in on_start().
        # For now, default to 0.0
        self._start_positions = [0.0] * len(self.joint_names)

    # %% Phase 2: on_start

    def on_start(self, context=None):
        """
        Run the publisher script inside the bridge container via docker exec.
        """
        if not self._container:
            logger.error("[bridge] No container — cannot publish")
            self._done = True
            return

        # 1. Read real joint states
        state_cmd = (
            "source /opt/ros/noetic/setup.bash && "
            "([ -f /catkin_ws/install/setup.bash ] && source /catkin_ws/install/setup.bash || true) && "
            "python3 /workspace/read_robot_state.py"
        )
        try:
            state_result = subprocess.check_output(
                ["docker", "exec", self._container, "bash", "-c", state_cmd],
                text=True,
                timeout=15.0,
            )
            raw_out = state_result.strip()
            json_start = raw_out.find("{")
            if json_start != -1:
                state_dict = json.loads(raw_out[json_start:])
                self._start_positions = [
                    float(state_dict.get(name, _DEFAULT_JOINT_POSITIONS.get(name, 0.0)))
                    for name in self.joint_names
                ]
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
            logger.warning(f"[bridge] Failed to read real joint state: {e}")
            # fallback to defaults
            self._start_positions = [
                _DEFAULT_JOINT_POSITIONS.get(name, 0.0) for name in self.joint_names
            ]

        controller = _controller_topic(self.joint_names)
        params = {
            "joint_names": self.joint_names,
            "start_positions": self._start_positions,
            "positions": self.positions,
            "duration_sec": self.duration_sec,
            "controller": controller,
        }
        params_json = json.dumps(params)

        logger.info(f"[bridge] Trajectory: {self.joint_names}")
        logger.info(
            f"[bridge]   start  = {[round(p, 3) for p in self._start_positions]}"
        )
        logger.info(f"[bridge]   target = {[round(p, 3) for p in self.positions]}")
        logger.info(f"[bridge]   → {controller}  ({self.duration_sec}s)")
        logger.info(f"[bridge] Running publisher inside {self._container} ...")

        cmd = (
            f"source /opt/ros/foxy/setup.bash && "
            f"python3 /workspace/pr2_publish_trajectory.py '{params_json}'"
        )

        try:
            result = subprocess.run(
                ["docker", "exec", self._container, "bash", "-c", cmd],
                timeout=self.duration_sec + 10,
                text=True,
            )
            # Log container output
            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    logger.info(f"  {line}")
            if result.returncode == 0:
                logger.info("[bridge] Publisher finished successfully")
            else:
                logger.error(f"[bridge] Publisher exited with code {result.returncode}")
                if result.stderr:
                    logger.error(result.stderr[:500])
        except subprocess.TimeoutExpired:
            logger.error("[bridge] Publisher timed out")
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"[bridge] docker exec failed: {e}")

        self._start_time = time.time()
        self._done = True

    # %% Phase 3: on_tick

    def on_tick(self, context=None) -> ObservationStateValues:
        """
        Return TRUE once on_start() has completed.
        """
        if self._done:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


# %% PR2MoveJointsMotion — AlternativeMotion[PR2]


@dataclass
class PR2MoveJointsMotion(MoveJointsMotion, AlternativeMotion[PR2]):
    """
    PR2-specific handler for MoveJointsMotion with BRIDGE execution.

    Activated when:
        - ``MotionExecutor.execution_type == ExecutionType.BRIDGE``
        - The current robot is an instance of ``PR2``

    Routes to PR2ROS1TrajectoryTask which publishes via docker exec into
    the bridge container (Foxy rclpy), avoiding DDS version mismatches.
    """

    execution_type = ExecutionType.BRIDGE

    @property
    def _motion_chart(self) -> PR2ROS1TrajectoryTask:
        return PR2ROS1TrajectoryTask(
            joint_names=list(self.names),
            positions=list(self.positions),
            duration_sec=5.0,
            timeout_sec=15.0,
        )

    def perform(self):
        """
        Handled by MotionExecutor._execute_for_bridge().
        """
        return


# %% PR2LookingMotion — AlternativeMotion[PR2]


@dataclass
class PR2LookingMotion(LookingMotion, AlternativeMotion[PR2]):
    """
    PR2-specific handler for LookingMotion with BRIDGE execution.

    Points the head to neutral (pan=0, tilt=0).
    """

    execution_type = ExecutionType.BRIDGE

    @property
    def _motion_chart(self) -> PR2ROS1TrajectoryTask:
        return PR2ROS1TrajectoryTask(
            joint_names=["head_pan_joint", "head_tilt_joint"],
            positions=[0.0, 0.0],
            duration_sec=2.0,
            timeout_sec=8.0,
        )

    def perform(self):
        return


# %% PR2NavigateTask — base navigation via base_cmd_vel.py (docker exec)


@dataclass
class PR2NavigateTask:
    """
    Moves the PR2 base by publishing /cmd_vel via docker exec (ROS 1 rospy).

    Uses base_cmd_vel.py inside the bridge container — open-loop velocity
    control for a computed duration based on the desired displacement.
    No Giskard dependency, avoids the OmniDriveSample class-name issue.

    Fields
    ------
    dx, dy : displacement in world X/Y (metres)
    speed  : translational speed (m/s, default 0.1)
    """

    target_x: float
    target_y: float
    speed: float = 0.1

    _container: Optional[str] = field(init=False, default=None)
    _done: bool = field(init=False, default=False)

    def build(self, context=None):
        self._container = _find_bridge_container()
        if not self._container:
            logger.warning("[bridge_nav] No bridge container found")

    def on_start(self, context=None):
        if not self._container:
            logger.error("[bridge_nav] No container")
            self._done = True
            return

        import math

        # 1. Get true current state from ROS 1
        state_cmd = (
            "source /opt/ros/noetic/setup.bash && "
            "([ -f /catkin_ws/install/setup.bash ] && source /catkin_ws/install/setup.bash || true) && "
            "python3 /workspace/read_robot_state.py"
        )
        try:
            state_result = subprocess.check_output(
                ["docker", "exec", self._container, "bash", "-c", state_cmd],
                text=True,
                timeout=15.0,
            )
            raw_out = state_result.strip()
            json_start = raw_out.find("{")
            if json_start != -1:
                state = json.loads(raw_out[json_start:])
            else:
                state = {}
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
            logger.warning(f"[bridge_nav] Could not read robot state: {e}")
            state = {}

        base_x = float(state.get("base_x", 1.5))
        base_y = float(state.get("base_y", 2.5))
        base_yaw = float(state.get("base_yaw", 0.0))

        # 2. Compute world displacement
        dx_world = self.target_x - base_x
        dy_world = self.target_y - base_y

        # 3. Rotate world displacement to base_footprint frame
        c = math.cos(-base_yaw)
        s = math.sin(-base_yaw)
        dx_base = c * dx_world - s * dy_world
        dy_base = s * dx_world + c * dy_world

        distance = math.sqrt(dx_base**2 + dy_base**2)
        if distance < 0.01:
            logger.info("[bridge_nav] Already at target")
            self._done = True
            return

        logger.info(
            f"[bridge_nav] Base pos: ({base_x:.2f}, {base_y:.2f})  Target: ({self.target_x:.2f}, {self.target_y:.2f})"
        )
        logger.info(
            f"[bridge_nav] base_cmd_vel: dx={dx_base:.2f}  dy={dy_base:.2f}  speed={self.speed:.2f} m/s"
        )

        params = json.dumps({"dx": dx_base, "dy": dy_base, "speed": self.speed})
        cmd = (
            f"source /opt/ros/noetic/setup.bash && "
            "([ -f /catkin_ws/install/setup.bash ] && source /catkin_ws/install/setup.bash || true) && "
            f"python3 /workspace/base_cmd_vel.py '{params}'"
        )
        timeout = distance / self.speed + 10.0
        try:
            result = subprocess.run(
                ["docker", "exec", self._container, "bash", "-c", cmd],
                timeout=timeout,
                text=True,
            )
            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    logger.info(f"  {line}")
            if result.returncode == 0:
                logger.info("[bridge_nav] Navigation complete")
            else:
                logger.error(f"[bridge_nav] Exited {result.returncode}")
                if result.stderr:
                    logger.error(result.stderr[:500])
        except subprocess.TimeoutExpired:
            logger.error("[bridge_nav] Timed out")
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"[bridge_nav] docker exec failed: {e}")
        self._done = True

    def on_tick(self, context=None):
        return (
            ObservationStateValues.TRUE if self._done else ObservationStateValues.FALSE
        )


# %% PR2NavigateMotion — AlternativeMotion[PR2]


@dataclass
class PR2NavigateMotion(MoveMotion, AlternativeMotion[PR2]):
    """
    PR2-specific handler for MoveMotion with BRIDGE execution.

    Routes base navigation through base_cmd_vel.py inside the bridge container (ROS 1),
    avoiding the ROS 2 action server discovery issue where Giskard's action server
    disappears between goals.
    """

    execution_type = ExecutionType.BRIDGE

    @property
    def _motion_chart(self) -> PR2NavigateTask:
        x = float(self.target.position.x)
        y = float(self.target.position.y)
        return PR2NavigateTask(target_x=x, target_y=y, speed=0.1)

    def perform(self):
        return
