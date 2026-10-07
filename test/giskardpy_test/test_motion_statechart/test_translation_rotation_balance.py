"""
Plots how a pose goal trades its translation error off against its rotation error, for
pose offsets ranging from mostly translation to mostly rotation.

The plots are written next to this file, into :class:`BalancePlotFile`, to be looked at;
the tests themselves only check that every pose goal is reached.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointState
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.robots.tracy import TracyJoint
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% scenarios


@dataclass
class PoseOffset:
    """
    How far a pose goal lies from the tip, expressed in the tip's own frame.
    """

    label: str
    """
    What the offset is meant to show, used as the title of its plot.
    """

    translation: float
    """
    Distance to the goal position in meters, along :data:`TRANSLATION_DIRECTION`.
    """

    rotation: float
    """
    Angle to the goal orientation in radians, about :data:`ROTATION_AXIS`.
    """

    def goal_pose(self, tip: KinematicStructureEntity) -> Pose:
        """
        :return: The goal pose, relative to the tip.
        """
        x, y, z = self.translation * TRANSLATION_DIRECTION
        return Pose.from_xyz_axis_angle(
            x=x,
            y=y,
            z=z,
            axis=Vector3(*ROTATION_AXIS),
            angle=self.rotation,
            reference_frame=tip,
        )


TRANSLATION_DIRECTION = np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
"""
Direction of every translation offset in the tip frame, not aligned with any of its
axes.
"""

ROTATION_AXIS = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
"""
Axis of every rotation offset in the tip frame, not aligned with any of its axes.
"""

POSE_OFFSETS = [
    PoseOffset("little translation, much rotation", 0.02, np.radians(120)),
    PoseOffset("much translation, little rotation", 0.3, np.radians(5)),
    PoseOffset("much translation, much rotation", 0.25, np.radians(100)),
    PoseOffset("some translation, some rotation", 0.1, np.radians(45)),
    PoseOffset("little translation, little rotation", 0.02, np.radians(5)),
    PoseOffset("translation only", 0.2, 0.0),
    PoseOffset("rotation only", 0.0, np.radians(90)),
]
"""
Pose offsets from mostly translation to mostly rotation, with both errors small or
large.
"""


class BalancePlotFile(Enum):
    """
    Where the plot of each robot is written.
    """

    PR2 = Path(__file__).parent / "plots" / "translation_rotation_balance_pr2.pdf"
    TRACY = Path(__file__).parent / "plots" / "translation_rotation_balance_tracy.pdf"


# %% recording


@dataclass
class ErrorTrace:
    """
    The translation and rotation error of a pose goal at every control cycle.
    """

    times: list[float] = field(default_factory=list)
    """
    Time of every sample in seconds.
    """

    translation_errors: list[float] = field(default_factory=list)
    """
    Distance of the tip to the goal position in meters.
    """

    rotation_errors: list[float] = field(default_factory=list)
    """
    Angle between the tip and the goal orientation in radians.
    """

    def record(self, time: float, root_T_tip: np.ndarray, root_T_goal: np.ndarray):
        """
        Appends the errors of the tip pose against the goal pose.
        """
        rotation_difference = root_T_goal[:3, :3].T @ root_T_tip[:3, :3]
        cosine = np.clip((np.trace(rotation_difference) - 1) / 2, -1, 1)
        self.times.append(time)
        self.translation_errors.append(
            float(np.linalg.norm(root_T_goal[:3, 3] - root_T_tip[:3, 3]))
        )
        self.rotation_errors.append(float(np.arccos(cosine)))

    @staticmethod
    def settling_time(times: list[float], errors: list[float], threshold: float):
        """
        :return: The first time after which the error stays below `threshold`.
        """
        settled_from = len(errors)
        while settled_from > 0 and errors[settled_from - 1] <= threshold:
            settled_from -= 1
        return times[min(settled_from, len(times) - 1)]


@dataclass
class BalanceRun:
    """
    One pose goal, run on one robot, with the trace of its errors.
    """

    offset: PoseOffset
    """
    How far away the goal was.
    """

    goal: CartesianPose
    """
    The pose goal that was run.
    """

    trace: ErrorTrace
    """
    The errors of the tip at every control cycle.
    """

    reached: bool
    """
    Whether the pose goal was reached before the run gave up.
    """

    @property
    def translation_settling_time(self) -> float:
        return ErrorTrace.settling_time(
            self.trace.times,
            self.trace.translation_errors,
            self.goal.translation_threshold,
        )

    @property
    def rotation_settling_time(self) -> float:
        return ErrorTrace.settling_time(
            self.trace.times,
            self.trace.rotation_errors,
            self.goal.orientation_threshold,
        )


def run_pose_goal(
    world: World,
    root: KinematicStructureEntity,
    tip: KinematicStructureEntity,
    offset: PoseOffset,
    maximum_cycles: int = 2000,
) -> BalanceRun:
    """
    Moves the tip to a pose goal at `offset` and records its errors on every cycle, for
    at most `maximum_cycles`.
    """
    goal = CartesianPose(root_link=root, tip_link=tip, goal_pose=offset.goal_pose(tip))
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(goal)
    motion_statechart.add_node(EndMotion.when_true(goal))
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=motion_statechart)

    root_T_goal = world.compute_forward_kinematics_np(root, tip) @ (
        offset.goal_pose(tip).to_np()
    )
    trace = ErrorTrace()
    trace.record(0.0, world.compute_forward_kinematics_np(root, tip), root_T_goal)
    for _ in range(maximum_cycles):
        executor.tick()
        trace.record(
            executor.time.total_seconds(),
            world.compute_forward_kinematics_np(root, tip),
            root_T_goal,
        )
        if executor.motion_statechart.is_end_motion():
            return BalanceRun(offset=offset, goal=goal, trace=trace, reached=True)
    return BalanceRun(offset=offset, goal=goal, trace=trace, reached=False)


def run_every_pose_offset(
    world: World, root: KinematicStructureEntity, tip: KinematicStructureEntity
) -> list[BalanceRun]:
    """
    Runs a pose goal for every offset in :data:`POSE_OFFSETS`, each from the same start.
    """
    start_state = world.state._data.copy()
    runs = []
    for offset in POSE_OFFSETS:
        world.state._data[:] = start_state
        world.notify_state_change()
        runs.append(run_pose_goal(world, root, tip, offset))
    return runs


# %% plotting


def plot_runs(runs: list[BalanceRun], robot_name: str, plot_file: BalancePlotFile):
    """
    Plots both errors of every run over time, and every run's errors against each other.

    Each error over time is drawn on its own axis, translation in centimeters on the
    left and rotation in degrees on the right; dotted lines mark when each error settled
    below its threshold. The last panel normalizes both errors by their start value, so
    a run that reduces both at the same rate follows the diagonal.
    """
    columns = 2
    rows = int(np.ceil((len(runs) + 1) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows))
    axes = axes.flatten()
    for axis, run in zip(axes, runs):
        rotation_axis = axis.twinx()
        axis.plot(
            run.trace.times,
            np.array(run.trace.translation_errors) * 100,
            color="tab:blue",
        )
        rotation_axis.plot(
            run.trace.times,
            np.degrees(run.trace.rotation_errors),
            color="tab:red",
        )
        axis.axvline(run.translation_settling_time, color="tab:blue", linestyle=":")
        rotation_axis.axvline(
            run.rotation_settling_time, color="tab:red", linestyle=":"
        )
        axis.set_ylim(bottom=0)
        rotation_axis.set_ylim(bottom=0)
        axis.set_xlabel("time [s]")
        axis.set_ylabel("translation error [cm]", color="tab:blue")
        rotation_axis.set_ylabel("rotation error [deg]", color="tab:red")
        axis.set_title(
            f"{run.offset.label}{'' if run.reached else ' (not reached)'}\n"
            f"settled: translation {run.translation_settling_time:.2f} s, "
            f"rotation {run.rotation_settling_time:.2f} s",
            fontsize=9,
        )

    balance_axis = axes[len(runs)]
    for run in runs:
        if run.offset.translation == 0 or run.offset.rotation == 0:
            continue
        balance_axis.plot(
            np.array(run.trace.translation_errors) / run.trace.translation_errors[0],
            np.array(run.trace.rotation_errors) / run.trace.rotation_errors[0],
            label=run.offset.label,
        )
    balance_axis.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    balance_axis.set_xlabel("translation error / start")
    balance_axis.set_ylabel("rotation error / start")
    balance_axis.set_title("errors against each other, normalized", fontsize=9)
    balance_axis.legend(fontsize=7)
    for unused_axis in axes[len(runs) + 1 :]:
        unused_axis.set_visible(False)

    figure.suptitle(f"{robot_name}: translation against rotation of a pose goal")
    figure.tight_layout()
    plot_file.value.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_file.value)
    plt.close(figure)


def assert_every_goal_reached(runs: list[BalanceRun]):
    """
    Every run ended with both errors below the thresholds of its pose goal.
    """
    assert [run.offset.label for run in runs if not run.reached] == []
    for run in runs:
        assert run.trace.translation_errors[-1] <= run.goal.translation_threshold
        assert run.trace.rotation_errors[-1] <= run.goal.orientation_threshold


# %% robots

TRACY_ARM_POSE = {
    TracyJoint.RIGHT_SHOULDER_PAN: 0.0,
    TracyJoint.RIGHT_SHOULDER_LIFT: -np.pi / 2,
    TracyJoint.RIGHT_ELBOW: np.pi / 2,
    TracyJoint.RIGHT_WRIST_1: -np.pi / 2,
    TracyJoint.RIGHT_WRIST_2: -np.pi / 2,
    TracyJoint.RIGHT_WRIST_3: 0.0,
}
"""
An elbow-up pose of Tracy's right arm, away from its singularities.
"""


@pytest.fixture()
def tracy_world_copy(tracy_world: World) -> World:
    return deepcopy(tracy_world)


def test_pr2_translation_rotation_balance(
    pr2_world_state_reset: World, better_pr2_pose
):
    world = pr2_world_state_reset
    JointState.from_str_dict(better_pr2_pose, world=world).apply_to(world)
    root = world.get_kinematic_structure_entity_by_name("base_footprint")
    tip = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")

    runs = run_every_pose_offset(world, root, tip)

    plot_runs(runs, "PR2", BalancePlotFile.PR2)
    assert_every_goal_reached(runs)


def test_tracy_translation_rotation_balance(tracy_world_copy: World):
    world = tracy_world_copy
    JointState.from_str_dict(TRACY_ARM_POSE, world=world).apply_to(world)
    tip = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")

    runs = run_every_pose_offset(world, world.root, tip)

    plot_runs(runs, "Tracy", BalancePlotFile.TRACY)
    assert_every_goal_reached(runs)
