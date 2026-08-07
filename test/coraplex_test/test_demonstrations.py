"""
Coverage for the scaffolding demonstrations share.

The pieces exercised here are the ones a second demonstration would otherwise re-derive:
which world a run acts on, whether it has to spawn its scene, and who owns the ROS
context. None of it needs a controller.
"""

from dataclasses import dataclass, field

import pytest
import rclpy

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import code
from coraplex.plans.plan_node import PlanNode
from coraplex.demonstrations import RobotDemonstration, RobotDemonstrationRosSession
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World


class PlanDeliberatelyFailed(Exception):
    """
    Raised by a demonstration whose plan is meant to fail.
    """


@dataclass(kw_only=True)
class RecordingDemonstration(RobotDemonstration):
    """
    A demonstration that records which of its hooks ran and in which environment.
    """

    world: World
    """
    World handed to this demonstration instead of one it builds itself.
    """

    scene_already_populated: bool = False
    """
    What :meth:`is_scene_populated` reports.
    """

    fail_the_plan: bool = False
    """
    Whether the plan raises :class:`PlanDeliberatelyFailed` instead of recording.
    """

    populate_scene_calls: int = 0
    """
    How often the scene was spawned.
    """

    tear_down_calls: int = 0
    """
    How often the ROS session was released.
    """

    observed_execution_type: ExecutionType | None = field(default=None)
    """
    Execution type in force while the plan ran.
    """

    observed_collision_avoidance: bool | None = field(default=None)
    """
    Collision avoidance setting in force while the plan ran.
    """

    def build_simulated_world(self) -> World:
        return self.world

    def is_scene_populated(self, world: World) -> bool:
        return self.scene_already_populated

    def populate_scene(self, world: World) -> None:
        self.populate_scene_calls += 1

    def build_context(self, world: World) -> Context:
        return Context(world, world.get_semantic_annotations_by_type(MinimalRobot)[0])

    def build_plan(self, context: Context) -> PlanNode:
        return code(self.run_plan_body, context)

    def run_plan_body(self) -> None:
        """
        Record the execution environment, or fail if this demonstration is meant to.
        """
        if self.fail_the_plan:
            raise PlanDeliberatelyFailed()
        self.observed_execution_type = GiskardExecutable.execution_type
        self.observed_collision_avoidance = GiskardExecutable.collision_avoidance

    def tear_down(self) -> None:
        self.tear_down_calls += 1
        super().tear_down()


# %% world acquisition


def test_simulated_run_still_builds_its_own_world(cylinder_bot_world):
    """
    A simulated run builds its own world rather than fetching one from a controller,
    even though it starts a ROS session of its own to visualize that world.
    """
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot
    )

    acquired_world = demonstration.acquire_world()

    assert acquired_world is cylinder_bot_world
    assert demonstration.ros_session is not None
    assert demonstration.ros_node is not None

    demonstration.tear_down()


# %% scene population


def test_scene_is_spawned_when_the_world_does_not_have_it(cylinder_bot_world):
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot
    )

    demonstration.run()

    assert demonstration.populate_scene_calls == 1


def test_scene_is_not_spawned_again_into_a_world_that_has_it(cylinder_bot_world):
    """
    A run against a controller receives a world that may already hold this
    demonstration's objects, and must not spawn a second copy of them.
    """
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world,
        used_robot=MinimalRobot,
        scene_already_populated=True,
    )

    demonstration.run()

    assert demonstration.populate_scene_calls == 0


# %% execution environment


def test_plan_runs_in_the_demonstrations_execution_environment(cylinder_bot_world):
    """
    The plan is what the execution type and collision avoidance settings exist for, so
    they have to be in force while it runs and restored once it is done.
    """
    previous_execution_type = GiskardExecutable.execution_type
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world,
        used_robot=MinimalRobot,
        execution_type=ExecutionType.SIMULATED,
        collision_avoidance=True,
    )

    demonstration.run()

    assert demonstration.observed_execution_type is ExecutionType.SIMULATED
    assert demonstration.observed_collision_avoidance is True
    assert GiskardExecutable.execution_type is previous_execution_type
    assert GiskardExecutable.collision_avoidance is False


def test_run_returns_the_world_it_acted_on(cylinder_bot_world):
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot
    )

    assert demonstration.run() is cylinder_bot_world


# %% tear down


def test_tear_down_runs_when_the_plan_fails(cylinder_bot_world):
    """
    A demonstration that dies mid-plan still has to release what it acquired, or a real
    run leaves its ROS session behind.
    """
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world, used_robot=MinimalRobot, fail_the_plan=True
    )

    with pytest.raises(PlanDeliberatelyFailed):
        demonstration.run()

    assert demonstration.tear_down_calls == 1


# %% ros context ownership


def test_session_releases_a_context_it_started():
    """
    A demonstration run on its own owns the ROS context and must give it back, so the
    process it ran in is left as it was found.
    """
    assert not rclpy.ok(), "another test left a ROS context running"
    session = RobotDemonstrationRosSession.start("context_ownership_probe")

    assert session.owns_context

    session.stop()
    assert not rclpy.ok()


def test_session_leaves_a_context_somebody_else_started(rclpy_node):
    """
    Inside a process that already has a ROS context -- a test, or a larger application
    embedding the demonstration -- the session must not shut that context down.
    """
    session = RobotDemonstrationRosSession.start("context_borrowing_probe")

    assert not session.owns_context

    session.stop()
    assert rclpy.ok()
