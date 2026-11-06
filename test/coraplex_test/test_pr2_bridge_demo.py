"""
PR2 demonstration entry points must work without contacting a robot.
"""

from __future__ import annotations

# %% imports

from pathlib import Path
import importlib
import subprocess
import sys
from unittest.mock import Mock

import pytest
from typing_extensions import TYPE_CHECKING

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal

if TYPE_CHECKING:
    from types import ModuleType
    from semantic_digital_twin.world import World

# %% demonstration loading


@pytest.fixture
def demo_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """
    Import the PR2 script without starting its ROS session or command line.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "coraplex/demos/pr2_ros_bridge_demo"
    )
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("pr2_demo")


# %% controller context


def test_real_run_uses_controller_world_and_node(
    demo_module: ModuleType, monkeypatch: pytest.MonkeyPatch, pr2_world_copy: World
) -> None:
    """
    A hardware run plans against the served world with the live ROS node.
    """
    demonstration = demo_module.PR2Demonstration(executor_type=RealRobotPlanExecutor)
    session = Mock(spec=RobotDemonstrationRosSession)
    session.node = Mock()
    session.fetch_world.return_value = pr2_world_copy
    start = Mock(return_value=session)
    synchronize = Mock()
    monkeypatch.setattr(RobotDemonstrationRosSession, "start", start)
    monkeypatch.setattr("coraplex.demonstrations.WorldSynchronizer", synchronize)
    local_world = Mock()
    monkeypatch.setattr(demonstration, "build_simulated_world", local_world)

    world = demonstration.acquire_world()
    executor = demonstration.create_executor(world)
    context = executor.context

    assert world is pr2_world_copy
    assert context.world is pr2_world_copy
    assert executor.ros_node is session.node
    start.assert_called_once_with(demonstration.ros_node_name)
    synchronize.assert_called_once_with(_world=pr2_world_copy, node=session.node)
    local_world.assert_not_called()
    executor.close()


# %% manipulation plans


def test_torso_uses_semantic_configurations(
    demo_module: ModuleType, pr2_world_copy: World
) -> None:
    """
    Torso motion uses model configurations instead of hardcoded joint targets.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.TORSO
    )
    executor = demonstration.create_executor(pr2_world_copy)
    [plan] = demonstration.build_statechart(executor.context).top_level_nodes
    assert [type(child) for child in plan.nodes] == [
        MoveTorsoAction,
        MoveTorsoAction,
    ]
    assert [child.torso_state for child in plan.nodes] == [
        TorsoState.HIGH,
        TorsoState.LOW,
    ]
    executor.close()


def test_parking_includes_both_semantic_arms(
    demo_module: ModuleType, pr2_world_copy: World
) -> None:
    """
    Parking resolves both arms from the robot rather than copying joint names.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.ARMS
    )
    executor = demonstration.create_executor(pr2_world_copy)
    context = executor.context
    [plan] = demonstration.build_statechart(context).top_level_nodes
    assert [type(child) for child in plan.nodes] == [ParkArmsAction]
    assert plan.nodes[0].arms == list(
        context.require_extension(RobotAccess).robot.all_arms
    )
    executor.close()


def test_pick_and_place_preserves_object_pose_until_execution(
    demo_module: ModuleType,
) -> None:
    """
    Constructing the plan cannot attach or relocate the object ahead of execution.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.PICK_AND_PLACE
    )
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    assert demonstration.is_scene_populated(world) is True
    cereal = world.get_semantic_annotations_by_type(Cereal)[0]
    original_parent = cereal.root.parent_connection.parent
    original_pose = cereal.root.global_pose.to_np().copy()
    executor = demonstration.create_executor(world)
    context = executor.context

    [plan] = demonstration.build_statechart(context).top_level_nodes

    assert [type(child) for child in plan.nodes] == [
        ParkArmsAction,
        MoveTorsoAction,
        PickUpAction,
        PlaceAction,
        ParkArmsAction,
    ]
    pickup = plan.nodes[2]
    placement = plan.nodes[3]
    assert (
        pickup.arm
        is context.require_extension(RobotAccess).robot.get_right_arm_if_specified()
    )
    assert pickup.grasp.grasp_pose.reference_frame is cereal.root
    assert placement.object_designator is cereal
    assert float(placement.target_location.position.y) == pytest.approx(
        float(cereal.root.global_pose.position.y) + demonstration.placement_offset
    )
    assert cereal.root.parent_connection.parent is original_parent
    assert cereal.root.global_pose.to_np() == pytest.approx(original_pose)
    executor.close()


# %% simulated execution


def test_pick_and_place_executes_in_simulation(demo_module: ModuleType) -> None:
    """
    The complete sequence leaves the cereal at its target and released from the arm.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.PICK_AND_PLACE
    )
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    executor = demonstration.create_executor(world)
    statechart = demonstration.build_statechart(executor.context)
    [plan] = statechart.top_level_nodes
    placement = plan.nodes[3]
    expected_pose = placement.target_location.to_np().copy()

    executor.compile(statechart)
    executor.execute()

    cereal = world.get_semantic_annotations_by_type(Cereal)[0]
    assert cereal.root.parent_connection.parent is world.root
    assert cereal.root.global_pose.to_np() == pytest.approx(expected_pose, abs=0.01)
    executor.close()


# %% command line


def test_demo_help_needs_no_controller() -> None:
    """
    Every PR2 entry point can describe its options without starting ROS.
    """
    directory = (
        Path(__file__).resolve().parents[2]
        / "coraplex"
        / "demos"
        / "pr2_ros_bridge_demo"
    )
    for script in sorted(directory.glob("pr2_*.py")):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr


# %% reusable scene specifications


def test_scene_specification_contains_objects_without_a_robot(
    demo_module: ModuleType,
) -> None:
    """
    The controller scene can be merged without spawning a duplicate PR2.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.PICK_AND_PLACE
    )
    world = demonstration.scene_specification().create_world()
    assert world.get_semantic_annotations_by_type(demonstration.used_robot) == []
    cereal = world.get_semantic_annotations_by_type(Cereal)
    assert len(cereal) == 1
    assert cereal[0].root.global_pose.to_np() == pytest.approx(
        demonstration.cereal_start.to_np()
    )
    demonstration.populate_scene(world)
    assert world.get_semantic_annotations_by_type(Cereal) == cereal


def test_unpopulated_controller_world_is_detected(
    demo_module: ModuleType, pr2_world_copy: World
) -> None:
    """
    A controller with only the PR2 still needs the manipulation scene.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.PICK_AND_PLACE
    )
    assert demonstration.is_scene_populated(pr2_world_copy) is False


# %% measured controller scenes


def test_real_pick_and_place_requires_measured_object(
    demo_module: ModuleType, monkeypatch: pytest.MonkeyPatch, pr2_world_copy: World
) -> None:
    """
    A missing controller object must not be replaced by the simulated scene.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.PICK_AND_PLACE,
        executor_type=RealRobotPlanExecutor,
    )
    session = Mock(spec=RobotDemonstrationRosSession)
    session.node = Mock()
    session.fetch_world.return_value = pr2_world_copy
    monkeypatch.setattr(
        RobotDemonstrationRosSession, "start", Mock(return_value=session)
    )
    monkeypatch.setattr("coraplex.demonstrations.WorldSynchronizer", Mock())
    build_scene = Mock()
    monkeypatch.setattr(demonstration, "build_scene", build_scene)
    with pytest.raises(WorldEntityNotFoundError):
        demonstration.acquire_world()
    build_scene.assert_not_called()


def test_real_pick_and_place_reuses_unannotated_controller_object(
    demo_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A measured cereal body gains its annotation without duplicating the scene.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.PICK_AND_PLACE
    )
    controller_world = demonstration.build_simulated_world()
    cereal = controller_world.get_semantic_annotations_by_type(Cereal)[0]
    with controller_world.modify_world():
        controller_world.remove_semantic_annotation(cereal)
    original_bodies = list(controller_world.bodies)
    original_pose = cereal.root.global_pose.to_np().copy()
    demonstration.executor_type = RealRobotPlanExecutor
    session = Mock(spec=RobotDemonstrationRosSession)
    session.node = Mock()
    session.fetch_world.return_value = controller_world
    monkeypatch.setattr(
        RobotDemonstrationRosSession, "start", Mock(return_value=session)
    )
    monkeypatch.setattr("coraplex.demonstrations.WorldSynchronizer", Mock())

    world = demonstration.acquire_world()
    demonstration.build_context_extensions(world)
    demonstration.build_context_extensions(world)

    assert demonstration.ros_node is session.node
    assert world is controller_world
    assert world.bodies == original_bodies
    annotations = world.get_semantic_annotations_by_type(Cereal)
    assert len(annotations) == 1
    assert annotations[0].root is cereal.root
    assert cereal.root.global_pose.to_np() == pytest.approx(original_pose)


# %% command-line executor selection


@pytest.mark.parametrize(
    "mode, executor_type",
    [("sim", KinematicallySimulatedPlanExecutor), ("real", RealRobotPlanExecutor)],
)
def test_existing_cli_selects_executor_without_executing_hardware(
    demo_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    executor_type: type,
) -> None:
    """
    Existing mode names select the shared executor while preserving the task.
    """
    observed = []
    monkeypatch.setattr(
        demo_module.PR2Demonstration, "run", lambda demo: observed.append(demo)
    )
    monkeypatch.setattr(
        "sys.argv",
        ["pr2_demo.py", "--task", "arms", "--mode", mode, "--collision-avoidance"],
    )
    demo_module.main()
    [demonstration] = observed
    assert demonstration.executor_type is executor_type
    assert demonstration.task is demo_module.DemonstrationTask.ARMS
    assert demonstration.collision_avoidance is True
