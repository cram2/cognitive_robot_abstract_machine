"""
PR2 demonstration entry points must work without contacting a robot.
"""

from pathlib import Path
import importlib.util
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal
from semantic_digital_twin.world import World

# %% demonstration loading


@pytest.fixture
def demo_module() -> ModuleType:
    """
    Load the PR2 script without invoking its command line entry point.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "coraplex/demos/pr2_ros_bridge_demo/pr2_demo.py"
    )
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[path.stem] = module
    specification.loader.exec_module(module)
    return module


# %% controller context


def test_real_run_uses_controller_world_and_node(
    demo_module: ModuleType, monkeypatch: pytest.MonkeyPatch, pr2_world_copy: World
) -> None:
    """
    A hardware run plans against the served world with the live ROS node.
    """
    demonstration = demo_module.PR2Demonstration(execution_type=ExecutionType.REAL)
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
    context = demonstration.build_context(world)

    assert world is pr2_world_copy
    assert context.world is pr2_world_copy
    assert context.ros_node is session.node
    start.assert_called_once_with(demonstration.ros_node_name)
    synchronize.assert_called_once_with(_world=pr2_world_copy, node=session.node)
    local_world.assert_not_called()


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
    plan = demonstration.build_plan(demonstration.build_context(pr2_world_copy))
    assert [type(child.action) for child in plan.children] == [
        MoveTorsoAction,
        MoveTorsoAction,
    ]
    assert [child.action.torso_state for child in plan.children] == [
        TorsoState.HIGH,
        TorsoState.LOW,
    ]


def test_parking_includes_both_semantic_arms(
    demo_module: ModuleType, pr2_world_copy: World
) -> None:
    """
    Parking resolves both arms from the robot rather than copying joint names.
    """
    demonstration = demo_module.PR2Demonstration(
        task=demo_module.DemonstrationTask.ARMS
    )
    context = demonstration.build_context(pr2_world_copy)
    plan = demonstration.build_plan(context)
    assert [type(child.action) for child in plan.children] == [ParkArmsAction]
    assert plan.children[0].action.arms == list(context.robot.all_arms)


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
    context = demonstration.build_context(world)

    plan = demonstration.build_plan(context)

    assert [type(child.action) for child in plan.children] == [
        ParkArmsAction,
        MoveTorsoAction,
        PickUpAction,
        PlaceAction,
        ParkArmsAction,
    ]
    pickup = plan.children[2].action
    placement = plan.children[3].action
    assert pickup.arm is context.robot.get_right_arm_if_specified()
    assert pickup.grasp.grasp_pose.reference_frame is cereal.root
    assert placement.object_designator is cereal
    assert float(placement.target_location.position.y) == pytest.approx(
        float(cereal.root.global_pose.position.y) + demonstration.placement_offset
    )
    assert cereal.root.parent_connection.parent is original_parent
    assert cereal.root.global_pose.to_np() == pytest.approx(original_pose)


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
    plan = demonstration.build_plan(demonstration.build_context(world))
    placement = plan.children[3].action
    expected_pose = placement.target_location.to_np().copy()

    with ExecutionEnvironment(ExecutionType.KINEMATICALLY_SIMULATED):
        plan.perform()

    cereal = world.get_semantic_annotations_by_type(Cereal)[0]
    assert cereal.root.parent_connection.parent is world.root
    assert cereal.root.global_pose.to_np() == pytest.approx(expected_pose, abs=0.01)


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
    world = demonstration.scene_specification().to_domain_object()
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
        execution_type=ExecutionType.REAL,
    )
    session = Mock(spec=RobotDemonstrationRosSession)
    session.node = Mock()
    session.fetch_world.return_value = pr2_world_copy
    monkeypatch.setattr(
        RobotDemonstrationRosSession, "start", Mock(return_value=session)
    )
    monkeypatch.setattr("coraplex.demonstrations.WorldSynchronizer", Mock())
    with pytest.raises(WorldEntityNotFoundError):
        demonstration.acquire_world()


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
    demonstration.execution_type = ExecutionType.REAL
    session = Mock(spec=RobotDemonstrationRosSession)
    session.node = Mock()
    session.fetch_world.return_value = controller_world
    monkeypatch.setattr(
        RobotDemonstrationRosSession, "start", Mock(return_value=session)
    )
    monkeypatch.setattr("coraplex.demonstrations.WorldSynchronizer", Mock())

    world = demonstration.acquire_world()
    demonstration.build_context(world)

    assert world is controller_world
    assert world.bodies == original_bodies
    annotations = world.get_semantic_annotations_by_type(Cereal)
    assert len(annotations) == 1
    assert annotations[0].root is cereal.root
    assert cereal.root.global_pose.to_np() == pytest.approx(original_pose)
