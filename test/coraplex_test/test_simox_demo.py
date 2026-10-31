"""
Standard manipulation consumes Simox candidates without custom motion actions.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import pytest

from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.external_interfaces.simox_grasp_planner import (
    SimoxConfiguration,
    SimoxGraspPlanner,
    SimoxResponse,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal
from semantic_digital_twin.spatial_types import Pose

from .test_simox_candidates import RecordedGrasps

# %% demo loading


@pytest.fixture
def simox_demo_module() -> ModuleType:
    """
    Load the standalone demo without starting ROS.
    """
    path = Path(__file__).parents[2] / "coraplex/demos/pr2_simox_demo/demo.py"
    specification = importlib.util.spec_from_file_location("simox_demo", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


# %% full simulated manipulation


def test_simox_candidates_execute_pick_and_place(simox_demo_module: ModuleType) -> None:
    module = simox_demo_module
    configuration = SimoxConfiguration.for_pr2(Path("pr2.xml"))
    service = RecordedGrasps(SimoxResponse([], [], []))
    demonstration = module.SimoxDemonstration(
        planner=SimoxGraspPlanner(configuration, service)
    )
    world = demonstration.build_simulated_world()
    context = demonstration.build_context(world)
    cereal = world.get_semantic_annotations_by_type(Cereal)[0]
    arm = context.robot.get_right_arm_if_specified()
    grasp = cereal.grasp_candidates()[0]
    base_T_tool = world.transform(
        arm.end_effector.tool_frame_goal_for_grasp(grasp.grasp_pose), context.robot.root
    )
    base_T_simox = base_T_tool.homogeneous_matrix @ configuration.simox_T_tool.inverse()
    service.response = SimoxResponse(
        [Pose.from_xyz_rpy(100, 100, 100), base_T_simox.pose], [0.9, 0.7], [False, True]
    )
    expected = cereal.root.global_pose.to_np().copy()
    expected[1, 3] += demonstration.placement_offset
    plan = demonstration.build_plan(context)
    with ExecutionEnvironment(ExecutionType.KINEMATICALLY_SIMULATED):
        plan.perform()
    np.testing.assert_allclose(cereal.root.global_pose.to_np(), expected, atol=0.01)
    assert cereal.root.parent_connection.parent is world.root


def test_recorded_grasps_complete_placement(simox_demo_module: ModuleType) -> None:
    """
    A grasp must allow both pickup and placement before it is selected.
    """
    module = simox_demo_module
    configuration = SimoxConfiguration.for_pr2(Path("pr2.xml"))
    service = RecordedGrasps(SimoxResponse([], [], []))
    demonstration = module.SimoxDemonstration(
        planner=SimoxGraspPlanner(configuration, service)
    )
    world = demonstration.build_simulated_world()
    context = demonstration.build_context(world)
    cereal = world.get_semantic_annotations_by_type(Cereal)[0]
    with Path(__file__).with_name("simox_recorded_grasps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    service.response = SimoxResponse(
        [
            Pose.from_xyz_quaternion(
                float(row["PositionX"]),
                float(row["PositionY"]),
                float(row["PositionZ"]),
                float(row["OrientationX"]),
                float(row["OrientationY"]),
                float(row["OrientationZ"]),
                float(row["OrientationW"]),
            )
            for row in rows
        ],
        [float(row["Quality"]) for row in rows],
        [False] * len(rows),
    )
    expected = cereal.root.global_pose.to_np().copy()
    expected[1, 3] += demonstration.placement_offset
    with ExecutionEnvironment(ExecutionType.KINEMATICALLY_SIMULATED):
        demonstration.build_plan(context).perform()
    np.testing.assert_allclose(cereal.root.global_pose.to_np(), expected, atol=0.01)
    assert cereal.root.parent_connection.parent is world.root


# %% ROS lifecycle


@pytest.mark.parametrize("fails", [False, True])
def test_demo_stops_executor_before_context_shutdown(
    simox_demo_module: ModuleType, fails: bool
) -> None:
    """
    The demonstration releases its executor before the outer ROS session exits.
    """
    module = simox_demo_module
    events = Mock()
    with (
        patch.object(sys, "argv", ["demo.py", "--robot-xml", "robot.xml"]),
        patch.object(module.RobotDemonstrationRosSession, "start") as start,
        patch.object(module, "SimoxDemonstration") as demonstration_type,
    ):
        demonstration = demonstration_type.return_value
        events.attach_mock(demonstration.stop_visualization, "stop_visualization")
        events.attach_mock(start.return_value.__exit__, "exit_context")
        if fails:
            demonstration.run.side_effect = RuntimeError("execution failed")
            with pytest.raises(RuntimeError):
                module.main()
        else:
            module.main()
        assert [call[0] for call in events.mock_calls] == [
            "stop_visualization",
            "exit_context",
        ]
