import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    ObservationStateValues,
    RobotiqGripperActionServerTask,
    ParallelGripperCommand,
)

TARGET_MODULE = "giskardpy.motion_statechart.ros2_nodes.ros_tasks"

@pytest.fixture
def mock_context():
    """
    Mocks the MotionStatechartContext, satisfying both the ROS layer
    and the Giskardpy spatial/world transform requirements.
    """
    context = MagicMock()

    # Mock ROS Context Extension
    ros_ext = MagicMock()
    ros_ext.ros_node = MagicMock()
    context.require_extension.return_value = ros_ext

    # Mock world spatial objects/matrices for navigation calculations
    mock_pos_array = np.array([1.0, 2.0, 3.0])
    mock_ori_array = np.array([0.0, 0.0, 0.0, 1.0])

    mock_spatial_object = MagicMock()
    mock_spatial_object.to_position.return_value.to_np.return_value = mock_pos_array
    mock_spatial_object.to_quaternion.return_value.to_np.return_value = mock_ori_array

    # Wire methods used in NavigateActionServerTask.build/build_msg
    context.world.transform.return_value = mock_spatial_object

    # Create a mock expression that handles inequalities (<, >, <=, >=) safely
    mock_symbolic_expr = MagicMock()
    mock_symbolic_expr.__lt__.return_value = True
    mock_symbolic_expr.__gt__.return_value = False
    mock_symbolic_expr.__le__.return_value = True
    mock_symbolic_expr.__ge__.return_value = False

    mock_fk = MagicMock()
    mock_fk.to_position.return_value.euclidean_distance.return_value = (
        mock_symbolic_expr
    )
    mock_fk.to_rotation_matrix.return_value.rotational_error.return_value = (
        mock_symbolic_expr
    )
    context.world.compose_forward_kinematics_expression.return_value = mock_fk

    return context


# Tests for RobotiqGripperActionServerTask


@patch(f"{TARGET_MODULE}.ActionClient")
def test_gripper_task_build_and_msg(mock_action_client_cls, mock_context):
    """Verifies action client initialization and ParallelGripperCommand population."""
    task = RobotiqGripperActionServerTask(
        action_topic="/robotiq_gripper_controller/gripper_action",
        message_type=ParallelGripperCommand,
        target_position=0.085,
        target_velocity=5.0,
        target_effort=40.0,
    )

    task.build(mock_context)

    # Verify the internal ActionClient instantiation
    mock_action_client_cls.assert_called_once_with(
        mock_context.require_extension().ros_node,
        ParallelGripperCommand,
        "/robotiq_gripper_controller/gripper_action",
    )

    # Convert native ROS array templates to python lists for assertion comparisons
    assert list(task._msg.command.position) == [0.085]
    assert list(task._msg.command.velocity) == [5.0]
    assert list(task._msg.command.effort) == [40.0]


def test_gripper_task_on_tick_states():
    """Validates Statechart state machine outputs depending on action results."""
    task = RobotiqGripperActionServerTask(
        action_topic="/gripper",
        message_type=ParallelGripperCommand,
        target_position=0.0,
    )

    # 1. Unknown state (no action server response returned yet)
    task._result = None
    assert task.on_tick(MagicMock()) == ObservationStateValues.UNKNOWN

    # Setup mock result structure
    mock_res = MagicMock()
    task._result = mock_res

    # 2. Reached goal -> TRUE
    mock_res.result.reached_goal = True
    mock_res.result.stalled = False
    assert task.on_tick(MagicMock()) == ObservationStateValues.TRUE

    # 3. Gripper stalled (contacted object / successful grasp) -> TRUE
    mock_res.result.reached_goal = False
    mock_res.result.stalled = True
    assert task.on_tick(MagicMock()) == ObservationStateValues.TRUE

    # 4. Failed sequence -> FALSE
    mock_res.result.reached_goal = False
    mock_res.result.stalled = False
    assert task.on_tick(MagicMock()) == ObservationStateValues.FALSE
