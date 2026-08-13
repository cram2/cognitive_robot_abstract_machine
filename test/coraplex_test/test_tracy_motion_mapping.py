import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from control_msgs.action import ParallelGripperCommand

from coraplex.datastructures.enums import ExecutionType, Arms
from semantic_digital_twin.datastructures.definitions import GripperState
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import RobotiqGripperActionServerTask

from coraplex.alternative_motion_mappings.tracy_motion_mapping import TracyJointMotionMapping, TracyGripMotion


class TestTracyJointMotionMapping:
    """
    Unit tests for TracyJointMotionMapping.
    """

    def test_execution_type(self):
        """Verify the motion targets real hardware execution."""
        assert TracyJointMotionMapping.execution_type == ExecutionType.REAL

    def test_motion_chart_fallback(self):
        """Verify that _motion_chart delegates successfully to its superclass."""
        mock_goal = MagicMock()

        motion = TracyJointMotionMapping(names=["joint1"], positions=[0.0])

        with patch("coraplex.robot_plans.MoveJointsMotion._motion_chart",
                   new_callable=PropertyMock) as mock_super_chart:
            mock_super_chart.return_value = mock_goal
            assert motion._motion_chart == mock_goal

    def test_perform_runs_without_side_effects(self):
        """Ensure perform() executes cleanly without exceptions."""
        motion = TracyJointMotionMapping(names=["joint1"], positions=[0.0])
        assert motion.perform() is None


class TestTracyGripMotion:
    """
    Unit tests for TracyGripMotion focusing on initialization parameters,
    mutually exclusive conditions, and hardware task mappings.
    """

    def test_execution_type(self):
        """Verify the motion targets real hardware execution."""
        assert TracyGripMotion.execution_type == ExecutionType.REAL

    def test_mutual_exclusivity_validation(self):
        """
        Verify that initialization fails if both or neither 'motion'
        and 'position' are specified.
        """
        # Neither parameter provided
        with pytest.raises(ValueError, match="You must specify either 'motion' or 'position'."):
            TracyGripMotion(gripper=Arms.LEFT)

        # Both parameters provided
        with pytest.raises(ValueError, match="Cannot specify both 'motion' and 'position' at the same time."):
            TracyGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN, position=0.3)

    def test_motion_chart_left_arm_semantic_open(self):
        """ Verify Left Arm discrete OPEN state maps to the proper topic and position (0.0). """
        motion = TracyGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert chart.action_topic == "/left_gripper/robotiq_gripper_controller/gripper_cmd"
        assert chart.message_type == ParallelGripperCommand
        assert chart.target_position == 0.0

    def test_motion_chart_right_arm_semantic_close(self):
        """ Verify Right Arm discrete CLOSE state maps to the proper topic and position (0.7). """
        motion = TracyGripMotion(gripper=Arms.RIGHT, motion=GripperState.CLOSE)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert chart.action_topic == "/right_gripper/robotiq_gripper_controller/gripper_cmd"
        assert chart.message_type == ParallelGripperCommand
        assert chart.target_position == 0.7

    def test_motion_chart_custom_position(self):
        """ Verify that passing an explicit position overrides the semantic default mappings. """
        custom_pos = 0.45
        motion = TracyGripMotion(gripper=Arms.LEFT, position=custom_pos)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert chart.action_topic == "/left_gripper/robotiq_gripper_controller/gripper_cmd"
        assert chart.target_position == custom_pos

    def test_motion_chart_missing_gripper_raises_error(self):
        """ An unassigned gripper attribute must bubble up a ValueError. """
        motion = TracyGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN)
        # Force gripper to None to simulate runtime state failures
        motion.gripper = None

        with pytest.raises(ValueError, match="No gripper specified"):
            _ = motion._motion_chart

    def test_motion_chart_unsupported_gripper_raises_error(self):
        """ An invalid arm enum or token identifier must bubble up a ValueError. """
        motion = TracyGripMotion(gripper="INVALID_ARM_NAME", motion=GripperState.OPEN)

        with pytest.raises(ValueError, match="Unsupported gripper INVALID_ARM_NAME"):
            _ = motion._motion_chart

    def test_perform_logs_action(self):
        """ Ensure perform() returns cleanly. """
        motion = TracyGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN)
        assert motion.perform() is None