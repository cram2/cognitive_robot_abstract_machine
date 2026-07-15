import pytest
from unittest.mock import MagicMock, patch
from control_msgs.action import ParallelGripperCommand
from coraplex.datastructures.enums import ExecutionType, Arms
from semantic_digital_twin.datastructures.definitions import GripperState
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    RobotiqGripperActionServerTask,
)

from coraplex.alternative_motion_mappings.tiago_motion_mapping import (
    TiagoGripMotion,
    TiagoMoveSim,
)


class TestTiagoMoveSim:
    """
    Unit tests for TiagoMoveSim base motion wrapper.
    """

    def test_motion_chart_generation(self):
        """
        Verify that _motion_chart transforms the target coordinates properly,
        zeroes out the Z axis, and builds a DifferentialDriveBaseGoal.
        """
        mock_world = MagicMock()
        mock_target_pose = MagicMock()

        # Mock spatial transform logic to yield a manageable position object
        mock_transformed_pose = MagicMock()
        mock_transformed_pose.z = 1.5  # Give it a non-zero height initially
        mock_world.transform.return_value = mock_transformed_pose
        mock_world.root = "world_root"

        # Initialize with required target parameter
        motion = TiagoMoveSim(target=mock_target_pose)

        # Use patch.object to safely override the read-only property 'world'
        with patch.object(TiagoMoveSim, "world", new=mock_world):
            chart = motion._motion_chart

        # Verify coordinates are cast correctly onto the 2D navigation plane
        mock_world.transform.assert_called_once_with(mock_target_pose, "world_root")
        assert mock_transformed_pose.z == 0

        # Verify the wrapper outputs the correct target primitive type
        assert isinstance(chart, DifferentialDriveBaseGoal)
        assert chart.goal_pose == mock_transformed_pose
        assert chart.threshold == 0.01


class TestTiagoGripMotion:
    """
    Unit tests for TiagoGripMotion plan configuration wrapper.
    """

    def test_mutual_exclusivity_validation(self):
        """
        Verify instantiation rules reject missing or overloaded parameters.
        """
        # Neither parameter provided
        with pytest.raises(ValueError, match="You must specify either 'motion' or 'position'."):
            TiagoGripMotion(gripper=Arms.LEFT)

        # Both parameters provided
        with pytest.raises(ValueError, match="Cannot specify both 'motion' and 'position' at the same time."):
            TiagoGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN, position=0.05)

    def test_motion_chart_left_arm_semantic_open(self):
        """
        Verify Left Arm discrete open state maps onto the proper topic and target metrics.
        """
        motion = TiagoGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert chart.action_topic == "/left_gripper/robotiq_gripper_controller/gripper_cmd"
        assert chart.message_type == ParallelGripperCommand
        assert chart.target_position == 0.0

    def test_motion_chart_right_arm_semantic_close(self):
        """
        Verify Right Arm discrete close state maps onto the proper topic and target metrics.
        """
        motion = TiagoGripMotion(gripper=Arms.RIGHT, motion=GripperState.CLOSE)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert chart.action_topic == "/right_gripper/robotiq_gripper_controller/gripper_cmd"
        assert chart.message_type == ParallelGripperCommand
        assert chart.target_position == 0.7

    def test_motion_chart_custom_position(self):
        """
        Verify that passing an explicit position overrides the semantic mappings.
        """
        custom_position = 0.35
        motion = TiagoGripMotion(gripper=Arms.LEFT, position=custom_position)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert chart.target_position == custom_position

    def test_motion_chart_missing_gripper_raises_error(self):
        """
        An unassigned gripper token must bubble up a ValueError.
        """
        # Explicitly patch or pass None to verify enforcement inside the property block
        motion = TiagoGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN)
        motion.gripper = None

        with pytest.raises(ValueError, match="No gripper specified"):
            _ = motion._motion_chart

    def test_motion_chart_unsupported_gripper_raises_error(self):
        """
        An invalid arm identifier must bubble up a ValueError.
        """
        motion = TiagoGripMotion(gripper="INVALID_ARM_NAME", motion=GripperState.OPEN)

        with pytest.raises(ValueError, match="Unsupported gripper INVALID_ARM_NAME"):
            _ = motion._motion_chart

    def test_perform_logs_action(self):
        """
        Ensure perform() executes without throwing unexpected side-effects.
        """
        motion = TiagoGripMotion(gripper=Arms.LEFT, motion=GripperState.OPEN)
        assert motion.perform() is None