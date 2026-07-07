import pytest
from unittest.mock import MagicMock, patch
from control_msgs.action import ParallelGripperCommand
from coraplex.datastructures.enums import ExecutionType, Arms
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

    @pytest.fixture
    def mock_grip_motion(self):
        """
        Helper factory fixture to construct TiagoGripMotion and mock out its fields
        manually, matching constructor signature requirements.
        """

        def _create_motion(gripper, target_position=0.0):
            # Satisfy base constructor requirement by passing mock positional arguments
            motion = TiagoGripMotion(motion=MagicMock(), gripper=gripper)

            # Explicitly bind fields that the property method reads for validation
            motion.gripper = gripper
            motion.target_position = target_position
            return motion

        return _create_motion

    def test_motion_chart_left_arm(self, mock_grip_motion):
        """
        Verify Left Arm routing configures the proper topic and task.
        """
        motion = mock_grip_motion(gripper=Arms.LEFT, target_position=0.045)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert (
            chart.action_topic == "/left_gripper/robotiq_gripper_controller/gripper_cmd"
        )
        assert chart.message_type == ParallelGripperCommand
        assert motion.target_position == 0.045

    def test_motion_chart_right_arm(self, mock_grip_motion):
        """
        Verify Right Arm routing configures the proper topic and task.
        """
        motion = mock_grip_motion(gripper=Arms.RIGHT, target_position=0.120)
        chart = motion._motion_chart

        assert isinstance(chart, RobotiqGripperActionServerTask)
        assert (
            chart.action_topic
            == "/right_gripper/robotiq_gripper_controller/gripper_cmd"
        )
        assert chart.message_type == ParallelGripperCommand
        assert motion.target_position == 0.120

    def test_motion_chart_missing_gripper_raises_error(self, mock_grip_motion):
        """
        An unassigned gripper token must bubble up a ValueError.
        """
        motion = mock_grip_motion(gripper=None, target_position=0.05)

        with pytest.raises(ValueError, match="No gripper specified"):
            _ = motion._motion_chart

    def test_motion_chart_unsupported_gripper_raises_error(self, mock_grip_motion):
        """
        An invalid arm identifier must bubble up a ValueError.
        """
        motion = mock_grip_motion(gripper="INVALID_ARM_NAME", target_position=0.05)

        with pytest.raises(ValueError, match="Unsupported gripper INVALID_ARM_NAME"):
            _ = motion._motion_chart

    def test_perform_logs_action(self, mock_grip_motion):
        """
        Ensure perform() executes without throwing unexpected side-effects.
        """
        motion = mock_grip_motion(gripper=Arms.LEFT, target_position=0.0)
        assert motion.perform() is None
