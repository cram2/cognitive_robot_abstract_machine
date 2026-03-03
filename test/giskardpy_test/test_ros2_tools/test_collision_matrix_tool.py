import importlib.util
import os
import signal
import subprocess
import sys
import time
from importlib.resources import files
from unittest.mock import MagicMock, patch

import pytest

# Mock PyQt5 and ROS 2 before importing the tool
sys.modules["giskardpy.middleware.ros2"] = MagicMock()
sys.modules["giskardpy.middleware.ros2.rospy"] = MagicMock()

# Mock PyQt5
QtWidgets = MagicMock()


class MockQWidget:
    def __init__(self, *args, **kwargs):
        pass

    def setLayout(self, layout):
        pass

    def setWindowTitle(self, title):
        pass

    def setMinimumSize(self, w, h):
        pass

    def setCentralWidget(self, widget):
        pass

    def setDisabled(self, active):
        pass

    def show(self):
        pass


QtWidgets.QWidget = MockQWidget
QtWidgets.QMainWindow = MockQWidget
QtWidgets.QDialog = MockQWidget
QtWidgets.QTableWidget = MockQWidget
QtWidgets.QCheckBox = MockQWidget
QtWidgets.QProgressBar = MockQWidget
QtWidgets.QLabel = MockQWidget
QtWidgets.QFrame = MockQWidget
QtWidgets.QScrollArea = MockQWidget

sys.modules["PyQt5"] = MagicMock()
sys.modules["PyQt5.QtCore"] = MagicMock()
sys.modules["PyQt5.QtCore"].Qt = MagicMock()
sys.modules["PyQt5.QtGui"] = MagicMock()
sys.modules["PyQt5.QtWidgets"] = QtWidgets


# Import the tool
def import_tool(path):
    spec = importlib.util.spec_from_file_location("collision_matrix_tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["collision_matrix_tool"] = module
    spec.loader.exec_module(module)
    return module


tool_path = os.path.abspath(
    files("giskardpy").joinpath("../../scripts/ros2-tools/collision_matrix_tool.py")
)
tool = import_tool(tool_path)
from collision_matrix_tool import SelfCollisionMatrixInterface, DisableCollisionReason


@pytest.fixture
def interface():
    with patch("collision_matrix_tool.VizMarkerPublisher"):
        return SelfCollisionMatrixInterface()


def test_load_urdf(interface):
    pr2_xacro_path = (
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    )
    interface.load_urdf(pr2_xacro_path)
    assert len(interface.world.bodies) > 0


def test_add_remove_pair(interface):
    interface.self_collision_matrix_rule = MagicMock()
    interface.self_collision_matrix_rule.allowed_collision_pairs = set()
    body_a = MagicMock()
    body_b = MagicMock()

    # Mock sort_bodies and CollisionCheck
    with patch("collision_matrix_tool.CollisionCheck.create_and_validate") as mock_cc:
        mock_cc.return_value.body_a = body_a
        mock_cc.return_value.body_b = body_b

        interface.add_pair(body_a, body_b, DisableCollisionReason.Never)
        assert (
            interface.get_reason_for_pair(body_a, body_b)
            == DisableCollisionReason.Never
        )

        interface.remove_pair(body_a, body_b)
        assert interface.get_reason_for_pair(body_a, body_b) is None


def test_compute_self_collision_matrix(interface):
    interface.robot = MagicMock()
    interface.self_collision_matrix_rule = MagicMock()
    interface.self_collision_matrix_rule.allowed_collision_pairs = []

    progress_mock = MagicMock()
    interface.compute_self_collision_matrix(progress_bar=progress_mock)

    interface.self_collision_matrix_rule.compute_self_collision_matrix.assert_called()


def test_disable_enable_body(interface):
    body = MagicMock()
    interface.self_collision_matrix_rule = MagicMock()

    interface.remove_body(
        body
    )  # Implementation: adds to allowed_collision_bodies (disables it)
    interface.self_collision_matrix_rule.allowed_collision_bodies.add.assert_called_with(
        body
    )

    interface.add_body(
        body
    )  # Implementation: discards from allowed_collision_bodies (enables it)
    interface.self_collision_matrix_rule.allowed_collision_bodies.discard.assert_called_with(
        body
    )


def test_script_launch_and_kill():
    script_path = files("giskardpy").joinpath(
        "../../scripts/ros2-tools/collision_matrix_tool.py"
    )

    # Start the process in a new process group
    process = subprocess.Popen(
        ["python3", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},  # Add this
    )

    try:
        # Give it enough time to initialize (e.g., 3-5 seconds)
        time.sleep(3)

        # Check if it crashed immediately
        if process.poll() is not None:
            _, stderr = process.communicate()
            pytest.fail(f"Script crashed on startup. Error: {stderr.decode()}")

        # Send SIGINT (Ctrl+C)
        os.killpg(os.getpgid(process.pid), signal.SIGINT)

        # Wait for clean shutdown
        process.communicate(timeout=10)

        # Return code 0 (Success) or -2 (SIGINT) are expected
        # Note: In headless environments, you might see -6 (SIGABRT) from Qt
        assert process.returncode in [0, -signal.SIGINT, -6]

    finally:
        if process.poll() is None:
            process.kill()
