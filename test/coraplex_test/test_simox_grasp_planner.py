from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from geometry_msgs.msg import Point, Pose as RosPose, Quaternion

from coraplex.datastructures.enums import Arms, SimoxApproachDirection
from coraplex.external_interfaces.simox_grasp_planner import (
    GraspPose,
    _classify_approach,
    _detect_stl_unit,
    _ensure_simox_object_xml,
    _scale_stl,
    _simox_pose_to_coraplex_tool_pose,
    plan_grasps_for_body,
)
from coraplex.tf_transformations import quaternion_from_matrix

# %% Helper to create dummy binary STL files


def _write_dummy_stl(file_path: Path, max_val: float) -> None:
    """
    Write a minimal valid binary STL with 1 triangle up to max_val.
    """
    header = b"\x00" * 80
    num_triangles = 1
    with open(file_path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", num_triangles))
        # Normal
        f.write(struct.pack("<3f", 0.0, 0.0, 1.0))
        # 3 Vertices
        f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
        f.write(struct.pack("<3f", max_val, 0.0, 0.0))
        f.write(struct.pack("<3f", 0.0, max_val, 0.0))
        # Attribute byte count
        f.write(struct.pack("<H", 0))


# %% Approach Classification Tests


def test_classify_approach_top():
    """
    Simox forward vector pointing downwards in robot base frame (-Z).
    """
    # R_simox with +Z pointing towards -Z_robot:
    # Column 2 = [0, 0, -1]
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    q = quaternion_from_matrix(T)
    ros_q = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    direction = _classify_approach(ros_q)
    assert direction == SimoxApproachDirection.TOP


def test_classify_approach_bottom_skipped():
    """
    Simox forward vector pointing upwards in robot base frame (+Z).
    """
    # Column 2 = [0, 0, 1]
    R = np.eye(3)
    T = np.eye(4)
    T[:3, :3] = R
    q = quaternion_from_matrix(T)
    ros_q = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    direction = _classify_approach(ros_q)
    assert direction == SimoxApproachDirection.SKIPPED


def test_classify_approach_front():
    """
    Simox forward vector pointing along +X (robot forward).
    """
    # Column 2 = [1, 0, 0]
    R = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    q = quaternion_from_matrix(T)
    ros_q = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    direction = _classify_approach(ros_q)
    assert direction == SimoxApproachDirection.FRONT


def test_classify_approach_back():
    """
    Simox forward vector pointing along -X (robot backward).
    """
    # Column 2 = [-1, 0, 0]
    R = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    q = quaternion_from_matrix(T)
    ros_q = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    direction = _classify_approach(ros_q)
    assert direction == SimoxApproachDirection.BACK


def test_classify_approach_left_and_right():
    """
    Simox forward vector pointing along -Y (LEFT) and +Y (RIGHT).
    """
    # Column 2 = [0, -1, 0] -> gripper is on left, pointing rightwards (-Y)
    R_left = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R_left
    q_left = quaternion_from_matrix(T)
    ros_q_left = Quaternion(
        x=float(q_left[0]),
        y=float(q_left[1]),
        z=float(q_left[2]),
        w=float(q_left[3]),
    )
    assert _classify_approach(ros_q_left) == SimoxApproachDirection.LEFT

    # Column 2 = [0, 1, 0] -> gripper is on right, pointing leftwards (+Y)
    R_right = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    T[:3, :3] = R_right
    q_right = quaternion_from_matrix(T)
    ros_q_right = Quaternion(
        x=float(q_right[0]),
        y=float(q_right[1]),
        z=float(q_right[2]),
        w=float(q_right[3]),
    )
    assert _classify_approach(ros_q_right) == SimoxApproachDirection.RIGHT


# %% Frame Transformation Tests


def test_simox_pose_to_coraplex_tool_pose():
    """
    Verify Simox TCP to CoraPlex tool center point conversion.

    - Simox +Z (approach) offset by +0.05m
    - -90 deg pitch around Y
    """
    ros_pose = RosPose(
        position=Point(x=1.0, y=2.0, z=3.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    coraplex_pose = _simox_pose_to_coraplex_tool_pose(ros_pose)

    # In identity orientation, Simox +Z is [0, 0, 1], so +0.05m along Z:
    assert pytest.approx(coraplex_pose.position.x, abs=1e-5) == 1.0
    assert pytest.approx(coraplex_pose.position.y, abs=1e-5) == 2.0
    assert pytest.approx(coraplex_pose.position.z, abs=1e-5) == 3.05

    # Check orientation: Simox identity @ pitch(-90 deg)
    # Pitch -90 deg around Y: quat = [0, -sin(45 deg), 0, cos(45 deg)] = [0, -0.7071, 0, 0.7071]
    expected_y = -np.sin(np.pi / 4)
    expected_w = np.cos(np.pi / 4)
    assert pytest.approx(coraplex_pose.orientation.y, abs=1e-4) == expected_y
    assert pytest.approx(coraplex_pose.orientation.w, abs=1e-4) == expected_w


# %% STL Scaling & XML Generation Tests


def test_detect_stl_unit_and_scaling(tmp_path: Path):
    """
    Test STL unit detection (meters vs mm) and scale conversion.
    """
    meter_stl = tmp_path / "test_meter.stl"
    _write_dummy_stl(meter_stl, max_val=0.15)  # 15 cm = 0.15 m
    assert _detect_stl_unit(meter_stl) == "meters"

    mm_stl = tmp_path / "test_mm.stl"
    _scale_stl(meter_stl, mm_stl, scale=1000.0)
    assert mm_stl.exists()
    assert _detect_stl_unit(mm_stl) == "millimeters"


def test_ensure_manipulation_object_xml(tmp_path: Path):
    """
    Test ManipulationObject XML creation for Simox.
    """
    mesh_path = tmp_path / "sample_mesh.stl"
    _write_dummy_stl(mesh_path, max_val=0.10)

    mock_body = MagicMock()
    mock_body.name = "sample_mesh.stl"
    mock_body.mesh_path = str(mesh_path)

    xml_path = _ensure_simox_object_xml(
        body=mock_body,
        end_effector_name="r_gripper",
        object_dir=tmp_path,
    )

    assert xml_path is not None
    assert xml_path.exists()
    content = xml_path.read_text()
    assert '<ManipulationObject name="sample_mesh.stl">' in content
    assert '<File type="stl">' in content
    assert str(mesh_path.name) in content


# %% Service Mocking & Planning Tests


def test_plan_grasps_for_body_mocked(tmp_path: Path):
    """
    Test plan_grasps_for_body with a mocked ROS 2 service client.
    """
    # 1. Create a dummy STL mesh file
    mesh_file = tmp_path / "cereal_box.stl"
    _write_dummy_stl(mesh_file, max_val=0.20)

    # 2. Mock Body object and World
    mock_body = MagicMock()
    mock_body.name = "cereal_box.stl"
    mock_body.mesh_path = str(mesh_file)
    mock_body.global_pose = MagicMock()

    mock_world = MagicMock()
    mock_world.root = MagicMock()
    mock_world.root.name = "world"
    mock_world.bodies = []
    mock_world.transform.return_value = mock_body.global_pose

    # 3. Create mock ROS service response with 2 poses:
    # Pose 1: top approach
    R_top = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    T_top = np.eye(4)
    T_top[:3, :3] = R_top
    q_top = quaternion_from_matrix(T_top)

    # Pose 2: front approach
    R_front = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    T_front = np.eye(4)
    T_front[:3, :3] = R_front
    q_front = quaternion_from_matrix(T_front)

    mock_resp = MagicMock()
    mock_resp.success = True
    mock_resp.error_message = ""
    mock_resp.grasp_poses = [
        RosPose(
            position=Point(x=1.0, y=0.0, z=0.8),
            orientation=Quaternion(
                x=float(q_top[0]),
                y=float(q_top[1]),
                z=float(q_top[2]),
                w=float(q_top[3]),
            ),
        ),
        RosPose(
            position=Point(x=1.0, y=0.0, z=0.8),
            orientation=Quaternion(
                x=float(q_front[0]),
                y=float(q_front[1]),
                z=float(q_front[2]),
                w=float(q_front[3]),
            ),
        ),
    ]
    mock_resp.qualities = [0.85, 0.70]

    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_client.call.return_value = mock_resp

    mock_srv_module = MagicMock()
    mock_srv_module.PlanGrasp = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "grasp_planner_msgs": MagicMock(),
            "grasp_planner_msgs.srv": mock_srv_module,
        },
    ):
        with patch(
            "coraplex.external_interfaces.simox_grasp_planner.get_simox_client",
            return_value=mock_client,
        ):
            with patch(
                "coraplex.external_interfaces.simox_grasp_planner._ensure_simox_object_xml",
                return_value=tmp_path / "cereal_box.xml",
            ):
                with patch(
                    "coraplex.external_interfaces.simox_grasp_planner._fill_ros_pose"
                ):
                    result = plan_grasps_for_body(
                        body=mock_body,
                        arm=Arms.RIGHT,
                        end_effector_name="r_gripper",
                        kinematic_chain_name="RightArm",
                        world=mock_world,
                    )

                assert SimoxApproachDirection.TOP in result
                assert SimoxApproachDirection.FRONT in result
                assert len(result[SimoxApproachDirection.TOP]) == 1
                assert isinstance(result[SimoxApproachDirection.TOP][0], GraspPose)
                assert len(result[SimoxApproachDirection.FRONT]) == 1
                assert isinstance(result[SimoxApproachDirection.FRONT][0], GraspPose)
