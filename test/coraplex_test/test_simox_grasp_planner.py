"""
Service message and geometry contracts for Simox integration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch
from xml.etree.ElementTree import parse

import numpy as np
import pytest
import trimesh
from geometry_msgs.msg import Pose as RosPose
from rclpy.task import Future

from coraplex.external_interfaces.exceptions import (
    InvalidSimoxResponse,
    SimoxPlanningFailed,
)
from coraplex.external_interfaces.simox_grasp_planner import (
    RosSimoxService,
    SimoxConfiguration,
    SimoxGraspPlanner,
    SimoxRequest,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    PoseToSemDTConverter,
)
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)

# %% pose conversion


@pytest.mark.parametrize(
    "quaternion", [(0, 0, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)]
)
def test_ros_pose_preserves_metres_and_rotation(
    quaternion: tuple[int, int, int, int],
) -> None:
    message = RosPose()
    message.position.x, message.position.y, message.position.z = 0.6, -0.2, 0.9
    (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) = map(float, quaternion)
    pose = RosSimoxService.pose_from_message(message)
    np.testing.assert_allclose(pose.position.to_np()[:3], [0.6, -0.2, 0.9])
    expected = np.array(quaternion) / np.linalg.norm(quaternion)
    np.testing.assert_allclose(pose.orientation.to_np(), expected)


def test_zero_quaternion_is_rejected() -> None:
    message = RosPose()
    message.orientation.x = 0.0
    message.orientation.y = 0.0
    message.orientation.z = 0.0
    message.orientation.w = 0.0
    with pytest.raises(InvalidSimoxResponse):
        RosSimoxService.pose_from_message(message)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf")])
def test_nonfinite_ros_position_is_rejected(coordinate: float) -> None:
    """
    Malformed translations are rejected at the service boundary.
    """
    message = RosPose()
    message.position.x = coordinate
    message.orientation.w = 1.0
    with pytest.raises(InvalidSimoxResponse):
        RosSimoxService.pose_from_message(message)


def test_service_uses_shared_converters_and_explicit_robot_configuration() -> None:
    """
    Transport reuses standard conversions and forwards the supplied robot model.
    """
    configuration = SimoxConfiguration(
        robot_model_path=Path("custom_robot.xml"),
        end_effector_name="custom_hand",
        simox_T_tool=HomogeneousTransformationMatrix(),
        preshape_name="custom_open",
        kinematic_chain_name="custom_arm",
    )
    expected_pose = Pose.from_xyz_rpy(0.6, -0.2, 0.9, roll=0.3, pitch=-0.7, yaw=1.1)
    request = SimoxRequest(configuration, Path("custom_object.xml"), expected_pose)
    response_message = PoseToRos2Converter.convert(expected_pose)
    response = Mock(
        success=True,
        grasp_poses=[response_message],
        qualities=[0.7],
        are_force_closure=[True],
    )
    future = Future()
    future.set_result(response)
    node = Mock()
    client = node.create_client.return_value
    client.wait_for_service.return_value = True
    client.call_async.return_value = future
    service = RosSimoxService(node)
    with (
        patch("coraplex.external_interfaces.simox_grasp_planner.import_module") as load,
        patch.object(
            PoseToRos2Converter, "convert", wraps=PoseToRos2Converter.convert
        ) as outgoing,
        patch.object(
            PoseToSemDTConverter, "convert", wraps=PoseToSemDTConverter.convert
        ) as incoming,
    ):
        load.return_value.PlanGrasp.Request.return_value.object_pose = RosPose()
        result = service.plan(request)

    sent = client.call_async.call_args.args[0]
    assert sent.robot_model_path == str(configuration.robot_model_path)
    assert sent.object_model_path == str(request.object_model_path)
    assert sent.end_effector_name == configuration.end_effector_name
    assert sent.kinematic_chain_name == configuration.kinematic_chain_name
    assert sent.preshape_name == configuration.preshape_name
    assert sent.object_pose == response_message
    outgoing.assert_called_once_with(request.object_pose)
    assert incoming.call_count == 1
    assert result.grasp_poses[0].to_np() == pytest.approx(expected_pose.to_np())
    assert result.qualities == response.qualities
    assert result.are_force_closure == response.are_force_closure
    node.destroy_client.assert_called_once_with(client)


# %% placed geometry


def test_export_preserves_object_mesh_scale(
    pr2_apartment_context: tuple, tmp_path: Path
) -> None:
    world, _, _ = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    original = milk.grasp_surface()
    xml_path = SimoxGraspPlanner.export_object(milk, tmp_path)
    mesh_name = parse(xml_path).find("CollisionModel/File").text
    loaded = trimesh.load_mesh(tmp_path / mesh_name)
    np.testing.assert_allclose(loaded.bounds, original.bounds, atol=1e-7)


# %% ROS failures


@pytest.mark.parametrize("available", [False, True])
def test_service_unavailable_or_timeout_cleans_up(available: bool) -> None:
    node = Mock()
    client = node.create_client.return_value
    client.wait_for_service.return_value = available
    future = Future()
    client.call_async.return_value = future
    service = RosSimoxService(node, availability_timeout=0, response_grace_period=0)
    configuration = SimoxConfiguration.for_pr2(Path("robot.xml"))
    configuration.timeout_ms = 0
    request = SimoxRequest(configuration, Path("object.xml"), Pose())
    with patch("coraplex.external_interfaces.simox_grasp_planner.import_module"):
        with pytest.raises(SimoxPlanningFailed):
            service.plan(request)
    node.destroy_client.assert_called_once_with(client)
    if available:
        assert future.cancelled()
