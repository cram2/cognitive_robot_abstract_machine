"""
Static camera pose binding preserves semantic frame names.
"""

from __future__ import annotations

# %% Imports
import pytest
from py_trees.common import Status
from sensor_msgs.msg import CameraInfo

from robokudo.annotators.static_camera_transform import StaticCameraTransformAnnotator
from robokudo.cas import CAS
from robokudo.io.camera_model_adapters import RosCameraModelAdapter
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from .test_camera_interface_observation import camera_info, runtime_world


# %% Namespaced camera frames
@pytest.mark.parametrize("camera_prefix", [None, "robot"])
def test_static_transform_binds_observation_camera_frame(
    camera_prefix: str | None,
    runtime_world: World,
    camera_info: CameraInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Static poses target the observed camera even when local names collide.
    """
    camera_body = Body(
        name=PrefixedName(name=camera_info.header.frame_id, prefix=camera_prefix)
    )
    other_camera_body = Body(
        name=PrefixedName(name=camera_info.header.frame_id, prefix="other_robot")
    )
    reference_body = Body(name=PrefixedName(name="map"))
    with runtime_world.modify_world():
        runtime_world.add_body(reference_body)
        for body in (camera_body, other_camera_body):
            runtime_world.add_body(body)
            runtime_world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=reference_body,
                    child=body,
                    world=runtime_world,
                )
            )
    camera_model = RosCameraModelAdapter.from_camera_info(camera_info)
    cas = CAS()
    cas.camera_observation = CameraObservation(
        camera=Camera(
            name=camera_body.name,
            root=camera_body,
            camera_model=camera_model,
        ),
        effective_camera_model=camera_model,
        world_T_camera=None,
        timestamp_nanoseconds=123,
    )
    annotator = StaticCameraTransformAnnotator()
    monkeypatch.setattr(annotator, "get_cas", lambda: cas)

    assert annotator.update() is Status.SUCCESS
    assert (
        cas.require_camera_observation().world_T_camera_or_raise().child_frame
        is camera_body
    )
