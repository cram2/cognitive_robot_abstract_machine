"""
Optical camera metadata and row-first image publication.
"""

from __future__ import annotations

# %% Imports
from unittest.mock import Mock

import numpy as np
import pytest

from robokudo.io.cas_view_codecs import CameraObservationCodec
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from robokudo.cas import CAS, CASViews
from robokudo.types.camera import CameraObservation
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
    WorldDescriptorSource,
)
from robokudo.io.sensor_world import RecordedCameraWorld
from robokudo.io.ray_tracing_camera_interface import RayTracingCameraInterface
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.camera_rendering import RenderedRGBDFrame
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


# %% Rectangular rendered observation
@pytest.fixture
def perception_world() -> World:
    """
    An empty world perception perceives into.
    """
    return World()


@pytest.fixture
def optical_interface(perception_world: World) -> RayTracingCameraInterface:
    """
    Provide a simulated camera with an nonidentity optical root pose.
    """
    interface = RayTracingCameraInterface(
        RayTracingCameraConfiguration(
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_cylinders"
            )
        )
    )
    context = interface.context_resolver.resolve(perception_world=perception_world)
    root = context.camera.root
    with context.world.modify_world():
        root.parent_connection.origin = HomogeneousTransformationMatrix(
            data=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, y=0.2, z=0.3, yaw=np.pi / 2
            ),
            reference_frame=root.parent_connection.parent,
            child_frame=root,
        )
    return interface


@pytest.fixture
def rectangular_frame(
    optical_interface: RayTracingCameraInterface, perception_world: World
) -> RenderedRGBDFrame:
    """
    Provide distinct values at every pixel of a width-first rendered frame.
    """
    camera = optical_interface.context_resolver.resolve(
        perception_world=perception_world
    ).camera
    resolution = CameraResolution(width=4, height=3)
    camera_model = PinholeCameraModel.from_field_of_view(
        resolution=resolution, field_of_view=camera.field_of_view
    )
    pixels = np.arange(resolution.width * resolution.height).reshape(
        resolution.width, resolution.height
    )
    return RenderedRGBDFrame(
        color_bgr=np.repeat(pixels[:, :, None], 3, axis=2).astype(np.uint8),
        depth_in_millimeters=pixels.astype(np.uint16),
        segmentation=pixels.astype(np.int32),
        body_colors=[],
        camera_model=camera_model,
    )


# %% Pixel and pose publication
@pytest.mark.parametrize(
    "view", [CASViews.COLOR_IMAGE, CASViews.DEPTH_IMAGE, CASViews.OBJECT_IMAGE]
)
def test_interface_publishes_row_first_pixels(
    optical_interface: RayTracingCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
    perception_world: World,
    view: str,
) -> None:
    """
    Image publication preserves pixel values while placing height before width.
    """
    context = optical_interface.context_resolver.resolve(
        perception_world=perception_world
    )
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas=cas, world=perception_world)
    images = {
        CASViews.COLOR_IMAGE: rectangular_frame.color_bgr,
        CASViews.DEPTH_IMAGE: rectangular_frame.depth_in_millimeters,
        CASViews.OBJECT_IMAGE: rectangular_frame.segmentation,
    }
    np.testing.assert_array_equal(cas.get(view), images[view].swapaxes(0, 1))


def test_interface_publishes_rendered_optical_pose(
    optical_interface: RayTracingCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
    perception_world: World,
) -> None:
    """
    The published camera pose matches the optical root pose used for rendering.
    """
    context = optical_interface.context_resolver.resolve(
        perception_world=perception_world
    )
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas=cas, world=perception_world)
    np.testing.assert_allclose(
        cas.camera_observation.world_T_camera.to_np(),
        context.camera.root.global_transform.to_np(),
    )


# %% Recorded optical pose


def test_sensor_projection_preserves_optical_root_pose(
    optical_interface: RayTracingCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
    perception_world: World,
) -> None:
    """
    Projecting a rendered observation preserves the optical root pose.
    """
    context = optical_interface.context_resolver.resolve(
        perception_world=perception_world
    )
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas=cas, world=perception_world)
    world = RecordedCameraWorld(cas.camera_observation).project()
    camera = world.get_semantic_annotation_by_id(cas.camera_observation.camera.id)
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        context.camera.root.global_transform.to_np(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        cas.camera_observation.world_T_camera.to_np(),
        atol=1e-12,
    )


def test_replayed_optical_pose_updates_camera_root(
    optical_interface: RayTracingCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
    perception_world: World,
) -> None:
    """
    Replaying an optical pose updates the camera root directly.
    """
    context = optical_interface.context_resolver.resolve(
        perception_world=perception_world
    )
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas=cas, world=perception_world)
    world = RecordedCameraWorld(cas.camera_observation).project()
    camera = world.get_semantic_annotation_by_id(cas.camera_observation.camera.id)
    world_T_optical = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, roll=0.2, reference_frame=world.root
    )
    observation = CameraObservation(
        camera=camera,
        effective_camera_model=rectangular_frame.camera_model,
        world_T_camera=world_T_optical,
        timestamp_nanoseconds=0,
    )
    RecordedCameraWorld.apply_pose(world, observation)
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(), world_T_optical.to_np(), atol=1e-12
    )


# %% Recorded optical observations


def test_optical_observation_roundtrip_preserves_root_pose(
    optical_interface: RayTracingCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
    perception_world: World,
) -> None:
    """
    Recorded optical images retain the sampled root pose for replay.
    """
    context = optical_interface.context_resolver.resolve(
        perception_world=perception_world
    )
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas=cas, world=perception_world)
    codec = CameraObservationCodec()
    restored = codec.decode_with_tracker(
        codec.encode(cas.camera_observation),
        WorldEntityWithIDKwargsTracker.from_world(perception_world),
    )
    world = RecordedCameraWorld(restored).project()
    camera = world.get_semantic_annotation_by_id(cas.camera_observation.camera.id)
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        context.camera.root.global_transform.to_np(),
        atol=1e-12,
    )
