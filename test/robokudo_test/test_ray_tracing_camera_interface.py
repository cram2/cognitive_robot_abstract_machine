from __future__ import annotations

# %% Imports
from unittest.mock import Mock
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import MonkeyPatch

import numpy as np
from sensor_msgs.msg import CameraInfo

from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
    WorldDescriptorSource,
)
from robokudo.io.ray_tracing_camera_interface import (
    RayTracingCameraInterface,
)
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.camera_rendering import (
    RenderedRGBDFrame,
    ColorRenderingMode,
)

# %% Interface integration


def test_interface_publishes_aligned_semantic_rgbd_frame():
    """
    The complete interface publishes images and matching semantic camera state.
    """
    interface = RayTracingCameraInterface(
        RayTracingCameraConfiguration(
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_cylinders"
            )
        )
    )
    cas = CAS()

    interface.set_data(cas)

    assert cas.color_image.shape[:2] == cas.depth_image.shape
    assert cas.get(CASViews.OBJECT_IMAGE).shape == cas.depth_image.shape
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())
    context = interface.context_resolver.resolve()
    assert cas.camera_observation.camera is context.camera
    assert cas.ground_truth_world_ref is context.world


def test_interface_publishes_effective_model_for_field_of_view_camera():
    """
    An FOV camera produces a calibrated effective observation model.
    """
    interface = RayTracingCameraInterface(
        RayTracingCameraConfiguration(
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_cylinders"
            )
        )
    )
    context = interface.context_resolver.resolve()
    native_field_of_view = context.camera.field_of_view
    resolution = CameraResolution(width=32, height=24)
    context.camera.camera_model = FieldOfViewCameraModel(
        angular_extent=native_field_of_view,
        image_resolution=resolution,
    )
    cas = CAS()

    interface.set_data(cas)

    assert cas.camera_observation.camera is context.camera
    assert cas.camera_observation.effective_camera_model.resolution is resolution
    assert np.isclose(
        cas.camera_observation.field_of_view.horizontal_angle,
        native_field_of_view.horizontal_angle,
    )
    assert np.isclose(
        cas.camera_observation.field_of_view.vertical_angle,
        native_field_of_view.vertical_angle,
    )
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())


# %% Shared renderer integration


def test_interface_publishes_shared_rendered_frame(monkeypatch: MonkeyPatch) -> None:
    """
    The camera interface publishes row-first images from a shared width-first frame.
    """
    interface = RayTracingCameraInterface(
        RayTracingCameraConfiguration(
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_cylinders"
            )
        )
    )
    context = interface.context_resolver.resolve()
    frame = RenderedRGBDFrame(
        color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_in_millimeters=np.full((2, 3), 1000, dtype=np.uint16),
        segmentation=np.full((2, 3), context.camera.root.index, dtype=np.int32),
        object_color_map={},
        camera_model=context.camera.camera_model,
    )
    render = Mock(return_value=frame)
    monkeypatch.setattr(context.world.ray_tracer, "render", render)
    cas = CAS()

    interface.set_data(cas)

    render.assert_called_once_with(
        context.camera, rgb_mode=interface.camera_config.rgb_mode
    )
    assert interface.camera_config.rgb_mode is ColorRenderingMode.SEMANTIC
    np.testing.assert_array_equal(cas.color_image, frame.color_bgr.swapaxes(0, 1))
    np.testing.assert_array_equal(
        cas.depth_image, frame.depth_in_millimeters.swapaxes(0, 1)
    )
    np.testing.assert_array_equal(
        cas.get(CASViews.OBJECT_IMAGE), frame.segmentation.swapaxes(0, 1)
    )
    assert cas.get(CASViews.OBJECT_COLOR_MAP) == frame.object_color_map
    assert cas.camera_observation.effective_camera_model is frame.camera_model
