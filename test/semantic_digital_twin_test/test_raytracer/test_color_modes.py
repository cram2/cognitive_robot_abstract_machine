"""
Extension through custom color rendering strategies.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.camera_rendering import (
    ColorRenderingMode,
    RenderedColorFrame,
)
from semantic_digital_twin.testing import ray_test_world

from .test_camera_rendering import camera_intersections

if TYPE_CHECKING:
    import numpy.typing as npt
    from trimesh import Scene

    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
    from semantic_digital_twin.world import World

    from .test_camera_rendering import CameraIntersections


# %% Custom color rendering
@dataclass(frozen=True)
class ForegroundColorRenderingMode(ColorRenderingMode):
    """
    Color foreground pixels green and background pixels black.
    """

    def validate_camera_model(self, camera_model: PinholeCameraModel) -> None:
        pass

    def render(
        self,
        segmentation: npt.NDArray[np.int32],
        camera_model: PinholeCameraModel,
        scene: Scene,
        world: World,
    ) -> RenderedColorFrame:
        color_image = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        color_image[segmentation >= 0] = [0, 255, 0]
        return RenderedColorFrame(color_image=color_image, body_colors=[])


# %% Color mode dispatch
def test_custom_color_mode_controls_rendered_colors(
    camera_intersections: CameraIntersections,
) -> None:
    """
    A new color mode supplies the input image without changes to the raytracer.
    """
    fixture = camera_intersections
    frame = fixture.ray_tracer.render(
        fixture.camera, rgb_mode=ForegroundColorRenderingMode()
    )
    expected = np.zeros(
        (fixture.resolution.width, fixture.resolution.height, 3), dtype=np.uint8
    )
    expected[:2, 0] = [0, 255, 0]
    np.testing.assert_array_equal(frame.color_bgr, expected)
