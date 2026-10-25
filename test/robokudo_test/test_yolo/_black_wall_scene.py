"""
A camera facing a black wall, which the YOLO annotators under test are run on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.cv import ColorToDepthRatio

# %% black wall scene


@dataclass
class BlackWallScene:
    """
    A camera facing a black wall square-on: a black color image and a depth image that
    measures the wall's distance at every pixel.
    """

    models_directory: Path
    """
    Directory the annotators under test load pretrained models from.
    """

    resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(width=80, height=40)
    )
    """
    Resolution of the color images in the CAS.
    """

    wall_distance: float = 1.0
    """
    Distance in meters between the camera and the wall, which every depth image pixel
    measures.
    """

    @classproperty
    def depth_image_units_per_meter(cls) -> float:
        """
        Depth image value per meter of measured distance, since RoboKudo stores depth
        images in millimeters.
        """
        return 1000.0

    def color_image(self) -> np.ndarray:
        """
        :return: A black color image of the scene's resolution.
        """
        return np.zeros(
            (self.resolution.height, self.resolution.width, 3), dtype=np.uint8
        )

    def place_in_pipeline(
        self, annotator: BaseAnnotator, color_to_depth_ratio: float = 1.0
    ) -> None:
        """
        Set the annotator up in a pipeline whose CAS holds the scene's color image and a
        matching depth image.

        :param color_to_depth_ratio: Size of the depth image relative to the color
            image.
        """
        annotator.setup()
        pipeline = Pipeline("Sequence")
        pipeline.add_child(annotator)
        pipeline.setup()

        color_height, color_width = self.resolution.height, self.resolution.width
        depth_height = int(color_height * color_to_depth_ratio)
        depth_width = int(color_width * color_to_depth_ratio)
        cas = annotator.get_cas()
        cas.set(CASViews.COLOR_IMAGE, self.color_image())
        cas.set(
            CASViews.DEPTH_IMAGE,
            np.full(
                (depth_height, depth_width),
                round(self.wall_distance * self.depth_image_units_per_meter),
                dtype=np.uint16,
            ),
        )
        cas.set(
            CASViews.COLOR_TO_DEPTH_RATIO,
            ColorToDepthRatio(
                horizontal=color_to_depth_ratio, vertical=color_to_depth_ratio
            ),
        )
        cas.set(
            CASViews.CAMERA_INTRINSIC,
            o3d.camera.PinholeCameraIntrinsic(
                color_width,
                color_height,
                100.0,
                100.0,
                color_width / 2.0,
                color_height / 2.0,
            ),
        )
