"""
Output management for RoboKudo annotators.

This module provides classes for managing annotator outputs across pipelines.
It supports:

* Per-pipeline output organization
* Structured output storage
* Output type management
* Pipeline-specific output mapping
* Dynamic output initialization

The module is used for:

* Output data organization
* Pipeline result management
* Cross-pipeline data handling
* Output type safety
"""

from __future__ import annotations


import numpy as np
from typing_extensions import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import numpy.typing as npt
    import open3d as o3d


class AnnotatorOutputStruct:
    """
    Container for annotator-specific outputs. These will get consumed by the GUI after the pipeline is run,
    or you have used the SetPipelineRedraw Behaviour. Check out the GUI-related classes for more details.

    This class stores and manages different types of outputs
    (e.g., images, point clouds) for a single annotator.
    """

    def __init__(self) -> None:
        """Initialize an empty output structure."""
        self.image: npt.NDArray[np.uint8] = np.zeros((640, 480, 3), dtype="uint8")
        """Image output data"""

        self.geometries: List[Dict[str, o3d.geometry.Geometry3D]] = []
        """Open3D Geometries data. Will be passed to o3d.visualization.O3DVisualizer.add_geometry."""

        self.render_next_time = True
        """Whether the outputs should be rendered on the next GUI update. Defaults to True for the first run."""

    def set_image(self, img: npt.NDArray[np.uint8]) -> None:
        """Set the image in this AnnotatorOutputStruct and instruct the GUI to render it next time.

        :param img: The image to be displayed on the next GUI update.
        """
        self.image = img
        self.render_next_time = True

    def set_geometries(self, geometries: List[Dict[str, Any]]) -> None:
        """Set the geometries in this AnnotatorOutputStruct and instruct the GUI to render it next time.

        :param geometries: This parameter holds the geometries to be drawn. It should behave like o3d.visualization.draw,
        which means that you can either pass a drawable geometry, a dict with a drawable geometry
        (see
        https://github.com/isl-org/Open3D/blob/73bbddc8851b1670b7e74b7cf7af969360f48317/examples/python/visualization/draw.py#L123
        for an example) or a list of both.
        """
        self.geometries = geometries
        self.render_next_time = True


class AnnotatorOutputs:
    """Container for all annotator outputs in a pipeline.

    This class manages output structures for multiple annotators
    within a single pipeline.
    """

    def __init__(self) -> None:
        """Initialize an empty outputs container."""

        self.outputs: Dict[str, AnnotatorOutputStruct] = {}
        """Dictionary mapping annotator names to their outputs"""

        self.redraw = True
        """Whether the outputs should be redrawn on the next GUI update."""

    def init_annotator(self, annotator_name: str) -> None:
        """Initialize the output structure for an annotator.

        :param annotator_name: Name of the annotator
        """
        self.outputs[annotator_name] = AnnotatorOutputStruct()

    def clear_outputs(self) -> None:
        annotator_names = [key for key in self.outputs]
        for name in annotator_names:
            self.init_annotator(name)


class AnnotatorOutputPerPipelineMap:
    """Container for annotator outputs across multiple pipelines.

    This class manages output structures for all annotators across
    all pipelines in the system.
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline map."""

        self.map: Dict[str, AnnotatorOutputs] = {}
        """Dictionary mapping pipeline names to their annotator outputs"""
