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
import logging

import numpy as np
import py_trees
from py_trees.behaviour import Behaviour

import robokudo.defs
import robokudo.pipeline
import robokudo.utils.tree


class AnnotatorOutputStruct:
    """
    Container for annotator-specific outputs. These will get consumed by the GUI after the pipeline is run,
    or you have used the SetPipelineRedraw Behaviour. Check out the GUI-related classes for more details.

    This class stores and manages different types of outputs
    (e.g., images, point clouds) for a single annotator.

    :ivar image: Image output data
    :type image: numpy.ndarray
    :ivar geometries: Open3D Geometries data. Will be passed to o3d.visualization.O3DVisualizer.add_geometry.
    """

    def __init__(self):
        """
        Initialize an empty output structure.
        """
        self.image = np.zeros((640, 480, 3), dtype="uint8")
        self.geometries = None
        self.render_next_time = True  # render defaults on the first run

    def set_image(self, img):
        """
        Set the image in this AnnotatorOutputStruct and instruct the GUI to render it next time.
        """
        self.image = img
        self.render_next_time = True

    def set_geometries(self, geometries):
        """
        Set the geometries in this AnnotatorOutputStruct and instruct the GUI to render it next time.

        :param geometries: This parameter holds the geometries to be drawn. It should behave like o3d.visualization.draw,
        which means that you can either pass a drawable geometry, a dict with a drawable geometry
        (see
        https://github.com/isl-org/Open3D/blob/73bbddc8851b1670b7e74b7cf7af969360f48317/examples/python/visualization/draw.py#L123
        for an example) or a list of both.
        """
        self.geometries = geometries
        self.render_next_time = True


class AnnotatorOutputs:
    """
    Container for all annotator outputs in a pipeline.

    This class manages output structures for multiple annotators
    within a single pipeline.

    :ivar outputs: Dictionary mapping annotator names to their outputs
    :type outputs: dict[str, AnnotatorOutputStruct]
    """

    def __init__(self):
        """
        Initialize an empty outputs container.
        """
        self.outputs = {}
        self.redraw = True

    def init_annotator(self, annotator_name):
        """
        Initialize the output structure for an annotator.

        :param annotator_name: Name of the annotator
        :type annotator_name: str
        """
        self.outputs[annotator_name] = AnnotatorOutputStruct()

    def clear_outputs(self):
        annotator_names = [key for key in self.outputs]
        for name in annotator_names:
            self.init_annotator(name)


class AnnotatorOutputPerPipelineMap:
    """
    Container for annotator outputs across multiple pipelines.

    This class manages output structures for all annotators across
    all pipelines in the system.

    :ivar map: Dictionary mapping pipeline names to their annotator outputs
    :type map: dict[str, AnnotatorOutputs]
    """

    def __init__(self):
        """
        Initialize an empty pipeline map.
        """
        self.map = {}  # Key: Pipeline Name, Value: AnnotatorOutputs


class ClearAnnotatorOutputs(Behaviour):
    """
    Put directly in the corresponding RK Pipeline
    """

    def __init__(self, name='ClearAnnotatorOutputs'):
        super().__init__("ClearAnnotatorOutputs")
        self.rk_logger = logging.getLogger(robokudo.defs.PACKAGE_NAME)

    def update(self):
        self.rk_logger.debug("%s.update()" % (self.__class__.__name__))
        blackboard = py_trees.blackboard.Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get("annotator_output_pipeline_map_buffer")
        assert (isinstance(annotator_output_pipeline_map_buffer,
                           robokudo.annotators.outputs.AnnotatorOutputPerPipelineMap))

        pipeline = robokudo.utils.tree.find_parent_of_type(self, robokudo.pipeline.Pipeline)
        annotator_outputs = annotator_output_pipeline_map_buffer.map[pipeline.name]
        assert (isinstance(annotator_outputs, robokudo.annotators.outputs.AnnotatorOutputs))
        annotator_outputs.clear_outputs()

        return py_trees.common.Status.SUCCESS
