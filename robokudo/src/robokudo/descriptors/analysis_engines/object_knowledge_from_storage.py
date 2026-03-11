"""Analysis engine for object detection and knowledge visualization from stored data.

This module provides an analysis engine that demonstrates object detection and
knowledge visualization using stored camera data. It combines static object
detection with various visualization components to display object hypotheses,
knowledge, and color information.

The pipeline implements the following functionality:
- Reading stored camera data from MongoDB
- Image preprocessing
- Static object detection with predefined parameters
- Visualization of object hypotheses
- Visualization of object knowledge
- Color analysis of detected objects

.. note::
    This engine uses predefined object detection parameters optimized for a
    specific use case (mug detection). Adjust the parameters for other objects
    or scenarios.
"""

import robokudo.analysis_engine
from robokudo.annotators.cluster_color import ClusterColorAnnotator

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer
from robokudo.annotators.object_knowledge_visualizer import ObjectKnowledgeVisualizer

import robokudo.descriptors.camera_configs.config_mongodb_playback

import robokudo.io.storage_reader_interface

import robokudo.behaviours.clear_errors

import robokudo.idioms
import robokudo.pipeline
from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for object detection and knowledge visualization.

    This class implements a pipeline that combines static object detection with
    various visualization components. It processes stored camera data to detect
    objects and visualize their properties and associated knowledge.

    The pipeline includes:
    - Collection reader for accessing stored data
    - Image preprocessing
    - Static object detection with predefined parameters
    - Object hypothesis visualization
    - Object knowledge visualization
    - Color analysis and visualization

    .. note::
        The static object detector is configured with specific parameters for
        mug detection. These parameters include bounding box dimensions and
        pose information that should be adjusted for different objects.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "object_knowledge_from_storage"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a pipeline for object detection and knowledge visualization.

        This method constructs a processing pipeline that reads stored camera data,
        performs object detection with predefined parameters, and visualizes the
        results including object hypotheses, knowledge, and color information.

        The static object detector is configured with specific parameters for a mug:
        - Bounding box: 397x126 pixels with size 49x106
        - Position: (0.202, -0.109, 1.096)
        - Rotation: Quaternion (0.575, 0.666, -0.360, 0.310)

        :return: The configured pipeline for object detection and visualization
        :rtype: robokudo.pipeline.Pipeline
        """
        cr_storage_camera_config = (
            robokudo.descriptors.camera_configs.config_mongodb_playback.CameraConfig()
        )
        cr_storage_config = CollectionReaderAnnotator.Descriptor(
            camera_config=cr_storage_camera_config,
            camera_interface=robokudo.io.storage_reader_interface.StorageReaderInterface(
                cr_storage_camera_config
            ),
        )

        sod = StaticObjectDetectorAnnotator.Descriptor()
        sod.parameters.bounding_box_x = 397
        sod.parameters.bounding_box_y = 126
        sod.parameters.bounding_box_width = 49
        sod.parameters.bounding_box_height = 106
        sod.parameters.create_pose_annotation = True
        sod.parameters.pose_use_euler_angles = False
        sod.parameters.position_x = 0.2020410562464292
        sod.parameters.position_y = -0.10916767217331147
        sod.parameters.position_z = 1.095503650235392
        sod.parameters.rotation_x = 0.5745206288180745
        sod.parameters.rotation_y = 0.666488383502241
        sod.parameters.rotation_z = -0.3599883019817597
        sod.parameters.rotation_w = 0.31004468090154913
        sod.parameters.class_name = "Mug"

        seq = robokudo.pipeline.Pipeline("StoragePipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                StaticObjectDetectorAnnotator(descriptor=sod),
                ObjectHypothesisVisualizer(),
                ObjectKnowledgeVisualizer(),
                ClusterColorAnnotator(),
            ]
        )
        return seq
