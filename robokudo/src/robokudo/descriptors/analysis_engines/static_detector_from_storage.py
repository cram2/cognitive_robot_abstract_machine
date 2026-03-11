"""Analysis engine for static object detection from stored data.

This module provides an analysis engine that demonstrates static object detection
using stored camera data. It implements a pipeline that uses predefined bounding
box parameters to detect objects in stored images.

The pipeline implements the following functionality:

* Reading stored camera data from MongoDB
* Image preprocessing
* Static object detection with fixed bounding box
* Object hypothesis visualization

.. note::
    This engine uses a static object detector with fixed bounding box parameters,
    making it suitable for detecting objects with known and consistent dimensions
    in the camera view.
"""

import robokudo.analysis_engine
from robokudo.annotators.camera_viewpoint_visualizer import CameraViewpointVisualizer

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator

import robokudo.descriptors.camera_configs.config_mongodb_playback

import robokudo.io.storage_reader_interface

import robokudo.behaviours.clear_errors

import robokudo.idioms
import robokudo.pipeline
from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for static object detection from stored data.

    This class implements a pipeline that performs object detection using a
    static bounding box approach. It processes stored camera data and detects
    objects based on predefined bounding box dimensions.

    The pipeline includes:

    * Collection reader for stored data access
    * Image preprocessing
    * Static object detection with fixed parameters
    * Object hypothesis visualization

    .. note::
        The static object detector is configured with a fixed bounding box
        of 100x60 pixels, positioned at (40, 40). Adjust these parameters
        based on your specific object detection requirements.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "static_detector_from_storage"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a pipeline for static object detection.

        This method constructs a processing pipeline that performs object
        detection using a static bounding box approach. The detector uses
        fixed parameters suitable for objects with known dimensions.

        Static detector configuration:

        * Bounding box position: (40, 40)
        * Bounding box size: 100x60 pixels

        :return: The configured pipeline for static object detection
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
        sod.parameters.bounding_box_x = 40
        sod.parameters.bounding_box_y = 40
        sod.parameters.bounding_box_width = 100
        sod.parameters.bounding_box_height = 60

        seq = robokudo.pipeline.Pipeline("StoragePipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                StaticObjectDetectorAnnotator(descriptor=sod),
                ObjectHypothesisVisualizer(),
            ]
        )
        return seq
