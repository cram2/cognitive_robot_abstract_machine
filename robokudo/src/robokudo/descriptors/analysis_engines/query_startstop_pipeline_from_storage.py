"""Analysis engine for continuous perception with start/stop control.

This module provides an analysis engine that demonstrates how to implement a
continuous perception pipeline with external start/stop control through an
action server. It processes stored camera data and can be started, stopped,
and preempted by external commands.

The pipeline implements the following functionality:

* Query-based start/stop control
* Continuous processing of stored camera data
* Image preprocessing and point cloud analysis
* Object segmentation and visualization
* Automatic failure after 30 iterations
* Preemption handling

.. note::
    This engine demonstrates how to implement a long-running perception pipeline
    that can be controlled externally through ROS action server commands.

.. warning::
    The pipeline is designed to fail after 30 iterations to demonstrate
    error handling and recovery mechanisms.
"""

import py_trees

import robokudo.analysis_engine
from robokudo.annotators.blur import BlurAnnotator
from robokudo.annotators.cluster_color import ClusterColorAnnotator

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator

import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.io.storage_reader_interface
import robokudo.annotators.query

import robokudo.idioms
import robokudo.pipeline
from robokudo.annotators.vis import Redraw
from robokudo.behaviours.action_server_checks import (
    ActionServerCheck,
    ActionServerNoPreemptRequest,
    AbortGoal,
)


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for continuous perception with external control.

    This class implements a pipeline that runs continuous perception tasks
    while allowing external control through an action server. The pipeline
    can be started, stopped, and preempted, and includes automatic failure
    simulation for testing error handling.

    The pipeline includes:

    * Query handling for start/stop control
    * Continuous camera data processing
    * Point cloud analysis and segmentation
    * Visualization updates
    * Preemption checking
    * Simulated failure after 30 iterations

    .. note::
        The pipeline uses a condition decorator to continue processing
        until failure or preemption occurs.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "query_startstop_pipeline_from_storage"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a continuous perception pipeline with external control.

        This method constructs a processing pipeline that runs continuously
        until stopped or preempted. The pipeline processes stored camera data
        and includes automatic failure simulation after 30 iterations.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Wait for start command
        3. Begin continuous processing:
           * Read stored camera data
           * Preprocess images
           * Crop point cloud
           * Detect table plane
           * Extract object clusters
           * Update visualization
           * Check for preemption
           * Fail after 30 iterations
        4. Handle failure by aborting goal

        :return: The configured pipeline with start/stop control

        .. warning::
            The pipeline will automatically fail after 30 iterations to
            demonstrate error handling mechanisms.
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

        processing_sequence = py_trees.composites.Sequence()
        processing_sequence.add_children(
            [
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                Redraw(),
                ActionServerNoPreemptRequest(),
                py_trees.decorators.Inverter(
                    py_trees.behaviours.SuccessEveryN("Fail Sim after 30 iter", n=30)
                ),
            ]
        )

        pipeline = robokudo.pipeline.Pipeline("StoragePipeline")
        pipeline.add_children(
            [
                robokudo.idioms.pipeline_init(),
                robokudo.annotators.query.QueryAnnotator(),
                py_trees.decorators.Condition(
                    child=processing_sequence, status=py_trees.Status.FAILURE
                ),
                AbortGoal(),
            ]
        )

        return pipeline
