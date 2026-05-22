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

from py_trees.composites import Sequence
from py_trees.decorators import Inverter, Condition
from py_trees.behaviours import SuccessEveryN
from py_trees.common import Status

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryAnnotator

from robokudo.annotators.vis import Redraw
from robokudo.behaviours.action_server_checks import (
    ActionServerNoPreemptRequest,
    AbortGoal,
)
from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
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

    def implementation(self) -> Pipeline:
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
        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo")

        processing_sequence = Sequence()
        processing_sequence.add_children(
            [
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                Redraw(),
                ActionServerNoPreemptRequest(),
                Inverter(SuccessEveryN("Fail Sim after 30 iter", n=30)),
            ]
        )

        pipeline = Pipeline("StoragePipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                Condition(child=processing_sequence, status=Status.FAILURE),
                AbortGoal(),
            ]
        )

        return pipeline
