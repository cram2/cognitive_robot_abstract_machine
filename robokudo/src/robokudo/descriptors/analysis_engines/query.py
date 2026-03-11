"""Analysis engine for handling query-based processing.

This module provides an analysis engine that demonstrates how to implement
query-based processing in a pipeline. It shows how to set up a pipeline that
can receive queries, process them using camera data, and return responses.

The pipeline implements the following functionality:

* Query reception and handling
* Kinect camera data processing
* Query response generation
* Action server status checking

.. note::
    This is a basic query handling pipeline that can be extended with additional
    processing steps between query reception and response generation.
"""

import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
import robokudo.annotators.query
import robokudo.pipeline

import robokudo.descriptors.camera_configs.config_kinect_robot
import robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform

import robokudo.io.camera_interface
import robokudo.idioms
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.utils.tree import add_children_to_parent


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for query-based processing.

    This class implements a pipeline that handles incoming queries by processing
    them with camera data and generating appropriate responses. It uses an
    action server to manage the query-response cycle.

    The pipeline includes:

    * Query reception through QueryAnnotator
    * Camera data collection and preprocessing
    * Query response generation
    * Action server status monitoring

    .. note::
        The pipeline is designed to work with the ROS action server framework,
        allowing external systems to send queries and receive responses.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "query"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a pipeline for query-based processing.

        This method constructs a processing pipeline that can handle incoming
        queries. The pipeline receives queries through a QueryAnnotator,
        processes them using camera data, and generates responses.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Wait for query
        3. Read camera data
        4. Preprocess image
        5. Generate query response
        6. Check action server status

        :return: The configured pipeline for query processing
        """
        # kinect_camera_config = robokudo.descriptors.camera_configs.config_kinect_robot.CameraConfig()
        kinect_camera_config = (
            robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform.CameraConfig()
        )
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(
                kinect_camera_config
            ),
        )

        seq = robokudo.pipeline.Pipeline("RWPipeline")

        add_children_to_parent(
            seq,
            [
                robokudo.idioms.pipeline_init(),
                robokudo.annotators.query.QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                robokudo.annotators.query.QueryReply(),
                ActionServerCheck(),
            ],
        )

        return seq
