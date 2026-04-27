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

from robokudo.analysis_engine import AnalysisEngineInterface

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import QueryAnnotator, QueryReply
from robokudo.pipeline import Pipeline

from robokudo.idioms import pipeline_init
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.descriptors import CrDescriptorFactory
from robokudo.utils.tree import add_children_to_parent


class AnalysisEngine(AnalysisEngineInterface):
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

    def implementation(self) -> Pipeline:
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
        kinect_config = CrDescriptorFactory.create_descriptor("kinect_wo_tf")

        seq = Pipeline("RWPipeline")

        add_children_to_parent(
            seq,
            [
                pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                QueryReply(),
                ActionServerCheck(),
            ],
        )

        return seq
