"""Analysis engine for recording sensor data to storage.

This module provides an analysis engine that demonstrates how to record sensor
data from a Kinect camera to storage. It implements a simple pipeline that
captures, preprocesses, and stores camera data for later use.

The pipeline implements the following functionality:

* Reading data from a Kinect camera
* Image preprocessing
* Data storage using StorageWriter

.. note::
    This engine can be configured to use either a standard Kinect camera
    configuration or one without transform lookup, depending on the application
    requirements.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.storage import StorageWriter

from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for sensor data recording.

    This class implements a pipeline that records sensor data from a Kinect
    camera to storage. It can be configured to use either a standard Kinect
    configuration or one without transform lookup.

    The pipeline includes:

    * Collection reader for Kinect camera data
    * Image preprocessing
    * Storage writer for data persistence

    .. note::
        The pipeline uses the Kinect configuration without transform lookup
        by default. Uncomment the alternative configuration to enable
        transform lookup if needed.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "storage"

    def implementation(self) -> Pipeline:
        """Create a pipeline for recording sensor data.

        This method constructs a processing pipeline that captures and stores
        sensor data from a Kinect camera. The pipeline preprocesses the data
        before storing it for later use.

        Pipeline configuration options:

        * Standard Kinect config (with transform lookup)
        * Kinect config without transform lookup (default)

        :return: The configured pipeline for data recording
        """

        kinect_config = CrDescriptorFactory.create_descriptor("kinect")

        seq = Pipeline()
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                StorageWriter(),
            ]
        )
        return seq
