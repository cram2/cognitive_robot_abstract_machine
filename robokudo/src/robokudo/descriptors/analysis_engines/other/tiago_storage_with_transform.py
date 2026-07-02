"""Analysis engine for recording TIAGo sensor data with transforms.

This module provides an analysis engine that demonstrates how to record sensor
data from a TIAGo robot's camera system while preserving transformation data.
It implements a pipeline that captures, preprocesses, and stores camera data
along with associated coordinate transforms.

The pipeline implements the following functionality:

* Reading data from TIAGo's camera
* Image preprocessing
* Data storage with transform information

.. note::
    This engine uses the TIAGo camera configuration with transform lookup
    enabled, ensuring that coordinate transformations are properly recorded
    along with the sensor data.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.storage import StorageWriter
from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for TIAGo sensor data recording with transforms.

    This class implements a pipeline that records sensor data from a TIAGo
    robot's camera system while preserving coordinate transformation data.
    It ensures that spatial relationships between different coordinate frames
    are properly stored.

    The pipeline includes:

    * Collection reader for TIAGo camera data
    * Image preprocessing
    * Storage writer with transform preservation

    .. note::
        The pipeline is configured to use the TIAGo camera configuration
        with transform lookup enabled, which is essential for maintaining
        spatial relationships in the stored data.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "tiago_storage_with_transform"

    def implementation(self) -> Pipeline:
        """Create a pipeline for recording TIAGo data with transforms.

        This method constructs a processing pipeline that captures and stores
        sensor data from a TIAGo robot's camera system. The pipeline ensures
        that coordinate transformation data is preserved during storage.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Read TIAGo camera data (with transforms)
        3. Preprocess image
        4. Store data and transforms

        :return: The configured pipeline for data recording
        """
        tiago_config = CrDescriptorFactory.create_descriptor("tiago")

        seq = Pipeline()
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=tiago_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                StorageWriter(),
            ]
        )
        return seq
