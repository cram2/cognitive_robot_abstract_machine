"""
Analysis engine that records Kinect camera data to storage.
"""

# %% Imports
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.storage import StorageWriter
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


# %% Analysis engine
class AnalysisEngine(AnalysisEngineInterface):
    """
    Records preprocessed Kinect camera data with StorageWriter.
    """

    def name(self) -> str:
        return "storage"

    def implementation(self) -> Pipeline:
        """
        Build :meth:`AnalysisEngineInterface.implementation` for recording preprocessed
        Kinect frames.
        """
        kinect_config = CollectionReaderDescriptorFactory.create_descriptor("kinect")

        seq = Pipeline()
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                StorageWriter(),
            ]
        )
        return seq
