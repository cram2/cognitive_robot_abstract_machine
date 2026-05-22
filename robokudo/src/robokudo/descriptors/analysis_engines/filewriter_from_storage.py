"""Analysis engine for writing stored data to filesystem.

This module provides an analysis engine that demonstrates how to read data from
MongoDB storage and write it to the local filesystem. It implements a simple
pipeline for data transfer between storage systems.

The pipeline implements the following functionality:

* Reading stored data from MongoDB
* Image preprocessing
* Writing data to local filesystem

.. note::
    This engine requires a properly configured MongoDB database with stored data.
    The data will be written to the local filesystem in a format compatible with
    the FileReader interface.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.file_writer import FileWriter
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.pipeline import Pipeline
from robokudo.descriptors import CrDescriptorFactory


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for transferring data from MongoDB to filesystem.

    This class implements a pipeline that reads data from MongoDB storage and
    writes it to the local filesystem. It is designed to facilitate data
    transfer between different storage systems.

    The pipeline includes:

    * Collection reader for accessing MongoDB data
    * Image preprocessing for data preparation
    * File writer for saving data to filesystem

    .. note::
        The pipeline is configured to read data only once (no looping) to avoid
        duplicate files in the filesystem.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "filewriter_from_storage"

    def implementation(self) -> Pipeline:
        """Create a pipeline for writing MongoDB data to filesystem.

        This method constructs a processing pipeline that reads data from MongoDB
        and writes it to the local filesystem. The pipeline is configured to
        perform a single pass over the stored data.

        :return: The configured pipeline for data transfer
        """
        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo", loop=False)

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                FileWriter(),
            ]
        )
        return seq
