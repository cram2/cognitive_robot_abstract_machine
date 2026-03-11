"""Analysis engine for visualizing stored annotations.

This module provides an analysis engine that demonstrates how to read and display
data and annotations that have been previously stored in a MongoDB database. It
implements a simple pipeline for retrieving and visualizing stored object
hypotheses.

The pipeline implements the following functionality:
- Reading stored data from MongoDB
- Image preprocessing
- Visualization of stored object hypotheses

.. note::
    This engine requires pre-existing data in the MongoDB database. Make sure to
    store some annotated data before running this pipeline.
"""

import robokudo.analysis_engine
import robokudo.behaviours.clear_errors
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.idioms
import robokudo.pipeline
import robokudo.io.storage_reader_interface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for visualizing stored annotations.

    This class implements a pipeline that reads previously stored data and
    annotations from a MongoDB database and visualizes them. It is designed
    to demonstrate how stored object hypotheses can be retrieved and displayed.

    The pipeline includes:
    - Collection reader for accessing stored data
    - Image preprocessing
    - Object hypothesis visualization

    .. note::
        The pipeline expects data to be stored in a MongoDB database named
        'store_with_annotations'. Ensure this database exists and contains
        the required data before running the pipeline.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "annotations_from_storage"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a pipeline for visualizing stored annotations.

        This method constructs a processing pipeline that reads stored data and
        annotations from MongoDB and visualizes them. The pipeline is configured
        to read from a specific database named 'store_with_annotations'.

        :return: The configured pipeline for annotation visualization
        :rtype: robokudo.pipeline.Pipeline

        .. warning::
            Make sure to store some annotated data in the MongoDB database
            before running this pipeline, or it will not display anything.
        """
        cr_storage_camera_config = (
            robokudo.descriptors.camera_configs.config_mongodb_playback.CameraConfig()
        )
        cr_storage_camera_config.db_name = "store_with_annotations"
        cr_storage_config = CollectionReaderAnnotator.Descriptor(
            camera_config=cr_storage_camera_config,
            camera_interface=robokudo.io.storage_reader_interface.StorageReaderInterface(
                cr_storage_camera_config
            ),
        )

        seq = robokudo.pipeline.Pipeline("StoragePipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                ObjectHypothesisVisualizer(),
            ]
        )
        return seq
