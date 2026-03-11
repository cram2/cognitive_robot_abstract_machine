"""Analysis engine for region-based filtering using semantic maps.

This module provides an analysis engine that demonstrates how to filter point
cloud data based on predefined regions from semantic maps. It processes stored
camera data and applies region-based filtering to focus on specific areas of
interest.

The pipeline implements the following functionality:

* Reading stored camera data from MongoDB
* Image preprocessing
* Region-based filtering using semantic map data
* Optional pipeline trigger for step-by-step execution
* Optional camera viewpoint visualization

.. note::
    This engine requires properly configured semantic maps with defined regions
    of interest. The regions are used to filter the point cloud data during
    processing.
"""

import robokudo.analysis_engine
import robokudo.behaviours.clear_errors
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.idioms
import robokudo.io.storage_reader_interface
import robokudo.pipeline
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.region_filter import RegionFilter


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for region-based point cloud filtering.

    This class implements a pipeline that filters point cloud data based on
    predefined regions from semantic maps. It processes stored camera data
    and applies region filtering to focus on specific areas of interest.

    The pipeline includes:

    * Collection reader for stored data access
    * Image preprocessing
    * Region-based filtering
    * Optional pipeline trigger
    * Optional viewpoint visualization

    .. note::
        The pipeline can be configured to run continuously or with step-by-step
        execution using the pipeline trigger. Viewpoint visualization can be
        enabled for debugging purposes.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "regionfilter_from_storage"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a pipeline for region-based point cloud filtering.

        This method constructs a processing pipeline that applies region-based
        filtering to point cloud data. The regions are defined in semantic maps
        and used to filter the data during processing.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Read stored camera data
        3. Preprocess image
        4. Apply region filter
        5. Optional: Visualize camera viewpoint

        :return: The configured pipeline for region-based filtering
        :rtype: robokudo.pipeline.Pipeline

        .. note::
            The pipeline includes commented-out options for adding a trigger
            and camera viewpoint visualization, which can be useful for
            debugging and development.
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

        seq = robokudo.pipeline.Pipeline("StoragePipeline")
        seq.add_children(
            [
                # PipelineTrigger(),
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                RegionFilter(),
                # CameraViewpointVisualizer(),
            ]
        )
        return seq
