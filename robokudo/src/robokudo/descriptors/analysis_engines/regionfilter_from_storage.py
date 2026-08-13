"""Analysis engine for region-based filtering using world descriptors.

This module provides an analysis engine that demonstrates how to filter point
cloud data based on predefined regions from world descriptors. It processes stored
camera data and applies region-based filtering to focus on specific areas of
interest.

The pipeline implements the following functionality:

* Reading stored camera data from MongoDB
* Image preprocessing
* Region-based filtering using world descriptor data
* Optional pipeline trigger for step-by-step execution
* Optional camera viewpoint visualization

.. note::
    This engine requires properly configured world descriptors with defined regions
    of interest. The regions are used to filter the point cloud data during
    processing.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.region_filter import RegionFilter
from robokudo.annotators.world_descriptor_bootstrap import (
    WorldDescriptorBootstrapAnnotator,
)
from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for region-based point cloud filtering.

    This class implements a pipeline that filters point cloud data based on
    predefined regions from world descriptors. It processes stored camera data
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

    def implementation(self) -> Pipeline:
        """Create a pipeline for region-based point cloud filtering.

        This method constructs a processing pipeline that applies region-based
        filtering to point cloud data. The regions are defined in world descriptors
        and used to filter the data during processing.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Read stored camera data
        3. Bootstrap world descriptor entities into the shared world
        4. Preprocess image
        5. Apply region filter
        6. Optional: Visualize camera viewpoint

        :return: The configured pipeline for region-based filtering

        .. note::
            The pipeline includes commented-out options for adding a trigger
            and camera viewpoint visualization, which can be useful for
            debugging and development.
        """
        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo")
        bootstrap = WorldDescriptorBootstrapAnnotator.Descriptor()
        bootstrap.parameters.world_descriptor_name = "world_iai_kitchen20"

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                # PipelineTrigger(),
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                WorldDescriptorBootstrapAnnotator(descriptor=bootstrap),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                RegionFilter(),
                # CameraViewpointVisualizer(),
            ]
        )
        return seq
