"""
Compose tabletop localization and color annotation in a Kinect pipeline.
"""

from __future__ import annotations

# %% Imports
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.analysis_engines.subtree_tabletop_object_localization import (
    Subtree as TTLocalizationSubtree,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


# %% Analysis engine
class AnalysisEngine(AnalysisEngineInterface):
    """
    Combine reusable tabletop localization with object color and position annotations.
    """

    def name(self) -> str:
        return "demo_subtree"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline that does tabletop segmentation.
        """
        kinect_config = CollectionReaderDescriptorFactory.create_descriptor(
            "kinect_wo_tf"
        )

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                PointcloudCropAnnotator(),
                TTLocalizationSubtree().implementation(),
                ClusterColorAnnotator(),
                ClusterColorHistogramAnnotator(),
                ClusterPositionAnnotator(),
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
            ]
        )
        return seq
