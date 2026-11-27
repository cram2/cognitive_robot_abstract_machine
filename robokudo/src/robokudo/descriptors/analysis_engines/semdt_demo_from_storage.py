"""
Analysis engine putting the tabletop objects it segments in camera data replayed from
MongoDB into the world.
"""

from __future__ import annotations

from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outlier_removal_objecthypothesis import (
    OutlierRemovalOnObjectHypothesisAnnotator,
)
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.annotators.world_updating import WorldUpdatingAnnotator


class AnalysisEngine(AnalysisEngineInterface):
    """
    Segments the tabletop objects in camera data replayed from MongoDB, annotates their
    color histogram and bounding box, and puts them into the world.
    """

    def name(self) -> str:
        return "semdt_demo_from_storage"

    def implementation(self) -> Pipeline:
        """
        Create a tabletop segmentation pipeline that synchronizes detections to SemDT.
        """
        cr_storage_config = CollectionReaderDescriptorFactory.create_descriptor(
            camera="mongo"
        )

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorHistogramAnnotator(),
                OutlierRemovalOnObjectHypothesisAnnotator(),
                ClusterPoseBBAnnotator(),
                WorldUpdatingAnnotator(),
            ]
        )
        return seq
