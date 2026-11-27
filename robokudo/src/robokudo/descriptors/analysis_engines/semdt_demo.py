"""
Analysis engine putting the tabletop objects it segments in Kinect data into the world.
"""

from __future__ import annotations

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from robokudo.annotators.world_updating import WorldUpdatingAnnotator


class AnalysisEngine(AnalysisEngineInterface):
    """
    Segments the tabletop objects in Kinect data, annotates their color, color histogram
    and bounding box, and puts them into the world.
    """

    def name(self) -> str:
        return "semdt_demo"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline that does tabletop segmentation and integrates primary
        navigation using a YOLO annotator.
        """
        kinect_config = CollectionReaderDescriptorFactory.create_descriptor(
            camera="kinect"
        )

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                ClusterColorHistogramAnnotator(),
                ClusterPoseBBAnnotator(),
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
                WorldUpdatingAnnotator(),
            ]
        )
        return seq
