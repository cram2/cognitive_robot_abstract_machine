from robokudo.annotators.tsdf_annotator import TSDFAnnotator
import os

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.semantic_world_connector import SemanticDigitalTwinConnector
from robokudo.annotators.simple_yolo_annotator import SimpleYoloAnnotator
from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.io.ros import get_node
from robokudo.pipeline import Pipeline
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "semdt_demo_from_storage"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline that does tabletop segmentation and integrates primary navigation
        using a YOLO annotator.
        """

        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "pycram",
            "resources",
            "worlds",
            "apartment.urdf",
        )

        descriptor = SemanticDigitalTwinConnector.Descriptor()
        # descriptor.parameters.urdf_path = urdf_path
        sw_connector = SemanticDigitalTwinConnector(descriptor=descriptor)

        viz = VizMarkerPublisher(
            world=sw_connector.semdt_adapter.world, node=get_node()
        )

        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo", loop=False)

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                ClusterColorHistogramAnnotator(),
                ClusterPoseBBAnnotator(),
                TSDFAnnotator(),
                SimpleYoloAnnotator(),
                sw_connector,
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
            ]
        )
        return seq
