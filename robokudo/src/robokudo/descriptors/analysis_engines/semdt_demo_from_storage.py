from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outlier_removal_objecthypothesis import (
    OutlierRemovalOnObjectHypothesisAnnotator,
)
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.semantic_world_connector import SemanticDigitalTwinConnector
from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "semdt_demo_from_storage"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline that does tabletop segmentation and integrates primary navigation
        using a YOLO annotator.
        """

        descriptor = SemanticDigitalTwinConnector.Descriptor()
        sw_connector = SemanticDigitalTwinConnector(descriptor=descriptor)

        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo")

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                OutlierRemovalOnObjectHypothesisAnnotator(),
                sw_connector,
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
            ]
        )
        return seq
