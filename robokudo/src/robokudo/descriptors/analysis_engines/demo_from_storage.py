from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
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
from robokudo.io.ros import get_node
from robokudo.pipeline import Pipeline
from robokudo.world import world_instance


class AnalysisEngine(AnalysisEngineInterface):
    """
    Analysis engine for demonstrating tabletop segmentation from stored data.

    This class implements a basic pipeline for tabletop segmentation using stored camera
    data. It reads data from MongoDB storage and processes it through a sequence of
    annotators to detect and segment objects on a table surface.
    """

    def name(self) -> str:
        """
        Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "demo_from_storage"

    def implementation(self) -> Pipeline:
        """
        Create a basic pipeline for tabletop segmentation.

        This method constructs the processing pipeline by configuring and connecting the
        necessary annotators in sequence.

        :return: The configured pipeline for tabletop segmentation
        """
        tf_publisher = TFPublisher(_world=world_instance(), node=get_node())

        cr_storage_config = CollectionReaderDescriptorFactory.create_descriptor("mongo")

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                OutlierRemovalOnObjectHypothesisAnnotator(),
                ClusterPoseBBAnnotator(),
                # SlowAnnotator("SlowAnnotator",sleep_in_s=0),
            ]
        )
        return seq
