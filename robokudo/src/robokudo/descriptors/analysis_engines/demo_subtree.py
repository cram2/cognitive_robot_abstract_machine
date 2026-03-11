import robokudo.analysis_engine
import robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform
import robokudo.idioms
import robokudo.io.camera_interface
import robokudo.pipeline
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.analysis_engines.subtree_tabletop_object_localization import (
    Subtree as TTLocalizationSubtree,
)


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self) -> str:
        return "demo_subtree"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """
        Create a pipeline that does tabletop segmentation.
        """
        kinect_camera_config = (
            robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform.CameraConfig()
        )
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(
                kinect_camera_config
            ),
        )

        seq = robokudo.pipeline.Pipeline("RWPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                TTLocalizationSubtree().implementation(),
                ClusterColorAnnotator(),
                ClusterColorHistogramAnnotator(),
                ClusterPositionAnnotator(),
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
            ]
        )
        return seq
