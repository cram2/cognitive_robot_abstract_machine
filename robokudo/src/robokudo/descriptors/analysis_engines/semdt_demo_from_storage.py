import os

import rclpy

import robokudo.analysis_engine
import robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.idioms
import robokudo.io.camera_interface
import robokudo.io.storage_reader_interface
import robokudo.pipeline
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
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self) -> str:
        return "semdt_demo_from_storage"

    def implementation(self) -> robokudo.pipeline.Pipeline:
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

        node = rclpy.create_node("semantic_world")
        viz = VizMarkerPublisher(world=sw_connector.semdt_adapter.world, node=node)

        cr_storage_camera_config = (
            robokudo.descriptors.camera_configs.config_mongodb_playback.CameraConfig()
        )
        cr_storage_config = CollectionReaderAnnotator.Descriptor(
            camera_config=cr_storage_camera_config,
            camera_interface=robokudo.io.storage_reader_interface.StorageReaderInterface(
                cr_storage_camera_config
            ),
        )

        seq = robokudo.pipeline.Pipeline("RWPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                ClusterColorHistogramAnnotator(),
                ClusterPoseBBAnnotator(),
                SimpleYoloAnnotator(),
                sw_connector,
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
            ]
        )
        return seq
