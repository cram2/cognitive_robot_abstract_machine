"""
Analysis engine that perceives objects from recorded RGB-D files into the semantic
digital twin.

It runs without a camera or a database: the images come from the
``robokudo_test_data`` package, and the camera is placed at a fixed pose above the
scene. Each segmented object becomes a body in the perception world, so with
``_synchronize_world`` the objects appear in the world of every process sharing it::

    python -m robokudo.scripts.shared_world_server
    ros2 run robokudo_ros main _ae=semdt_shared_world_from_files _synchronize_world
"""

import math

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from robokudo.analysis_engine import AnalysisEngineInterface
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

CAMERA_HEIGHT = 1.2
"""
Height of the camera above the world frame, in meters.
"""

CAMERA_TILT = 0.6
"""
How far the camera looks down from the horizontal, in radians.
"""

WORLD_T_CAMERA = HomogeneousTransformationMatrix.from_xyz_rpy(
    z=CAMERA_HEIGHT, roll=-math.pi / 2 - CAMERA_TILT, yaw=-math.pi / 2
)
"""
Pose of the optical camera frame (z forward, x right, y down) in the world frame,
looking along the world x axis.
"""


class AnalysisEngine(AnalysisEngineInterface):
    """
    Segments the objects on the table in the recorded scene and adds them to the
    perception world.
    """

    def name(self) -> str:
        return "semdt_shared_world_from_files"

    def implementation(self) -> Pipeline:
        file_reader_config = CollectionReaderDescriptorFactory.create_descriptor(
            "file_reader",
            loop=True,
            target_ros_package="robokudo_test_data",
            target_dir="data",
            kinect_height_fix_mode=True,
            color2depth_ratio=(0.5, 0.5),
            static_camera_transform_enabled=True,
            static_world_T_camera=WORLD_T_CAMERA,
        )

        pipeline = Pipeline("SharedWorldFromFilesPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=file_reader_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterPoseBBAnnotator(),
            ]
        )
        return pipeline
