from pathlib import Path

import py_trees

import robokudo.cas
import robokudo.defs
import robokudo.descriptors.camera_configs.config_filereader_playback
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.garden
import robokudo.io.file_reader_interface
import robokudo.io.storage_reader_interface
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.data_downloader
import robokudo.utils.tree_execution
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors import CrDescriptorFactory
from robokudo.pipeline import Pipeline


class TestFullAEExecution(object):
    def test_run_simple_ae_successfully(self, node):
        # rclpy.init()
        # node = Node(robokudo.defs.TEST_ROS_NODE_NAME)

        cr_fr_config = CrDescriptorFactory.create_descriptor(
            "file_reader",
            loop=False,
            target_dir=robokudo.utils.data_downloader.test_data_path() / Path("data"),
            kinect_height_fix_mode=True,
            color2depth_ratio=(0.5, 0.5),
        )

        # Restrict FOV of pointcloud to robustly get only one object
        pc_crop_config = (
            robokudo.annotators.pointcloud_crop.PointcloudCropAnnotator.Descriptor()
        )
        pc_crop_config.parameters.min_x = -0.3
        pc_crop_config.parameters.max_x = 0.3

        seq = robokudo.pipeline.Pipeline("TestPipeline")
        seq.add_children(
            [
                robokudo.annotators.outputs.ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(descriptor=cr_fr_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(descriptor=pc_crop_config),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
            ]
        )

        tree_result = robokudo.utils.tree_execution.run_tree_once(seq, node)
        assert tree_result is py_trees.common.Status.SUCCESS

        # We should have found a plane and an object
        assert len(seq.cas.annotations) > 0
        types_of_annotations = list(map(type, seq.cas.annotations))

        assert types_of_annotations.count(robokudo.types.annotation.Plane) == 1
        assert types_of_annotations.count(robokudo.types.scene.ObjectHypothesis) == 1
