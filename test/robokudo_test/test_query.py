from robokudo.descriptors import CrDescriptorFactory
import multiprocessing
import queue
import threading
import time
from pathlib import Path

import py_trees
import pytest

from action_msgs.msg import GoalStatus

import robokudo.cas
import robokudo.defs
import robokudo.descriptors.camera_configs.config_filereader_playback
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.garden
import robokudo.idioms
import robokudo.io.file_reader_interface
import robokudo.io.storage_reader_interface
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.tree_execution
import robokudo.utils.data_downloader
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryAnnotator, GenerateQueryResult
from robokudo.pipeline import Pipeline

import robokudo.scripts.query_test_client

# def query_worker(fun):
#     return fun()


class QueryWorkerThread(threading.Thread):
    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node
        self.result = None

    def run(self):
        self.result = query_simple_pipeline(self.node)


def query_simple_pipeline(node):
    cr_fr_config = CrDescriptorFactory.create_descriptor(
        "file_reader",
        loop=True,
        target_dir=robokudo.utils.data_downloader.test_data_path() / Path("data"),
        kinect_height_fix_mode=True,
        color2depth_ratio=(0.5, 0.5),
    )

    # Restrict FOV of pointcloud to robustly get only one object
    pc_crop_config = PointcloudCropAnnotator.Descriptor()
    pc_crop_config.parameters.min_x = -0.3
    pc_crop_config.parameters.max_x = 0.3

    seq = Pipeline("TestPipeline")
    seq.add_children(
        [
            robokudo.idioms.pipeline_init(),
            QueryAnnotator(),
            CollectionReaderAnnotator(descriptor=cr_fr_config),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            PointcloudCropAnnotator(descriptor=pc_crop_config),
            PlaneAnnotator(),
            PointCloudClusterExtractor(),
            GenerateQueryResult(),
        ]
    )

    tree_result = robokudo.utils.tree_execution.run_tree_once(
        tree=seq, node=node, max_iterations=20, tick_rate=5
    )

    return seq, tree_result


@pytest.fixture
def tree_run(node):
    print("tree_run")
    return query_simple_pipeline(node)


class TestQueryInterface:
    def test_query_annotator_returns_running(self, tree_run):
        seq, tree_result = tree_run

        # We are not sending a query just yet. So the QueryAnnotator should block indefinitely with RUNNING
        assert tree_result is py_trees.common.Status.RUNNING

    def test_query(self, node):
        # seq, tree_result = query_simple_pipeline(node)
        multiprocessing.set_start_method("spawn", force=True)

        print("Starting QueryWorker Thread")
        # Start the ActionServer/Query Interface PPT
        t = QueryWorkerThread(node)
        t.start()

        # Wait until ActionServer is set up and PPT ticked
        time.sleep(0.75)

        # Start the action client
        print("Starting Client Process")
        client_results = queue.Queue()
        p = multiprocessing.Process(
            target=robokudo.scripts.query_test_client.main(result=client_results)
        )
        p.start()

        t.join()
        print("Thread joined")
        seq, tree_result = t.result
        p.join()
        print("Process joined")

        # PPT with action server completed successfully
        assert tree_result is py_trees.common.Status.SUCCESS
        # The action client did not time out
        client_result = client_results.get()
        assert client_result["timed_out"] is False
        assert client_result["goal_status"] is GoalStatus.STATUS_SUCCEEDED
        assert len(client_result["goal_result"].res) == 1

    # def test_normal_ae_run_without_query(self, node):
    #     cr_fr_camera_config = robokudo.descriptors.camera_configs.config_filereader_playback.CameraConfig()
    #     cr_fr_camera_config.loop = True
    #     cr_fr_camera_config.target_ros_package = "robokudo_test_data"
    #     cr_fr_camera_config.target_dir = "data"
    #     cr_fr_camera_config.kinect_height_fix_mode = True
    #     cr_fr_camera_config.color2depth_ratio = (0.5, 0.5)
    #
    #     cr_fr_config = CollectionReaderAnnotator.Descriptor(
    #         camera_config=cr_fr_camera_config,
    #         camera_interface=robokudo.io.file_reader_interface.RGBDFileReaderInterface(cr_fr_camera_config),
    #     )
    #
    #     # Restrict FOV of pointcloud to robustly get only one object
    #     pc_crop_config = PointcloudCropAnnotator.Descriptor()
    #     pc_crop_config.parameters.min_x = -0.3
    #     pc_crop_config.parameters.max_x = 0.3
    #
    #     seq = Pipeline("TestPipeline")
    #     seq.add_children(
    #         [
    #             robokudo.annotators.outputs.ClearAnnotatorOutputs(),
    #             CollectionReaderAnnotator(descriptor=cr_fr_config),
    #             ImagePreprocessorAnnotator("ImagePreprocessor"),
    #             PointcloudCropAnnotator(descriptor=pc_crop_config),
    #             PlaneAnnotator(),
    #             PointCloudClusterExtractor(),
    #             GenerateQueryResult(),
    #         ]
    #     )
    #
    #     tree_result = robokudo.utils.tree_execution.run_tree_once(
    #         tree=seq, node=node, max_iterations=20, tick_rate=5
    #     )
    #     assert (tree_result is py_trees.common.Status.SUCCESS)
    #
    #     # We should have found a plane and an object
    #     assert (len(seq.cas.annotations) > 0)
    #     types_of_annotations = list(map(type, seq.cas.annotations))
    #
    #     assert (types_of_annotations.count(robokudo.types.annotation.Plane) == 1)
    #     assert (types_of_annotations.count(robokudo.types.scene.ObjectHypothesis) == 1)

    #
    # def test_run_simple_ae_successfully(self, function_setup):
    #     rclpy.init()
    #     node = Node(robokudo.defs.TEST_ROS_NODE_NAME)
    #
    #     cr_fr_camera_config = robokudo.descriptors.camera_configs.config_filereader_playback.CameraConfig()
    #     cr_fr_camera_config.loop = True
    #     cr_fr_camera_config.target_ros_package = 'robokudo_test_data'
    #     cr_fr_camera_config.target_dir = 'data'
    #     cr_fr_camera_config.kinect_height_fix_mode = True
    #     cr_fr_camera_config.color2depth_ratio = (0.5, 0.5)
    #
    #     cr_fr_config = CollectionReaderAnnotator.Descriptor(
    #         camera_config=cr_fr_camera_config,
    #         camera_interface=robokudo.io.file_reader_interface.RGBDFileReaderInterface(cr_fr_camera_config))
    #
    #     # Restrict FOV of pointcloud to robustly get only one object
    #     pc_crop_config = robokudo.annotators.pointcloud_crop.PointcloudCropAnnotator.Descriptor()
    #     pc_crop_config.parameters.min_x = -0.3
    #     pc_crop_config.parameters.max_x = 0.3
    #
    #     seq = robokudo.pipeline.Pipeline("TestPipeline")
    #     seq.add_children(
    #         [
    #             robokudo.annotators.outputs.ClearAnnotatorOutputs(),
    #             QueryAnnotator(),
    #             CollectionReaderAnnotator(descriptor=cr_fr_config),
    #             ImagePreprocessorAnnotator("ImagePreprocessor"),
    #             PointcloudCropAnnotator(descriptor=pc_crop_config),
    #             PlaneAnnotator(),
    #             PointCloudClusterExtractor(),
    #             GenerateQueryResult(),
    #         ])
    #
    #     tree_result = robokudo.utils.tree_execution.run_tree_once(tree=seq, node=node, max_iterations=20, tick_rate=5)
    #     assert (tree_result is py_trees.common.Status.RUNNING)
    #
    #     # We should have found a plane and an object
    #     assert (len(seq.cas.annotations) > 0)
    #     types_of_annotations = list(map(type, seq.cas.annotations))
    #
    #     assert (types_of_annotations.count(robokudo.types.annotation.Plane) == 1)
    #     assert (types_of_annotations.count(robokudo.types.scene.ObjectHypothesis) == 1)
