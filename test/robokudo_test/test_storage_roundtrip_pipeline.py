import os
from pathlib import Path
import json
import uuid
import pytest

import numpy as np
import py_trees
from rclpy.node import Node

import robokudo.defs
from robokudo import world as rk_world
import robokudo.descriptors.camera_configs.config_filereader_playback
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.utils.data_downloader
import robokudo.utils.tree_execution
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.annotators.storage import StorageWriter
from robokudo.cas import CASViews
from robokudo.descriptors import CrDescriptorFactory
from robokudo.io.storage import Storage
from robokudo.pipeline import Pipeline
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot


pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="module temporarily disabled until storage functionality is migrated to ormatic",
)


def _cam_info_k_values(cam_info):
    if hasattr(cam_info, "k"):
        return list(cam_info.k)
    return list(cam_info.K)


def _build_writer_pipeline(db_name: str) -> Pipeline:
    file_reader_descriptor = CrDescriptorFactory.create_descriptor(
        "file_reader",
        loop=False,
        target_dir=robokudo.utils.data_downloader.test_data_path() / Path("data"),
        kinect_height_fix_mode=True,
        color2depth_ratio=(0.5, 0.5),
    )

    writer_descriptor = StorageWriter.Descriptor()
    writer_descriptor.parameters.db_name = db_name
    writer_descriptor.parameters.drop_database_on_storage = True

    pipeline = Pipeline("WriterPipeline")
    pipeline.add_children(
        [
            ClearAnnotatorOutputs(),
            CollectionReaderAnnotator(descriptor=file_reader_descriptor),
            StorageWriter(descriptor=writer_descriptor),
        ]
    )
    return pipeline


def _build_reader_pipeline(db_name: str) -> Pipeline:
    mongo_descriptor = CrDescriptorFactory.create_descriptor(
        "mongo", loop=False, db_name=db_name
    )
    pipeline = Pipeline("ReaderPipeline")
    pipeline.add_children(
        [
            ClearAnnotatorOutputs(),
            CollectionReaderAnnotator(descriptor=mongo_descriptor),
        ]
    )
    return pipeline


class TestStorageRoundtripPipeline:
    def test_store_and_replay_sensor_data_roundtrip(self):
        db_name = f"ONLY_UNITTESTS_roundtrip_{uuid.uuid4().hex}"
        storage = Storage(db_name)
        writer_node = Node(
            f"{robokudo.defs.TEST_ROS_NODE_NAME}_writer_{uuid.uuid4().hex}"
        )
        reader_node = Node(
            f"{robokudo.defs.TEST_ROS_NODE_NAME}_reader_{uuid.uuid4().hex}"
        )
        try:
            writer_pipeline = _build_writer_pipeline(db_name)
            writer_status = robokudo.utils.tree_execution.run_tree_once(
                writer_pipeline, writer_node
            )
            assert writer_status is py_trees.common.Status.SUCCESS
            assert storage.db.cas.count_documents({}) == 1
            stored_record = storage.db.cas.find_one({})
            assert stored_record is not None
            assert "world" in stored_record

            world_snapshot_payload = json.loads(stored_record["world"])
            assert "modifications" in world_snapshot_payload
            assert "state" in world_snapshot_payload
            assert {"ids", "states"}.issubset(world_snapshot_payload["state"])

            tracker = rk_world.init_world_with_entity_tracker()
            WorldModelSnapshot.apply_to_json_snapshot_to_world(
                rk_world.world_instance(),
                world_snapshot_payload,
                **tracker.create_kwargs(),
            )
            assert len(rk_world.world_instance().state.keys()) == len(
                world_snapshot_payload["state"]["ids"]
            )

            reader_pipeline = _build_reader_pipeline(db_name)
            reader_status = robokudo.utils.tree_execution.run_tree_once(
                reader_pipeline, reader_node
            )
            assert reader_status is py_trees.common.Status.SUCCESS

            np.testing.assert_array_equal(
                writer_pipeline.cas.get(CASViews.COLOR_IMAGE),
                reader_pipeline.cas.get(CASViews.COLOR_IMAGE),
            )
            np.testing.assert_allclose(
                writer_pipeline.cas.get(CASViews.DEPTH_IMAGE),
                reader_pipeline.cas.get(CASViews.DEPTH_IMAGE),
                equal_nan=True,
            )

            writer_cam_info = writer_pipeline.cas.get(CASViews.CAM_INFO)
            reader_cam_info = reader_pipeline.cas.get(CASViews.CAM_INFO)
            assert writer_cam_info.width == reader_cam_info.width
            assert writer_cam_info.height == reader_cam_info.height
            assert _cam_info_k_values(writer_cam_info) == _cam_info_k_values(
                reader_cam_info
            )

            writer_cam_intrinsic = writer_pipeline.cas.get(CASViews.CAM_INTRINSIC)
            reader_cam_intrinsic = reader_pipeline.cas.get(CASViews.CAM_INTRINSIC)
            assert writer_cam_intrinsic.width == reader_cam_intrinsic.width
            assert writer_cam_intrinsic.height == reader_cam_intrinsic.height
            np.testing.assert_allclose(
                writer_cam_intrinsic.intrinsic_matrix,
                reader_cam_intrinsic.intrinsic_matrix,
            )

            assert writer_pipeline.cas.get(CASViews.COLOR2DEPTH_RATIO) == (
                reader_pipeline.cas.get(CASViews.COLOR2DEPTH_RATIO)
            )
        finally:
            writer_node.destroy_node()
            reader_node.destroy_node()
            storage.drop_database()
