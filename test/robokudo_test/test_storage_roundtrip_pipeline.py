from __future__ import annotations

# %% Imports
import os
from pathlib import Path
import json
import uuid
import pytest

import numpy as np
import py_trees
from rclpy.node import Node

from robokudo.types.cv import ColorToDepthRatio
import robokudo.defs
from robokudo import world as rk_world
import robokudo.descriptors.camera_configs.config_filereader_playback
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.utils.data_downloader
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.storage import StorageWriter
from robokudo.cas import CASViews
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.io.storage import Storage, StorageDocumentField
import robokudo.utils.tree_execution
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from robokudo.behaviours.clear_annotator_outputs import ClearAnnotatorOutputs
from robokudo.pipeline import Pipeline
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="module temporarily disabled until storage functionality is migrated to ormatic",
)


def _build_writer_pipeline(db_name: str) -> Pipeline:
    file_reader_descriptor = CollectionReaderDescriptorFactory.create_descriptor(
        "file_reader",
        loop=False,
        target_dir=robokudo.utils.data_downloader.test_data_path() / Path("data"),
        color_to_depth_ratio=ColorToDepthRatio(horizontal=0.5, vertical=0.5),
    )
    camera_config = file_reader_descriptor.parameters.camera_config
    camera_config.static_camera_transform_enabled = True
    camera_config.static_world_T_camera = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.1,
            pos_y=0.2,
            pos_z=0.3,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
        )
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
    mongo_descriptor = CollectionReaderDescriptorFactory.create_descriptor(
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
    def test_store_and_replay_sensor_data_roundtrip(self) -> None:
        rk_world.init_world_with_entity_tracker()
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

            query_document = Storage.encode_view_document(
                CASViews.QUERY, {"recorded": True}
            )
            query_id = (
                storage.db[Storage.VIEW_COLLECTION_NAME]
                .insert_one(query_document)
                .inserted_id
            )
            view_ids = dict(stored_record[StorageDocumentField.VIEW_IDS])
            view_ids[CASViews.QUERY] = query_id
            storage.db.cas.update_one(
                {"_id": stored_record["_id"]},
                {"$set": {StorageDocumentField.VIEW_IDS: view_ids}},
            )

            rk_world.init_world_with_entity_tracker()
            assert rk_world.world_instance().is_empty()

            reader_pipeline = _build_reader_pipeline(db_name)
            reader_status = robokudo.utils.tree_execution.run_tree_once(
                reader_pipeline, reader_node
            )
            assert reader_status is py_trees.common.Status.SUCCESS
            assert not reader_pipeline.cas.contains(CASViews.QUERY)

            np.testing.assert_array_equal(
                writer_pipeline.cas.get(CASViews.COLOR_IMAGE),
                reader_pipeline.cas.get(CASViews.COLOR_IMAGE),
            )
            np.testing.assert_allclose(
                writer_pipeline.cas.get(CASViews.DEPTH_IMAGE),
                reader_pipeline.cas.get(CASViews.DEPTH_IMAGE),
                equal_nan=True,
            )

            writer_observation = writer_pipeline.cas.require_camera_observation()
            reader_observation = reader_pipeline.cas.require_camera_observation()
            assert reader_observation.camera is not writer_observation.camera
            assert (
                reader_observation.camera
                is rk_world.world_instance().get_semantic_annotation_by_id(
                    reader_observation.camera.id
                )
            )
            assert reader_observation.camera.id == writer_observation.camera.id
            assert (
                reader_observation.camera.root.id == writer_observation.camera.root.id
            )
            assert reader_observation.camera.name == writer_observation.camera.name
            assert (
                reader_observation.camera.camera_model
                == writer_observation.camera.camera_model
            )
            assert (
                reader_observation.camera.camera_range
                == writer_observation.camera.camera_range
            )
            assert tuple(reader_observation.camera.modalities) == tuple(
                writer_observation.camera.modalities
            )
            assert {
                str(key) for key in rk_world.world_instance().state.keys()
            }.isdisjoint({str(key) for key in world_snapshot_payload["state"]["ids"]})
            assert (
                writer_observation.effective_camera_model
                == reader_observation.effective_camera_model
            )
            assert (
                writer_observation.timestamp_nanoseconds
                == reader_observation.timestamp_nanoseconds
            )
            reader_pose = reader_observation.world_T_camera_or_raise()
            writer_pose = writer_observation.world_T_camera_or_raise()
            np.testing.assert_array_equal(reader_pose.to_np(), writer_pose.to_np())
            assert reader_pose.reference_frame.id == writer_pose.reference_frame.id
            assert reader_pose.child_frame is reader_observation.camera.root
            assert (
                reader_observation.camera.root.parent_connection.parent
                is reader_pose.reference_frame
            )

            assert writer_pipeline.cas.get(CASViews.COLOR_TO_DEPTH_RATIO) == (
                reader_pipeline.cas.get(CASViews.COLOR_TO_DEPTH_RATIO)
            )
        finally:
            writer_node.destroy_node()
            reader_node.destroy_node()
            storage.drop_database()
