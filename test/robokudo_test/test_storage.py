import copy
import os
import uuid
from dataclasses import dataclass

import open3d as o3d
import pytest
import pymongo
import numpy as np

import robokudo.cas
from robokudo import world as rk_world
from robokudo.cas import CAS, CASViews
from robokudo.io.storage import Storage
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@dataclass
class CustomDataclassView:
    label: str
    confidence: float


pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="module temporarily disabled until storage functionality is migrated to ormatic",
)


@pytest.fixture(scope="module")
def storage_instance():
    db_name = "ONLY_UNITTESTS_test_db"
    storage = Storage(db_name)
    yield storage
    # Cleanup after tests
    storage.drop_database()


@pytest.fixture()
def cas_data():
    cas = CAS()
    """
    Mock CAS data for testing.
    """
    cas.set(CASViews.COLOR_IMAGE, np.random.rand(100, 100, 3))
    cas.set(CASViews.DEPTH_IMAGE, np.random.rand(100, 100))

    return cas


def store_cas_in_storage(storage_instance, cas_data):
    flat_cas = storage_instance.generate_dict_from_real_cas(cas_data)
    flat_cas["view_ids"] = {}

    # step 1: persist each view
    storage_instance.store_views_in_mongo(flat_cas)

    # step 2: persist cas and pointers to each view
    del flat_cas["views"]
    return storage_instance.store_cas_dict(flat_cas)


def create_point_cloud():
    """
    Create an Open3D point cloud with some sample data.
    """
    # Create a numpy array of points
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Create the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud


def create_camera_intrinsic():
    """
    Create an Open3D pinhole camera intrinsic for testing.
    """
    return o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 526.0, 319.5, 239.5)


class TestStorage:
    def test_drop_database(self, storage_instance):
        db_name = storage_instance.db_name
        storage_instance.drop_database()

        # Verify the database no longer exists

        # Fetch environment variables which might have been used to configure
        # another MongoDB Host and Port
        # This was originally introduced to support unit tests.
        mongo_host = os.getenv("RK_MONGO_HOST", "localhost")
        mongo_port = int(os.getenv("RK_MONGO_PORT", 27017))
        client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
        assert db_name not in client.list_database_names()

    def test_store_and_retrieve_cas(self, storage_instance, cas_data):
        result = store_cas_in_storage(storage_instance, cas_data)
        # Generate a dict from the CAS and store it
        flat_cas = storage_instance.generate_dict_from_real_cas(cas_data)
        flat_cas["view_ids"] = {}

        # step 1: persist each view
        storage_instance.store_views_in_mongo(flat_cas)

        # step 2: persist cas and pointers to each view
        del flat_cas["views"]
        result = storage_instance.store_cas_dict(flat_cas)

        assert result.acknowledged

        # Retrieve the stored CAS
        retrieved_cas = storage_instance.db.cas.find_one({"_id": result.inserted_id})
        assert retrieved_cas is not None
        assert retrieved_cas["timestamp"] == cas_data.timestamp
        assert retrieved_cas["timestamp_readable"] == cas_data.timestamp_readable

    def test_store_and_retrieve_cas_with_single_oh(self, storage_instance, cas_data):
        # Generate a dict from the CAS and store it
        oh = ObjectHypothesis()
        oh.source = "test_storage"
        oh.id = 1
        oh.roi = ImageROI()
        oh.roi.roi.pos.x = 1
        oh.roi.roi.pos.y = 2
        oh.roi.roi.width = 20
        oh.roi.roi.height = 40
        oh.points = create_point_cloud()

        cas_data.annotations.append(oh)

        result = store_cas_in_storage(storage_instance, cas_data)

        flat_cas = storage_instance.generate_dict_from_real_cas(cas_data)
        flat_cas["view_ids"] = {}

        # step 1: persist each view
        storage_instance.store_views_in_mongo(flat_cas)

        # step 2: persist cas and pointers to each view
        del flat_cas["views"]
        result = storage_instance.store_cas_dict(flat_cas)

        assert result.acknowledged

        # Retrieve the stored CAS
        retrieved_cas_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_cas_record is not None
        assert retrieved_cas_record["timestamp"] == cas_data.timestamp
        assert retrieved_cas_record["timestamp_readable"] == cas_data.timestamp_readable

        retrieved_cas = robokudo.cas.CAS()
        retrieved_cas_record["views"] = {}
        storage_instance.load_views_from_mongo_in_cas(retrieved_cas_record)

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in retrieved_cas_record["views"].items():
            retrieved_cas.set(view_name, view_content)

        storage_instance.load_annotations_from_mongo_in_cas(
            retrieved_cas_record, retrieved_cas
        )

        # Compare the most important views: COLOR and DEPTH Image
        assert np.array_equal(
            cas_data.get(CASViews.COLOR_IMAGE), retrieved_cas.get(CASViews.COLOR_IMAGE)
        )
        assert np.array_equal(
            cas_data.get(CASViews.DEPTH_IMAGE), retrieved_cas.get(CASViews.DEPTH_IMAGE)
        )

        assert len(retrieved_cas.annotations) == 1
        retrieved_oh = retrieved_cas.annotations[0]

        assert oh.source == retrieved_oh.source
        assert oh.id == retrieved_oh.id
        assert oh.roi.roi.pos.x == retrieved_oh.roi.roi.pos.x
        assert oh.roi.roi.pos.y == retrieved_oh.roi.roi.pos.y
        assert oh.roi.roi.width == retrieved_oh.roi.roi.width
        assert oh.roi.roi.height == retrieved_oh.roi.roi.height

        # Compare PointCloud
        test_pcd = create_point_cloud()

        np.testing.assert_array_equal(
            np.asarray(test_pcd.points), np.asarray(retrieved_oh.points.points)
        )

    def test_annotations_are_serialized_with_krrood(self, storage_instance, cas_data):
        oh = ObjectHypothesis()
        oh.source = "test_storage"
        oh.id = 1
        cas_data.annotations.append(oh)

        result = store_cas_in_storage(storage_instance, cas_data)
        assert result.acknowledged

        retrieved_cas_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_cas_record is not None
        assert retrieved_cas_record["annotations_format"] == "krrood_v1"
        assert isinstance(retrieved_cas_record["annotations"], list)
        assert "__json_type__" in retrieved_cas_record["annotations"][0]

    def test_store_and_retrieve_dynamic_extra_views(self, storage_instance, cas_data):
        cas_data.set("custom_json", {"labels": ["cup", "plate"], "valid": True})
        cas_data.set("custom_tuple", (1.2, 3.4))
        cas_data.set("custom_numpy", np.array([[1, 2], [3, 4]], dtype=np.int16))
        cas_data.set("custom_cloud", create_point_cloud())
        cas_data.set("pc_cam_intrinsic", create_camera_intrinsic())

        result = store_cas_in_storage(storage_instance, cas_data)
        assert result.acknowledged

        retrieved_cas_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_cas_record is not None
        assert "extra_views" not in retrieved_cas_record
        assert "view_ids" in retrieved_cas_record
        assert len(retrieved_cas_record["view_ids"]) >= 6

        retrieved_cas_record["views"] = {}
        storage_instance.load_views_from_mongo_in_cas(retrieved_cas_record)

        assert retrieved_cas_record["views"]["custom_json"] == {
            "labels": ["cup", "plate"],
            "valid": True,
        }
        assert retrieved_cas_record["views"]["custom_tuple"] == (1.2, 3.4)
        np.testing.assert_array_equal(
            retrieved_cas_record["views"]["custom_numpy"],
            np.array([[1, 2], [3, 4]], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            np.asarray(retrieved_cas_record["views"]["custom_cloud"].points),
            np.asarray(create_point_cloud().points),
        )
        restored_intrinsic = retrieved_cas_record["views"]["pc_cam_intrinsic"]
        expected_intrinsic = create_camera_intrinsic()
        assert restored_intrinsic.width == expected_intrinsic.width
        assert restored_intrinsic.height == expected_intrinsic.height
        np.testing.assert_allclose(
            np.asarray(restored_intrinsic.intrinsic_matrix),
            np.asarray(expected_intrinsic.intrinsic_matrix),
        )

    def test_extra_views_krrood_fallback_for_dataclass(
        self, storage_instance, cas_data
    ):
        cas_data.set(
            "custom_dataclass", CustomDataclassView(label="cup", confidence=0.9)
        )

        result = store_cas_in_storage(storage_instance, cas_data)
        assert result.acknowledged

        retrieved_cas_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_cas_record is not None

        retrieved_cas_record["views"] = {}
        storage_instance.load_views_from_mongo_in_cas(retrieved_cas_record)

        restored = retrieved_cas_record["views"]["custom_dataclass"]
        assert isinstance(restored, CustomDataclassView)
        assert restored.label == "cup"
        assert restored.confidence == 0.9

    def test_unknown_dynamic_view_type_raises(self):
        class UnsupportedView:
            pass

        cas = CAS()
        cas.set(CASViews.COLOR_IMAGE, np.random.rand(10, 10, 3))
        cas.set(CASViews.DEPTH_IMAGE, np.random.rand(10, 10))
        cas.set("custom_unsupported", UnsupportedView())

        with pytest.raises(TypeError):
            Storage.generate_dict_from_real_cas(cas)

    def test_cam_to_world_transform_roundtrip_preserves_world_references(
        self, storage_instance, cas_data
    ):
        world_frame = f"map_{uuid.uuid4().hex[:8]}"
        camera_frame = f"camera_{uuid.uuid4().hex[:8]}"

        rk_world.init_world_with_entity_tracker()
        rk_world.setup_world_for_camera_frame(
            world_frame=world_frame, camera_frame=camera_frame
        )
        sem_world = rk_world.world_instance()
        camera_body = sem_world.get_body_by_name(camera_frame)
        world_body = sem_world.get_body_by_name(world_frame)

        transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.1,
            pos_y=0.2,
            pos_z=0.3,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
            child_frame=camera_body,
            reference_frame=world_body,
        )
        cas_data.set(CASViews.CAM_TO_WORLD_TRANSFORM, transform)

        result = store_cas_in_storage(storage_instance, cas_data)
        assert result.acknowledged

        retrieved_cas_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_cas_record is not None

        retrieved_cas_record["views"] = {}
        storage_instance.load_views_from_mongo_in_cas(retrieved_cas_record)
        restored_transform = retrieved_cas_record["views"][
            CASViews.CAM_TO_WORLD_TRANSFORM
        ]

        assert isinstance(restored_transform, HomogeneousTransformationMatrix)
        np.testing.assert_allclose(restored_transform.to_np(), transform.to_np())
        assert restored_transform.child_frame is not None
        assert restored_transform.reference_frame is not None
        assert str(restored_transform.child_frame.name) == str(camera_body.name)
        assert str(restored_transform.reference_frame.name) == str(world_body.name)
