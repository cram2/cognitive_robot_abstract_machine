from __future__ import annotations

from robokudo.cas import CAS, CASViews
from robokudo.io.camera_placement import CameraPlacement, MeasuredCameraPose
from robokudo.io.storage import Storage
from robokudo.io.storage_reader_interface import StorageReaderInterface

from ._perception_world import FrameName, SharedWorld, camera_pose

# %% recorded camera pose


class TestRecordedCameraPose:
    def test_camera_recorded_in_a_shared_world_is_replayed_under_its_frame(
        self, shared_world: SharedWorld
    ) -> None:
        cas = CAS()
        measured_pose = MeasuredCameraPose(
            world_frame=FrameName.WORLD,
            camera_frame=FrameName.CAMERA,
            world_T_camera=camera_pose(x=1.0),
        )
        CameraPlacement(shared_world.world).place(cas, measured_pose)
        view_document = Storage.encode_view_document(
            view_name=CASViews.CAMERA_TO_WORLD_TRANSFORM,
            view_value=cas.camera_to_world_transform,
        )
        reader = StorageReaderInterface.__new__(StorageReaderInterface)

        recorded_pose = reader._decode_recorded_camera_pose(
            view_document, Storage.generate_dict_from_real_cas(cas)
        )

        assert recorded_pose.camera_frame == measured_pose.camera_frame
