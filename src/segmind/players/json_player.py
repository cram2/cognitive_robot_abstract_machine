import datetime
import json
import os
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import trimesh
from trimesh import Geometry

from segmind import logger, set_logger_level, LogLevel
from segmind.players.data_player import FilePlayer, FrameData, FrameDataGenerator
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

set_logger_level(LogLevel.DEBUG)

class JSONPlayer(FilePlayer):
    data_object_names: Set[str]


    def __init__(self, file_path: str, scene_id: int = 1, world: Optional[World] = None,
                 mesh_scale: float = 0.001,
                 time_between_frames: datetime.timedelta = datetime.timedelta(milliseconds=50),
                 obj_id_to_name: Optional[Dict[int, str]] = None):
        """
        Initializes the FAMEEpisodePlayer with the specified json file and scene id.

        :param file_path: The json file that contains the data frames.
        :param scene_id: The scene id.
        :param world: The world that is used to replay the episode.
        :param mesh_scale: The scale of the mesh.
        :param time_between_frames: The time between frames.
        :param objects_to_ignore: A list of object ids to ignore.
        """


        self.scene_id = scene_id

        super().__init__(time_between_frames=time_between_frames, world=world, stop_after_ready=False,
                         use_realtime=False,
                         file_path=file_path)

        self.mesh_scale = mesh_scale
        self.obj_id_to_name: Optional[Dict[int, str]] = obj_id_to_name
        self.object_meshes: Dict[Body, Geometry] = {}
        self.correction_quaternions: Dict[Body, np.ndarray] = {}
        self.base_origin_of_objects: Dict[Body, np.ndarray] = {}
        self.average_rotation_correction_matrix: Optional[np.ndarray] = None

    def get_frame_data_generator(self):
        with open(self.file_path, 'r') as f:
            self.data_frames = json.load(f)[str(self.scene_id)]
        self.data_frames = {int(frame_id): objects_data for frame_id, objects_data in self.data_frames.items()}
        self.data_frames = dict(sorted(self.data_frames.items(), key=lambda x: x[0]))
        for i, (frame_id, objects_data) in enumerate(self.data_frames.items()):
            yield FrameData(i * self.time_between_frames.total_seconds(), objects_data, frame_idx=i)


    def _pause(self): ...

    def _resume(self): ...

    def get_objects_poses(self, frame_data: FrameData) -> Dict[Body, Pose]:
        objects_data = frame_data.objects_data
        objects_poses: Dict[Body, Pose] = {}
        for obj_name, obj_data in objects_data.items():
            if obj_name == "5" or obj_name == "2":
                continue
            for det in obj_data:
                R = det["R"]
                t = det["t"]
                r = np.array(R).reshape(3, 3)
                q = self.rotation_matrix_to_quaternion(r)
                q = q / np.linalg.norm(q)
                obj_pose = Pose.from_xyz_quaternion(
                    pos_x=t[0],
                    pos_y=t[1],
                    pos_z=t[2],
                    quat_x=q[0],
                    quat_y=q[1],
                    quat_z=q[2],
                    quat_w=q[3],
                )
                obj_pose.timestamp = det["time"]
                body_name = self.obj_id_to_name[int(obj_name)]
                body = self.world.get_body_by_name(body_name)
                objects_poses[body] = obj_pose

        return objects_poses

    def rotation_matrix_to_quaternion(self, R):
        """
        Transformation of a rotation matrix into a quaternion.

        :param R: rotation matrix
        :return: quaternion
        """
        q = np.zeros(4)
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[0] = 0.25 / s
            q[1] = (R[2, 1] - R[1, 2]) * s
            q[2] = (R[0, 2] - R[2, 0]) * s
            q[3] = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                q[0] = (R[2, 1] - R[1, 2]) / s
                q[1] = 0.25 * s
                q[2] = (R[0, 1] + R[1, 0]) / s
                q[3] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                q[0] = (R[0, 2] - R[2, 0]) / s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = 0.25 * s
                q[3] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                q[0] = (R[1, 0] - R[0, 1]) / s
                q[1] = (R[0, 2] + R[2, 0]) / s
                q[2] = (R[1, 2] + R[2, 1]) / s
                q[3] = 0.25 * s

        q_xyzw=[q[1], q[2], q[3], q[0]]
        return q_xyzw


    def get_name_from_id(self, obj_id: int) -> str:
        if self.obj_id_to_name and obj_id in self.obj_id_to_name:
            return self.obj_id_to_name[obj_id]
        return str(obj_id)

    def get_body_from_name(self, obj_name):
        return self.world.get_body_by_name(obj_name)


    def transform_to_stl(self, path: str):
        for filename in os.listdir(path):
            if filename.lower().endswith(".ply"):
                ply_path = os.path.join(path, filename)
                stl_path = os.path.join(
                    path,
                    os.path.splitext(filename)[0] + ".stl"
                )

                try:
                    mesh = trimesh.load(ply_path)
                    mesh.export(stl_path)
                    print(f"Converted: {filename} → {os.path.basename(stl_path)}")
                except Exception as e:
                    print(f"Failed: {filename} ({e})")