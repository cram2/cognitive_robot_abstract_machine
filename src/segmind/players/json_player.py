import datetime
import json
import os
import shutil
import time

import numpy as np
import scipy
import trimesh
from pycram.tf_transformations import euler_matrix, quaternion_from_matrix, quaternion_matrix, quaternion_from_euler, \
    euler_from_quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from trimesh import Geometry, Trimesh
from typing_extensions import Type, List, Tuple, Union, Dict, Optional

from pycram.datastructures.pose import Header, PoseStamped, Transform
from ..episode_player import EpisodePlayer
from .data_player import FilePlayer, FrameData
from ..utils import calculate_quaternion_difference


class JSONPlayer(FilePlayer):
    def __init__(self, file_path: str, scene_id: int = 1, world: Optional[World] = None,
                 mesh_scale: float = 0.001,
                 time_between_frames: datetime.timedelta = datetime.timedelta(milliseconds=50),
                 objects_to_ignore: Optional[List[int]] = None,
                 obj_id_to_name: Optional[Dict[int, str]] = None,
                 obj_id_to_type: Optional[Dict[int, Type[Body]]] = None):
        """
        Initializes the FAMEEpisodePlayer with the specified json file and scene id.

        :param file_path: The json file that contains the data frames.
        :param scene_id: The scene id.
        :param world: The world that is used to replay the episode.
        :param mesh_scale: The scale of the mesh.
        :param time_between_frames: The time between frames.
        :param objects_to_ignore: A list of object ids to ignore.
        """
        self.objects_to_ignore: Optional[List[int]] = objects_to_ignore
        self.scene_id = scene_id

        super().__init__(time_between_frames=time_between_frames, world=world, stop_after_ready=False, use_realtime=False,
                         file_path=file_path)
        
        self.mesh_scale = mesh_scale
        self.obj_id_to_name: Optional[Dict[int, str]] = obj_id_to_name
        self.obj_id_to_type: Optional[Dict[int, Type[PhysicalObject]]] = obj_id_to_type
        self.object_meshes: Dict[Object, Geometry] = {}
        self.correction_quaternions: Dict[Object, np.ndarray] = {}
        self.base_origin_of_objects: Dict[Object, np.ndarray] = {}
        self.average_rotation_correction_matrix: Optional[np.ndarray] = None

    def get_frame_data_generator(self):
        with open(self.file_path, 'r') as f:
            self.data_frames = json.load(f)[str(self.scene_id)]
        self.data_frames = {int(frame_id): objects_data for frame_id, objects_data in self.data_frames.items()}
        if self.objects_to_ignore is not None:
            self._remove_ignored_objects(self.objects_to_ignore)
        self.data_frames = dict(sorted(self.data_frames.items(), key=lambda x: x[0]))
        for i, (frame_id, objects_data) in enumerate(self.data_frames.items()):
            yield FrameData(i * self.time_between_frames.total_seconds(), objects_data, frame_idx=i)

    def _remove_ignored_objects(self, objects_to_ignore: List[int]):
        """
        Remove the objects to ignore from the data frames.

        :param objects_to_ignore: The list of object ids to ignore.
        """
        self.data_frames = {frame_id: {obj_id: v for obj_id, v in objects_data.items()
                                       if int(obj_id) not in objects_to_ignore}
                            for frame_id, objects_data in self.data_frames.items()}

    def _pause(self):
        ...

    def _resume(self):
        ...

    def get_objects_poses(self, frame_data: FrameData) -> Dict[Body, Pose]:
        objects_poses: Dict[Body, Pose] = {}
        objects_data = frame_data.objects_data
        for object_id, object_poses_data in objects_data.items():

            # Get the object pose in the map frame
            pose = self.get_pose_and_transform_to_map_frame(object_poses_data[0])

            # Get the object and mesh names
            obj_name = self.get_object_name(int(object_id))
            obj_type = self.get_object_type(int(object_id))
            mesh_name = self.get_mesh_name(object_id)

            # Create the object if it does not exist in the world and set its pose
            if obj_name not in self.world.get_object_names():
                obj = Object(obj_name, obj_type, mesh_name,
                             pose=PoseStamped(Pose(pose.position)), scale_mesh=self.mesh_scale)
                quat_diff = calculate_quaternion_difference(pose.orientation.to_list(), [0, 0, 0, 1])
                euler_diff = euler_from_quaternion(quat_diff)
                quat_diff = quaternion_from_euler(euler_diff[0], euler_diff[1], 0)
                self.correction_quaternions[obj] = quat_diff
            else:
                obj = self.world.get_object_by_name(obj_name)
                objects_poses[obj] = self.apply_orientation_correction_to_object_pose(obj, pose)
        return objects_poses

    def apply_orientation_correction_to_object_pose(self, obj: Body, obj_pose: PoseStamped) -> Pose:
        """
        Correct the orientation of an object based on the orientation of the mesh, and return the corrected pose.

        :param obj: The object.
        :param obj_pose: The pose of the object.
        """
        if self.average_rotation_correction_matrix is None:
            self.average_rotation_correction_matrix = self.calculate_average_rotation()
        homogeneous_pose = obj_pose.to_transform_stamped(obj.tf_frame).transform.to_matrix()
        new_pose_transform = np.dot(self.average_rotation_correction_matrix, homogeneous_pose)
        new_quaternion = quaternion_from_matrix(new_pose_transform)
        new_translation = new_pose_transform[:3, 3].tolist()
        return PoseStamped(Pose(Vector3(*new_translation)), Quaternion(*new_quaternion))

    def calculate_average_rotation(self):
        """
        Calculate the average rotation of the correction rotations. (This is based on this stackoverflow post:
        https://stackoverflow.com/a/51572039)
        """
        mean_rotation = np.mean([quaternion_matrix(v) for v in self.correction_quaternions.values()], axis=0)
        u, s, vh = np.linalg.svd(mean_rotation)
        return np.dot(u, vh)

    def estimate_object_mesh_orientation(self, obj: Body) -> np.ndarray:
        """
        Estimate the rotation of an object based on the orientation of a plane that is fitted to the base of the mesh
         of the object.

        :param obj: The object.
        :return: The rotation matrix of the base plane.
        """
        base_vertices = self.get_base_points_of_object(obj)
        plane_normal = self.estimate_plane_normal_from_points(base_vertices)
        rotation_matrix = self.calculate_plane_rotation_from_normal(plane_normal)
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation_matrix
        return quaternion_from_matrix(homogeneous_matrix)

    def get_base_points_of_object(self, obj: Body, transform_points: bool = True) -> np.ndarray:
        """
        Get the base points of an object.

        :param obj: The object.
        :param transform_points: A boolean value that determines if the points should be transformed.
        :return: The base points of the object.
        """
        mesh = self.get_mesh_of_object(obj, apply_transform=False)
        base_points = self.get_base_points_from_mesh(mesh)
        if transform_points:
            base_points = np.dot(base_points, quaternion_matrix(obj.get_orientation.to_list())[:3, :3].T)
        return base_points

    def get_relative_base_origin_of_object(self, obj: Body) -> np.ndarray:
        """
        Get the origin of the base of the object relative to the origin of the object.

        :param obj: The object.
        :return: The origin of the base.
        """
        mesh = self.get_mesh_of_object(obj, apply_transform=False)
        base_vertices, min_z = self.get_base_points_from_mesh(mesh, return_min_z=True)
        base_origin = base_vertices.mean(axis=0)
        base_origin[2] = min_z
        return base_origin

    def get_mesh_of_object(self, obj: Body, apply_scale: bool = True,
                           apply_transform: bool = True) -> Trimesh:
        """
        Get the mesh of an object.
W
        :param obj: The object.
        :param apply_scale: A boolean value that determines if the mesh should be scaled.
        :param apply_transform: A boolean value that determines if the mesh should be transformed.
        :return: The mesh of the object.
        """
        return self.object_meshes.get(obj,
                                      self.load_scale_and_transform_mesh_of_object(obj, apply_scale, apply_transform))

    def load_scale_and_transform_mesh_of_object(self, obj: Body, apply_scale: bool = True,
                                                apply_transform: bool = True) -> Geometry:
        """
        Load, scale and transform the mesh of an object.

        :param obj: The object.
        :param apply_scale: A boolean value that determines if the mesh should be scaled.
        :param apply_transform: A boolean value that determines if the mesh should be transformed.
        :return: The loaded and processed mesh.
        """
        mesh_path = obj.description.original_path
        mesh = trimesh.load(mesh_path)
        if apply_scale:
            mesh.apply_scale(self.mesh_scale)
        if apply_transform:
            mesh_transform = quaternion_matrix(obj.get_orientation.to_list())
            mesh.apply_transform(mesh_transform)
        return mesh

    @staticmethod
    def get_base_points_from_mesh(mesh: Trimesh, threshold: float = 0.001,
                                  return_min_z: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Get the base points of an object from its mesh.

        :param mesh: The mesh of the object.
        :param threshold: The threshold to determine the base points.
        :param return_min_z: A boolean value that determines if the minimum z value should be returned.
        :return: The base points of the object and the minimum z value if return_min_z is True.
        """
        min_z = np.min(mesh.vertices[:, 2])
        base_vertices = mesh.vertices[mesh.vertices[:, 2] <= threshold + min_z]
        if return_min_z:
            return base_vertices, min_z
        else:
            return base_vertices

    @staticmethod
    def estimate_plane_normal_from_points(base_vertices: np.ndarray) -> np.ndarray:
        """
        Estimate the plane from a set of points.

        :param base_vertices: The base vertices.
        :return: The rotation matrix of the plane.
        """
        # barycenter of the points
        # compute centered coordinates
        G = base_vertices.mean(axis=0)

        # run SVD
        u, s, vh = np.linalg.svd(base_vertices - G)

        return vh[2, :]

    @staticmethod
    def calculate_plane_rotation_from_normal(u_norm: np.ndarray) -> np.ndarray:
        """
        Calculate the rotation matrix between two vectors.

        :param u_norm: The normal vector.
        """

        u_norm = np.expand_dims(u_norm, 0)
        vertical_axis = np.expand_dims(np.array([0, 0, 1]), 0)

        # get rotation matrix to align the normal vector with the vertical axis
        rotation_matrix = scipy.spatial.transform.Rotation.align_vectors(u_norm, vertical_axis)[0].as_matrix()

        return rotation_matrix

    def get_pose_and_transform_to_map_frame(self, object_pose: dict) -> Optional[PoseStamped]:
        """
        Get the pose of an object and transform it to the map frame.

        :param object_pose: The pose of the object.
        :return: The pose of the object in the map frame.
        """
        position, quaternion = self.get_pose_from_frame_object_data(object_pose)
        return self.transform_pose_to_map_frame(position, quaternion)

    @staticmethod
    def get_pose_from_frame_object_data(object_data: dict) -> Tuple[List[float], List[float]]:
        """
        Get the pose from the frame object data that is extracted from the data file.

        :param object_data: The object data.
        :return: The position and quaternion of the object.
        """
        position = np.array(list(map(float, object_data['t']))) / 1000  # Convert from mm to m
        position = position.tolist()
        rotation = np.array(list(map(float, object_data['R']))).reshape(3, 3)
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation
        quaternion = quaternion_from_matrix(homogeneous_matrix)
        return position, quaternion

    def transform_pose_to_map_frame(self, position: List[float], quaternion: List[float]) -> Optional[PoseStamped]:
        """
        Transform the pose of an object to the map frame.

        :param position: The position of the object.
        :param quaternion: The quaternion of the object.
        """
        new_transform = TransformStamped(pose=Pose(Vector3(0, 0, 1),
                                  Quaternion(*quaternion_from_matrix(euler_matrix(-np.pi / 2, 0, 0)))),
                                  child_frame_id=self.camera_frame_name)
        self.world.local_transformer.update_transforms([new_transform])
        pose = PoseStamped(Pose(Vector3(*position), Quaternion(*quaternion)), header=Header(frame_id=self.camera_frame_name))
        return self.world.local_transformer.transform_pose(pose, "map")

    @property
    def camera_frame_name(self) -> str:
        return "episode_camera_frame"

    def get_object_name(self, object_id: int) -> str:
        if self.obj_id_to_name is not None and object_id in self.obj_id_to_name:
            return self.obj_id_to_name[object_id]
        else:
            return f"object_{object_id}"

    def get_object_type(self, object_id: int) -> Type[Body]:
        if self.obj_id_to_type is not None and object_id in self.obj_id_to_type:
            return self.obj_id_to_type[int(object_id)]
        else:
            return Body

    @staticmethod
    def get_mesh_name(object_id: str) -> str:
        return f"obj_{object_id:0>6}.ply"

    def _join(self, timeout=None):
        pass
