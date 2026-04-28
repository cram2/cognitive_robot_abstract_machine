import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3, Quaternion
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, Set
from .data_player import FilePlayer, FrameData

logger = logging.getLogger(__name__)

@dataclass(unsafe_hash=True, init=False)
class CSVEpisodePlayer(FilePlayer):
    """
    This class is used to play a CSV file containing the data of an episode.
    """

    data_frames: pd.DataFrame = field(default=None)
    """
    The data frames of the episode.
    """

    data_object_names: Set[str] = field(default=None)
    """
    The name of the objects in the episode.
    """

    def get_frame_data_generator(self):
        """
        Reads the CSV file and generates the frame data.
        """
        logger.debug(f"Reading CSV file {self.file_path}")
        self.data_frames = pd.read_csv(self.file_path)
        self.data_object_names = {
            v.split(":")[0] for v in self.data_frames.columns if ":" in v
        }
        for i, (frame_id, objects_data) in enumerate(self.data_frames.iterrows()):
            yield FrameData(
                time=float(objects_data["time"]),
                objects_data=objects_data.to_dict(),
                frame_idx=i,
            )

    def get_joint_states(self, frame_data: FrameData) -> Dict[str, float]:
        return {}

    def _pause(self): ...

    def _resume(self): ...

    def get_objects_poses(self, frame_data: FrameData) -> Dict[Body, Pose]:
        """
        Extracts the poses of the objects from the frame data.

        :param frame_data: The frame data.
        :return: A dictionary mapping bodies to poses.
        """
        objects_poses: Dict[Body, Pose] = {}
        objects_data = frame_data.objects_data
        current_time = frame_data.time
        for obj_name in self.data_object_names:
            obj_position = [objects_data[f"{obj_name}:position_{i}"] for i in range(3)]
            obj_orientation = [
                objects_data[f"{obj_name}:quaternion_{i}"] for i in range(4)
            ]
            obj_orientation[0], obj_orientation[3] = (
                obj_orientation[3],
                obj_orientation[0],
            )
            # Transform object quaternion to euler
            obj_pose = Pose.from_xyz_quaternion(
                pos_x=obj_position[0],
                pos_y=obj_position[1],
                pos_z=obj_position[2],
                quat_x=obj_orientation[0],
                quat_y=obj_orientation[1],
                quat_z=obj_orientation[2],
                quat_w=obj_orientation[3],
            )
            obj_pose.timestamp = current_time

            if self.position_shift:
                obj_pose.x += self.position_shift.x
                obj_pose.y += self.position_shift.y
                obj_pose.z += self.position_shift.z

            obj = self.world.get_body_by_name(obj_name)
            objects_poses[obj] = obj_pose
        return objects_poses
