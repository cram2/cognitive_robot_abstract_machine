import logging
from dataclasses import dataclass, field
import pandas as pd

from segmind.players.data_player import FilePlayer, FrameData
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, Set

logger = logging.getLogger(__name__)

@dataclass(eq=False)
class CSVEpisodePlayer(FilePlayer):
    """
    Plays the episode from a CSV file.
    """

    data_frames: pd.DataFrame = field(default=None, init=False, hash=False, compare=False)
    """
    The data frames of the CSV file.
    """

    data_object_names: Set[str] = field(default=None, init=False, hash=False, compare=False)
    """
    The names of the objects in the CSV file.
    """

    def __post_init__(self):
        super().__post_init__()

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

    def _pause(self):
        """
        Pauses the episode player.
        """

    def _resume(self):
        """
        Resumes the episode player.
        """

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
