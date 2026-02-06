from datetime import datetime

import pandas as pd
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3, Quaternion
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, Set

from pycram.datastructures.pose import PoseStamped, Header
from .data_player import FilePlayer, FrameData


class CSVEpisodePlayer(FilePlayer):
    data_frames: pd.DataFrame
    data_object_names: Set[str]

    def get_frame_data_generator(self):
        self.data_frames = pd.read_csv(self.file_path)
        self.data_object_names = {v.split(':')[0] for v in self.data_frames.columns if ':' in v}
        for i, (frame_id, objects_data) in enumerate(self.data_frames.iterrows()):
            yield FrameData(time=float(objects_data["time"]), objects_data=objects_data.to_dict(), frame_idx=i)

    def get_joint_states(self, frame_data: FrameData) -> Dict[str, float]:
        return {}

    def _pause(self):
        ...

    def _resume(self):
        ...

    def get_objects_poses(self, frame_data: FrameData) -> Dict[Body, Pose]:
        objects_poses: Dict[Body, Pose] = {}
        objects_data = frame_data.objects_data
        current_time = frame_data.time
        for obj_name in self.data_object_names:
            obj_position = [objects_data[f"{obj_name}:position_{i}"] for i in range(3)]
            obj_orientation = [objects_data[f"{obj_name}:quaternion_{i}"] for i in range(4)]
            obj_orientation[0], obj_orientation[3] = obj_orientation[3], obj_orientation[0]
            obj_pose = PoseStamped(Pose(Vector3(*obj_position), Quaternion(*obj_orientation)),
                                   Header(stamp=datetime.fromtimestamp(current_time)))
            if self.position_shift is not None:
                obj_pose.position += self.position_shift
            obj_type = Body

            # Create the object if it does not exist in the world and set its pose
            if obj_name not in self.world.get_object_names():
                obj = Body(obj_name, obj_type, pose=obj_pose, path=f"{obj_name}.urdf")
            else:
                obj = self.world.get_object_by_name(obj_name)
                objects_poses[obj] = obj_pose
        return objects_poses
