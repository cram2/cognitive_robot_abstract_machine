from __future__ import annotations

import logging
from datetime import timedelta
import os

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Callable, Optional, Dict, Generator, List

from segmind.datastructures.enums import PlayerStatus
from segmind.episode_player import EpisodePlayer
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)



@dataclass
class FrameData:
    """
    A dataclass to store the frame data.
    """

    time: float
    """
    The time of the frame.
    """
    objects_data: Dict
    """
    The objects data which contains the poses of the objects.
    """
    frame_idx: int
    """
    The frame index.
    """


FrameDataGenerator = Generator[FrameData, None, None]


@dataclass(unsafe_hash=True, init=False)
class DataPlayer(EpisodePlayer, ABC):
    """
    A class that represents the thread that steps the world from a data generator.
    """

    def __init__(
        self,
        world: World,
        time_between_frames: Optional[timedelta] = None,
        use_realtime: bool = False,
        stop_after_ready: bool = False,
    ):
        super().__init__(
            time_between_frames=time_between_frames,
            use_realtime=use_realtime,
            stop_after_ready=stop_after_ready,
            world=world,
        )
        self.world = world
        self.frame_callbacks: List[Callable[[float], None]] = []
        self.frame_data_generator: FrameDataGenerator = self.get_frame_data_generator()
        self.sync_robot_only: bool = False

    def reset(self):
        """
        Reset the player to its initial state.
        """
        logger.debug("Resetting DataPlayer !!!!!!!!!!!!!!!!")
        self.ready = False
        self.frame_callbacks = []
        self.frame_data_generator = self.get_frame_data_generator()

    @abstractmethod
    def get_frame_data_generator(self) -> FrameDataGenerator:
        """
        :return: the frame data generator.
        """
        pass


    def _run(self):
        is_first_frame = True
        start_time: float = 0.0
        for frame_data in self.frame_data_generator:

            if self.kill_event.is_set():
                break

            self._wait_if_paused()

            last_processing_time = time.time()

            time.sleep(self.time_between_frames.total_seconds())

            current_time = frame_data.time
            if is_first_frame:
                start_time = current_time
            dt = current_time - start_time
            self.process_objects_data(frame_data)

            self.ready = True

            if self._status == PlayerStatus.STOPPED:
                break

            if self.use_realtime:
                wait_time = timedelta(seconds=dt)
                self._wait_to_maintain_frame_rate(last_processing_time, wait_time)

            is_first_frame = False
        self._status = PlayerStatus.STOPPED

    def process_objects_data(self, frame_data: FrameData):
        """
        Process the objects data, by extracting and setting the poses of objects.

        :param frame_data: The frame data.
        """
        objects_poses = self.get_objects_poses(frame_data)
        if len(objects_poses):
            for obj in self.world.bodies_with_collision:
                if obj in objects_poses:
                    obj.parent_connection.origin = (
                        objects_poses[obj].to_homogeneous_matrix()
                    )


    @abstractmethod
    def get_objects_poses(self, frame_data: FrameData) -> Dict[Body, Pose]:
        """
        Get the poses of the objects.

        :param frame_data: The frame data.
        :return: The poses of the objects.
        """
        pass

    @abstractmethod
    def get_joint_states(self, frame_data: FrameData) -> Dict[str, float]:
        # Not implemented
        pass


@dataclass(unsafe_hash=True, init=False)
class FilePlayer(DataPlayer, ABC):
    """
    Plays the episode from a file.
    """

    file_path: str = field(default="")
    """
    file path to the episode.
    """

    models_dir: Optional[str] = field(default=None)
    """
    models directory.
    """

    def __init__(
        self,
        file_path: str,
        world: World,
        models_dir: Optional[str] = None,
        time_between_frames: Optional[timedelta] = None,
        use_realtime: bool = False,
        stop_after_ready: bool = False,
        position_shift: Optional[Vector3] = None,
    ):
        """
        Initializes the FilePlayer with the specified file.

        :param file_path: The file that contains the data frames.
        :param world: The world that is used to replay the episode.
        :param time_between_frames: The time between frames.
        :param use_realtime: Whether to use realtime.
        :param stop_after_ready: Whether to stop the player after it is ready.
        :param models_dir: The directory that contains the model files.
        :param position_shift: The position shift to apply to the objects.
        """
        self.file_path = file_path
        super().__init__(
            time_between_frames=time_between_frames,
            use_realtime=use_realtime,
            world=world,
            stop_after_ready=stop_after_ready,
        )

        self.models_dir = models_dir or os.path.join(
            os.path.dirname(self.file_path), "models"
        )

        self.position_shift: Optional[Vector3] = position_shift


