from __future__ import annotations

import datetime
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, ClassVar, Optional

from segmind.datastructures.enums import PlayerStatus
from segmind.utils import PropagatingThread
from semantic_digital_twin.world import World


logger = logging.getLogger(__name__)

try:
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError:
    RDRCaseViewer = None


@dataclass(unsafe_hash=True)
class EpisodePlayer(PropagatingThread, ABC):
    """
    A class that represents the thread that steps the world.
    """

    _instances: ClassVar[dict[type, EpisodePlayer]] = {}
    """
    The singleton dictionary that stores the instances of the player.
    """

    pause_resume_lock: ClassVar[RLock] = RLock()
    """
    The lock for pausing and resuming the player.
    """

    def __new__(cls, *args, **kwargs):
        if cls not in EpisodePlayer._instances:
            # Create the instance and store it under its specific class key
            instance = super().__new__(cls)
            EpisodePlayer._instances[cls] = instance
            instance._initialized = False

        return EpisodePlayer._instances[cls]

    def __init__(
        self,
        time_between_frames: Optional[datetime.timedelta] = None,
        use_realtime: bool = False,
        stop_after_ready: bool = False,
        world: Optional[World] = None,
        rdr_viewer: Optional[RDRCaseViewer] = None,
    ):
        """
        Initializes the episode player.

        :param time_between_frames: The time between frames.
        :param use_realtime: Whether to use realtime.
        :param stop_after_ready: Whether to stop after ready.
        :param world: The world.
        :param rdr_viewer: The RDR case viewer.
        """
        if not self._initialized:
            super().__init__()
            self.rdr_viewer: Optional[RDRCaseViewer] = rdr_viewer
            self.stop_after_ready: bool = stop_after_ready
            self.world: World = world
            self._ready: bool = False
            self._status = PlayerStatus.CREATED
            self.time_between_frames: datetime.timedelta = (
                time_between_frames
                if time_between_frames is not None
                else datetime.timedelta(seconds=0.01)
            )
            self.use_realtime: bool = use_realtime
            self._initialized = True

    @property
    def status(self) -> PlayerStatus:
        """
        Retrieves the current status of the episode player.

        :return: The current status.
        """
        return self._status

    @property
    def ready(self) -> bool:
        """
        Retrieves the ready status of the episode player.

        :return: The ready status.
        """
        return self._ready

    @ready.setter
    def ready(self, value: bool):
        """
        Sets the ready status of the episode player.

        :param value: The ready status.
        """
        self._ready = value
        if value and self.stop_after_ready:
            self._status = PlayerStatus.STOPPED

    def run(self):
        """
        Starts the episode player.
        """
        self._status = PlayerStatus.PLAYING
        super().run()

    def pause(self):
        """
        Pause the episode player frame processing.
        """
        self._status = PlayerStatus.PAUSED
        self._pause()

    @abstractmethod
    def _pause(self):
        """
        Perform extra functionalities when the episode player is paused.
        """
        pass

    def resume(self):
        """
        Resume the episode player frame processing.
        """
        self._status = PlayerStatus.PLAYING
        self._resume()

    @abstractmethod
    def _resume(self):
        """
        Perform extra functionalities when the episode player is resumed.
        """
        pass

    def _wait_if_paused(self):
        """
        Wait if the episode player is paused.
        """
        while self.status == PlayerStatus.PAUSED and not self.kill_event.is_set():
            time.sleep(0.1)

    def _wait_to_maintain_frame_rate(
        self,
        last_processing_time: float,
        delta_time: Optional[datetime.timedelta] = None,
    ):
        """
        Wait to maintain the frame rate of the episode player.

        :param last_processing_time: The time of the last processing.
        :param delta_time: The delta time.
        """
        if delta_time is None:
            delta_time = self.time_between_frames
        time_to_wait = datetime.timedelta(seconds=time.time() - last_processing_time)
        if delta_time < time_to_wait:
            time.sleep((time_to_wait - delta_time).total_seconds())

    @classmethod
    def get_instance(cls) -> EpisodePlayer:
        """
        Retrieves the singleton instance of the player.

        :return: The singleton instance.
        """
        if cls not in EpisodePlayer._instances:
            raise RuntimeError(f"{cls.__name__} has not been initialized.")
        return EpisodePlayer._instances[cls]

    @classmethod
    def pause_resume_with_condition(cls, condition: Callable[[Any], bool]) -> Callable:
        """
        A decorator for pausing the player before a function call given a condition and then resuming it after the call
         ends.

        :param condition: The condition to check before pausing the player.
        :return: The wrapped callable.
        """

        def condition_wrapper(func: Callable) -> Callable:
            """
            A decorator for pausing the player before a function call and then resuming it after the call ends.

            :param func: The callable to wrap with the decorator.
            :return: The wrapped callable.
            """

            def wrapper(*args, **kwargs) -> Any:
                if not condition(*args, **kwargs):
                    return func(*args, **kwargs)
                with cls.pause_resume_lock:
                    instance = cls.get_instance()
                    if instance.status == PlayerStatus.PLAYING:
                        logger.debug("Pausing player")
                        instance.pause()
                        result = func(*args, **kwargs)
                        instance.resume()
                        logger.debug("Resuming player")
                        return result
                    else:
                        return func(*args, **kwargs)

            return wrapper

        return condition_wrapper

    @classmethod
    def pause_resume(cls, func: Callable) -> Callable:
        """
        A decorator for pausing the player before a function call and then resuming it after the call ends.

        :param func: The callable to wrap with the decorator.
        :return: The wrapped callable.
        """
        return cls.pause_resume_with_condition(lambda *args, **kwargs: True)(func)

    def _join(self, timeout=None):
        """
        Joins the episode player thread and removes the instance from the singleton dictionary.

        :param timeout: The timeout.
        """
        if self.__class__ in EpisodePlayer._instances:
            del EpisodePlayer._instances[self.__class__]
        super()._join(timeout)
