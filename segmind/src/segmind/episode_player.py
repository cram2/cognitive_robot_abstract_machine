from __future__ import annotations

import datetime
import logging
import time
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Any, Callable, ClassVar, Optional

from krrood.singleton import SingletonMeta
from segmind.datastructures.enums import PlayerStatus
from segmind.utils import PropagatingThread
from semantic_digital_twin.world import World


logger = logging.getLogger(__name__)

try:
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError:
    RDRCaseViewer = None


class SingletonABCMeta(SingletonMeta, ABCMeta):
    """
    Combines :class:`SingletonMeta` with :class:`abc.ABCMeta`.

    A class's metaclass must be a subclass of the metaclass of each of its bases, so
    :class:`SingletonMeta` on its own cannot be used with an abstract base class.
    """


@dataclass(eq=False)
class EpisodePlayer(PropagatingThread, ABC, metaclass=SingletonABCMeta):
    """
    The EpisodePlayer class is a base class for all episode players.
    It provides common functionality and properties for episode players.
    """

    world: Optional[World] = None
    """
    The world associated with the episode player.
    """

    time_between_frames: Optional[datetime.timedelta] = field(default_factory=lambda: datetime.timedelta(seconds=0.01))
    """
    The time between frames of the episode player.
    """

    use_realtime: bool = False
    """
    Whether to use realtime or fixed time step.
    """

    stop_after_ready: bool = False
    """
    If True, the episode player will stop after it is ready.
    """

    rdr_viewer: Optional[RDRCaseViewer] = None
    """
    The RDRCaseViewer instance.
    """

    pause_resume_lock: ClassVar[RLock] = field(default=RLock(), init=False)
    """
    A lock for pausing and resuming the episode player.
    """

    _ready : bool = field(default=False, init=False)
    """
    Whether the episode player is ready to start playing.
    """

    _status: PlayerStatus = field(default=PlayerStatus.CREATED, init=False)
    """
    The status of the episode player.
    """

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
        elapsed_time = datetime.timedelta(seconds=time.time() - last_processing_time)
        if delta_time > elapsed_time:
            time.sleep((delta_time - elapsed_time).total_seconds())

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
        A decorator that pauses the player before the decorated function call if
        the condition is met, then resumes it afterward.

        :param condition: The condition to check before pausing the player.
        :return: The wrapped callable.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if not condition(*args, **kwargs):
                    return func(*args, **kwargs)
                with cls.pause_resume_lock:
                    instance = cls.get_instance()
                    if instance.status != PlayerStatus.PLAYING:
                        return func(*args, **kwargs)
                    logger.debug("Pausing player")
                    instance.pause()
                    result = func(*args, **kwargs)
                    instance.resume()
                    logger.debug("Resuming player")
                    return result

            return wrapper

        return decorator

    @classmethod
    def pause_resume(cls, func: Callable) -> Callable:
        """
        A decorator that unconditionally pauses the player before the decorated
        function call and resumes it afterward.

        :param func: The callable to wrap.
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
