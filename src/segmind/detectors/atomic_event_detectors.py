import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from math import ceil
from queue import Queue, Empty, Full
from segmind import logger as loggerdebug
from segmind import set_logger_level, LogLevel

import numpy as np

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from pycram.datastructures.pose import PoseStamped
from semantic_digital_twin.reasoning.predicates import (
    contact,
    collision_between_bodies,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body, Agent
from .. import logger
from ..datastructures.mixins import (
    HasPrimaryTrackedObject,
    HasSecondaryTrackedObject,
    HasPrimaryAndSecondaryTrackedObjects,
)
from ..detectors.motion_detection_helpers import (
    is_displaced,
    is_stopped,
    ExponentialMovingAverage,
)
from ..episode_player import EpisodePlayer
from ..players.data_player import DataPlayer

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
from pycram.tf_transformations import euler_from_quaternion
from typing_extensions import Optional, List, Union, Type, Tuple, Callable
from ..event_logger import EventLogger
from ..datastructures.events import (
    Event,
    CloseContactEvent,
    LossOfCloseContactEvent,
    AgentContactEvent,
    AgentLossOfContactEvent,
    TranslationEvent,
    StopTranslationEvent,
    NewObjectEvent,
    RotationEvent,
    StopRotationEvent,
    MotionEvent,
    AgentInterferenceEvent,
    ContactEvent,
    AbstractContactEvent,
    AgentLossOfInterferenceEvent,
    LossOfContactEvent,
)
from .motion_detection_helpers import DataFilter
from ..utils import (
    calculate_quaternion_difference,
    calculate_translation,
    PropagatingThread,
)

set_logger_level(LogLevel.DEBUG)


class DetectorStateChart(MotionStatechart):
    pass


class AtomicEventDetector(PropagatingThread):
    """
    A thread that detects events in another thread and logs them. The event detector is a function that has no arguments
    and returns an object that represents the event. The event detector is called in a loop until the thread is stopped
    by setting the exit_thread attribute to True.
    """

    def __init__(
        self,
        logger: Optional[EventLogger] = None,
        wait_time: Optional[timedelta] = None,
        world: Optional[World] = None,
        episode_player: Optional[EpisodePlayer] = None,
        fit_mode: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param wait_time: An optional timedelta value that introduces a delay between calls to the event detector.
        :param world: An optional World instance that represents the world.
        :param episode_player: An optional EpisodePlayer instance that represents the thread that steps the world.
        :param fit_mode: A boolean value that indicates if the event detector is in fit mode, if true, then the detector
        will pause the episode player when it is fitting a case.
        """

        super().__init__()
        self.episode_player: EpisodePlayer = episode_player
        self.fit_mode = fit_mode
        self.logger: EventLogger = logger if logger else EventLogger.current_logger
        if world:
            self.world: World = world
        elif episode_player and episode_player.world:
            self.world: World = episode_player.world
        else:
            self.world = World()

        self.wait_time = (
            wait_time if wait_time is not None else timedelta(milliseconds=50)
        )

        self.queues: List[Queue] = []

        self.run_once = False
        self._pause: bool = False
        self.is_processing_jobs: bool = False

    def reset(self):
        pass

    @property
    def thread_id(self) -> str:
        return f"{self.__class__.__name__}_{self.ident}"

    @abstractmethod
    def detect_events(self) -> List[Event]:
        """
        The event detector function that is called in a loop until the thread is stopped.
        :return: A list of Event instances.
        """
        pass

    def pause(self):
        """
        Pause the event detector.
        """
        self._pause: bool = True

    def resume(self):
        """
        Resume the event detector.
        """
        self._pause: bool = False

    def _run(self):
        """
        The main loop of the thread. The event detector is called in a loop until the thread is stopped by setting the
        exit_thread attribute to True. Additionally, there is an optional wait_time attribute that can be set to a float
        value to introduce a delay between calls to the event detector.
        """
        while True:

            if (
                self.kill_event.is_set()
                and self.all_queues_empty
                and not self.is_processing_jobs
            ) or self.exc is not None:
                break

            self._wait_if_paused()
            if self.fit_mode and self.episode_player:
                self.episode_player.pause()

            last_processing_time = time.time()
            self.detect_and_log_events()

            if self.fit_mode and self.episode_player:
                self.episode_player.resume()

            if self.run_once:
                break
            else:
                self._wait_to_maintain_loop_rate(last_processing_time)

    @property
    def all_queues_empty(self) -> bool:
        """
        Check if all the queues are empty.

        :return: A boolean value that represents if all the queues are empty.
        """
        return all([q.empty() for q in self.queues])

    def detect_and_log_events(self):
        """
        Detect and log the events.
        """
        events = self.detect_events()
        if events:
            [self.log_event(event) for event in events]

    def _wait_if_paused(self):
        """
        Wait if the event detector is paused.
        """
        while self._pause and not self.kill_event.is_set():
            time.sleep(0.1)

    def _wait_to_maintain_loop_rate(self, last_processing_time: float):
        """
        Wait to maintain the loop rate of the event detector.

        :param last_processing_time: The time of the last processing.
        """
        time_diff = time.time() - last_processing_time
        if time_diff < self.wait_time.total_seconds():
            time.sleep(self.wait_time.total_seconds() - time_diff)

    def log_event(self, event: Event) -> None:
        """
        Logs the event using the logger instance.
        :param event: An object that represents the event.
        :return: None
        """
        event.detector_thread_id = self.thread_id
        self.logger.log_event(event)

    @property
    def detected_before(self) -> bool:
        """
        Checks if the event was detected before.

        :return: A boolean value that represents if the event was detected before.
        """
        return self.thread_id in self.logger.get_events_per_thread().keys()

    @abstractmethod
    def __str__(self): ...

    def __repr__(self):
        return self.__str__()


class NewObjectDetector(AtomicEventDetector):
    """
    A thread that detects if a new object is added to the scene and logs the NewObjectEvent.
    """

    def __init__(
        self,
        logger: EventLogger,
        wait_time: Optional[timedelta] = None,
        avoid_objects: Optional[Callable[[Body], bool]] = None,
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param wait_time: An optional timedelta value that introduces a delay between calls to the event detector.
        :param avoid_objects: An optional list of strings that represent the names of the objects to avoid.
        """
        super().__init__(logger, wait_time, *args, **kwargs)
        self.new_object_queue: Queue[Body] = Queue()
        self.queues.append(self.new_object_queue)
        self.avoid_objects = avoid_objects if avoid_objects else lambda obj: False
        # self.world.add_callback_on_add_object(self.on_add_object)

    def on_add_object(self, obj: Body):
        """
        Callback function that is called when a new object is added to the scene.
        """
        if not self.avoid_objects(obj):
            self.new_object_queue.put(obj)

    def detect_events(self) -> List[Event]:
        """
        Detect if a new object is added to the scene and invoke the NewObjectEvent.

        :return: A NewObjectEvent that represents the addition of a new object to the scene.
        """
        events = []
        try:
            while True:
                event = NewObjectEvent(
                    self.new_object_queue.get_nowait(),
                    tracked_object=self.tracked_object,
                )
                self.new_object_queue.task_done()
                events.append(event)
        except queue.Empty:
            return events

    def stop(self):
        """
        Remove the callback on the add object event and resume the thread to be able to join.
        """
        # if self.world is not None:
        #    self.world.remove_callback_on_add_object(self.on_add_object)
        super().stop()

    def _join(self, timeout=None):
        pass

    def __str__(self):
        return self.thread_id


class DetectorWithTrackedObject(AtomicEventDetector, HasPrimaryTrackedObject, ABC):
    """
    A mixin class that provides one tracked object for the event detector.
    """

    def __init__(
        self,
        logger: EventLogger,
        tracked_object: Body,
        wait_time: Optional[timedelta] = None,
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param tracked_object: An Object instance that represents the object to track.
        :param wait_time: An optional timedelta value that introduces a delay between calls to the event detector.
        """
        HasPrimaryTrackedObject.__init__(self, tracked_object=tracked_object)
        AtomicEventDetector.__init__(self, logger, wait_time, *args, **kwargs)

    def __str__(self):
        return f"{self.thread_id} - {self.tracked_object.name}"


class DetectorWithTwoTrackedObjects(
    AtomicEventDetector, HasPrimaryAndSecondaryTrackedObjects, ABC
):
    """
    A mixin class that provides two tracked objects for the event detector.
    """

    def __init__(
        self,
        logger: EventLogger,
        tracked_object: Body,
        with_object: Optional[Body] = None,
        wait_time: Optional[timedelta] = None,
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param tracked_object: An Object instance that represents the object to track.
        :param with_object: An optional Object instance that represents the object to track.
        :param wait_time: An optional timedelta value that introduces a delay between calls to the event detector.
        """
        HasPrimaryAndSecondaryTrackedObjects.__init__(
            self, tracked_object=tracked_object, with_object=with_object
        )
        AtomicEventDetector.__init__(self, logger, wait_time, *args, **kwargs)

    def __str__(self):
        with_object_name = (
            f" - {self.with_object.name}" if self.with_object is not None else ""
        )
        return f"{self.thread_id} - {self.tracked_object.name}" + with_object_name






class AbstractContactDetector(DetectorWithTwoTrackedObjects, ABC):
    def __init__(
        self,
        logger: EventLogger,
        tracked_object: Body,
        with_object: Optional[Body] = None,
        max_closeness_distance: Optional[float] = 0.1,
        wait_time: Optional[timedelta] = timedelta(milliseconds=500),
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param starter_event: An instance of the Event class that represents the event to start the event detector.
        :param max_closeness_distance: An optional float value that represents the maximum distance between the object
        :param wait_time: An optional timedelta value that introduces a delay between calls to the event detector.
        """
        DetectorWithTwoTrackedObjects.__init__(
            self, logger, tracked_object, with_object, wait_time, *args, **kwargs
        )
        self.max_closeness_distance = max_closeness_distance
        self.latest_contact_bodies: Optional[list[Body]] = []

    def get_events(
        self,
        new_bodies_contact: list[Body],
        contact_bodies: list[Body],
        event_type: Type[AbstractContactEvent],
    ):
        events = []
        for body in new_bodies_contact:
            events.append(
                event_type(
                    close_bodies=contact_bodies,
                    latest_close_bodies=self.latest_contact_bodies,
                    of_object=self.tracked_object,
                    with_object=body,
                )
            )
        return events

    @property
    def obj_type(self) -> Type[Body]:
        """
        The object type of the object to track.
        """
        return self.tracked_object.__class__

    def detect_events(self) -> List[Event]:
        """
        Detects the closest points between the object to track and another object in the scene if the with_object
        attribute is set, else, between the object to track and all other objects in the scene.
        """
        contact_bodies = self.get_contact_bodies()

        events = self.trigger_events(contact_bodies)

        self.latest_contact_bodies = contact_bodies

        return events

    def get_contact_bodies(self) -> list[Body]:
        contact_bodies = []
        objects = (
            [self.with_object]
            if self.with_object
            else self.tracked_object._world.bodies_with_collision
        )
        for obj in objects:
            if obj is self.tracked_object:
                continue
            if contact(self.tracked_object, obj):
                contact_bodies.append(obj)
        return contact_bodies


    @abstractmethod
    def trigger_events(
        self, contact_bodies: list[Body]
    ) -> List[Event]:
        """
        Checks if the detection condition is met, (e.g., the object is in contact with another object),
        and returns an object that represents the event.
        :param contact_points: The current contact points.
        :param interference_points: The current interference points.
        :return: An object that represents the event.
        """
        pass

    def _join(self, timeout=None):
        pass


@dataclass(eq=False, repr=False)
class ContactDetectorNode(MotionStatechartNode, AbstractContactDetector):
    latest_contact_bodies: List[Body] = field(
        default_factory=list, kw_only=True, init=False
    )
    tracked_object: Body = field(kw_only=True)
    logger: EventLogger = field(kw_only=True)
    with_object: Optional[Body] = field(default=None, kw_only=True)
    max_closeness_distance: float = field(default=0.1, kw_only=True)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts()


    def trigger_events(
        self, contact_bodies: list[Body]
    ) -> Union[List[CloseContactEvent], List[AgentContactEvent]]:
        """
        Check if the object got into contact with another object.

        :param contact_points: The current contact points.
        :param interference_points: The current interference points.
        :return: An instance of the ContactEvent/AgentContactEvent class that represents the event if the object got
         into contact, else None.
        """
        new_contact_objects = []
        for body in contact_bodies:
            if body not in self.latest_contact_bodies:
                new_contact_objects.append(body)
        if self.with_object is not None:
            new_contact_objects = [
                body for body in new_contact_objects if body == self.with_object
            ]
        if len(new_contact_objects) == 0:
            return []
        return self.get_events(
            new_contact_objects,
            contact_bodies,
            ContactEvent,
        )

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        events = self.detect_events()
        detected_contact = []
        if len(events) > 0:
            for event in events:
                if isinstance(event, ContactEvent):
                    self.logger.log_event(event)
                    detected_contact.append(event)

        if detected_contact:
            return ObservationStateValues.TRUE

        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False)
class LossOfContactDetectorNode(MotionStatechartNode, AbstractContactDetector):
    tracked_object: Body = field(kw_only=True)
    logger: EventLogger = field(kw_only=True)
    with_object: Optional[Body] = field(default=None, kw_only=True)
    latest_contact_bodies: List[Body] = field(
        default_factory=list, kw_only=True, init=False
    )


    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts()

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        events = self.detect_events()
        detected_loss_of_contact = []
        if len(events) > 0:
            for event in events:
                if isinstance(event, LossOfContactEvent):
                    self.logger.log_event(event)
                    detected_loss_of_contact.append(event)

        if detected_loss_of_contact:
            return ObservationStateValues.TRUE

        return ObservationStateValues.FALSE


    def trigger_events(
        self,  contact_bodies: list[Body]
    ) -> List[LossOfCloseContactEvent]:
        """
        Check if the object lost contact with another object.

        :param contact_points: The current contact points.
        :param interference_points: The current interference points.
        :return: An instance of the LossOfContactEvent/AgentLossOfContactEvent class that represents the event if the
         object lost contact, else None.
        """
        objects_that_lost_contact= (
            self.get_bodies_that_lost_contact(contact_bodies)
        )
        if (len(objects_that_lost_contact) == 0):
            return []
        return self.get_events(
            objects_that_lost_contact,
            contact_bodies,
            LossOfContactEvent
        )

    def get_bodies_that_lost_contact(
        self, contact_bodies: list[Body]
    ) -> List[Body]:
        """
        Get the objects that lost contact with the object to track.

        :param contact_points: The current contact points.
        :param interference_points: The current interference points.
        :return: A list of Object instances that represent the objects that lost contact with the object to track.
        """

        objects_that_lost_contact = []
        for body in self.latest_contact_bodies:
            if body not in contact_bodies:
                objects_that_lost_contact.append(body)

        if self.with_object is not None:
            objects_that_lost_contact = [
                body
                for body in objects_that_lost_contact
                if body.parent_entity == self.with_object
            ]

        return objects_that_lost_contact

    # def get_events(
    #     self,
    #     new_close_objects: list[Body],
    #     new_contact_objects: list[Body],
    #     close_bodies: list[Body],
    #     contact_bodies: list[Body],
    # ):
    #     events = []
    #     for obj in new_close_objects:
    #         events.append(
    #             LossOfCloseContactEvent(
    #                 close_bodies=close_bodies,
    #                 latest_close_bodies=self.latest_close_bodies,
    #                 of_object=self.tracked_object,
    #                 with_object=obj,
    #             )
    #         )
    #
    #     for obj in new_contact_objects:
    #         events.append(
    #             LossOfContactEvent(
    #                 close_bodies=contact_bodies,
    #                 latest_close_bodies=self.latest_contact_bodies,
    #                 of_object=self.tracked_object,
    #                 with_object=obj,
    #             )
    #         )
    #
    #     return events


class ContactDetector(AbstractContactDetector):
    """
    A thread that detects if the object got into contact with another object.
    """

    def trigger_events(
        self, close_bodies: list[Body], contact_bodies: list[Body]
    ) -> Union[List[CloseContactEvent], List[AgentContactEvent]]:
        """
        Check if the object got into contact with another object.

        :param contact_points: The current contact points.
        :param interference_points: The current interference points.
        :return: An instance of the ContactEvent/AgentContactEvent class that represents the event if the object got
         into contact, else None.
        """
        new_close_objects = []
        new_contact_objects = []
        for body in close_bodies:
            if body not in self.latest_close_bodies:
                new_close_objects.append(body)

        for body in contact_bodies:
            # if body not in self.latest_contact_bodies and body not in new_close_objects:
            if body not in self.latest_contact_bodies:
                new_contact_objects.append(body)

        # new_bodies_in_interference = contact_bodies.get_new_bodies(self.latest_interference_points)

        if self.with_object is not None:
            new_close_objects = [
                obj for obj in new_close_objects if obj == self.with_object
            ]
            new_contact_objects = [
                body for body in new_contact_objects if body == self.with_object
            ]
        if len(new_close_objects) == 0 and len(new_contact_objects) == 0:
            return []
        return self.get_events(
            new_close_objects,
            new_contact_objects,
            close_bodies,
            contact_bodies,
            CloseContactEvent,
        )


class LossOfContactDetector(AbstractContactDetector):
    """
    A thread that detects if the object lost contact with another object.
    """

    def trigger_events(
        self, close_bodies: list[Body], contact_bodies: list[Body]
    ) -> List[LossOfCloseContactEvent]:
        """
        Check if the object lost contact with another object.

        :param contact_points: The current contact points.
        :param interference_points: The current interference points.
        :return: An instance of the LossOfContactEvent/AgentLossOfContactEvent class that represents the event if the
         object lost contact, else None.
        """
        objects_that_lost_contact, bodies_that_lost_interference = (
            self.get_bodies_that_lost_contact(close_bodies, contact_bodies)
        )
        if (
            len(objects_that_lost_contact) == 0
            and len(bodies_that_lost_interference) == 0
        ):
            return []
        return self.get_events(
            objects_that_lost_contact,
            bodies_that_lost_interference,
            close_bodies,
            contact_bodies,
            LossOfCloseContactEvent,
        )

    def get_bodies_that_lost_contact(
        self, close_bodies: list[Body], contact_bodies: list[Body]
    ) -> Tuple[List[Body], List[Body]]:
        """
        Get the objects that lost contact with the object to track.

        :param contact_points: The current contact points.
        :param interference_points: The current interference points.
        :return: A list of Object instances that represent the objects that lost contact with the object to track.
        """
        objects_that_lost_closeness = []
        objects_that_lost_contact = []
        for body in self.latest_close_bodies:
            if body not in close_bodies:
                objects_that_lost_closeness.append(body)
        for body in self.latest_contact_bodies:
            if body not in contact_bodies:
                objects_that_lost_contact.append(body)

        # objects_that_lost_contact = close_bodies.get_objects_that_got_removed(self.latest_close_bodies)
        # bodies_that_lost_interference = contact_bodies.get_bodies_that_got_removed(self.latest_contact_bodies)
        if self.with_object is not None:
            objects_that_lost_closeness = [
                obj for obj in objects_that_lost_closeness if obj == self.with_object
            ]
            objects_that_lost_contact = [
                body
                for body in objects_that_lost_contact
                if body.parent_entity == self.with_object
            ]

        return objects_that_lost_closeness, objects_that_lost_contact


class MotionDetector(DetectorWithTrackedObject, ABC):
    """
    A thread that detects if the object starts or stops moving and logs the TranslationEvent or StopTranslationEvent.
    """

    latest_pose: Optional[Pose] = None
    """
    The latest pose of the object.
    """
    latest_time: Optional[float] = None
    """
    The latest time where the latest pose was recorded.
    """
    event_time: Optional[float] = None
    """
    The time when the event occurred.
    """
    start_pose: Optional[Pose] = None
    """
    The start pose of the object at start of detection.
    """
    filtered_distances: Optional[np.ndarray] = None
    """
    The filtered distances during the window timeframe.
    """
    velocity_threshold: float
    """
    The threshold for the velocity to detect movement.
    """
    stop_velocity_threshold: float
    """
    The threshold for the velocity to detect movement.
    """

    def __init__(
        self,
        logger: EventLogger,
        tracked_object: Body,
        velocity_threshold: Optional[float] = None,
        time_between_frames: timedelta = timedelta(milliseconds=200),
        window_size_in_seconds: int = 0.3,
        distance_filter_method: Optional[DataFilter] = ExponentialMovingAverage(0.99),
        stop_velocity_threshold: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param starter_event: An instance of the NewObjectEvent class that represents the event to start the event.
        :param tracked_object: An optional Object instance that represents the object to track.
        :param velocity_threshold: The threshold for the velocity to detect movement.
        :param time_between_frames: The time between frames of episode player.
        :param window_size: The size of the window that is used to calculate the distances (must be > 1).
        :param distance_filter_method: An optional DataFilter instance that is used to filter the distances.
        :param stop_velocity_threshold: The threshold for the velocity to detect stop.
        """
        DetectorWithTrackedObject.__init__(
            self, logger, tracked_object, *args, **kwargs
        )
        if self.episode_player is not None and isinstance(
            self.episode_player, DataPlayer
        ):
            self.episode_player.add_frame_callback(self.update_with_latest_motion_data)
            # self.time_between_frames = self.episode_player.time_between_frames
            # self.time_between_frames: timedelta = timedelta(milliseconds=50)
            self.time_between_frames: timedelta = time_between_frames
        else:
            self.time_between_frames: timedelta = time_between_frames
        self.window_size: int = round(
            window_size_in_seconds / self.time_between_frames.total_seconds()
        )
        self.velocity_threshold: float = (
            velocity_threshold
            if velocity_threshold is not None
            else self.velocity_threshold
        )
        self.stop_velocity_threshold: float = (
            stop_velocity_threshold
            if stop_velocity_threshold is not None
            else self.stop_velocity_threshold
        )
        self.data_queue: Queue[Tuple[float, Pose]] = Queue(5)
        self.queues = [self.data_queue]
        self.distance_threshold: float = (
            self.velocity_threshold * self.window_size_in_seconds
        )
        self.stop_distance_threshold: float = (
            self.stop_velocity_threshold * self.window_size_in_seconds
        )
        self.measure_timestep: timedelta = self.time_between_frames * 2
        self.filter: Optional[DataFilter] = distance_filter_method

        self._init_data_holders()

        self.was_moving: bool = False
        self.last_state_change_idx: int = 0

        self.plot_distances: bool = False
        self.plot_distance_windows: bool = False
        self.plot_frequencies: bool = False

    @property
    def window_size_in_seconds(self) -> float:
        return self.window_size * self.time_between_frames.total_seconds()

    @property
    def window_size(self) -> int:
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        if window_size < 2:
            raise ValueError("The window size must be greater than 1.")
        self._window_size = window_size

    def get_n_changes_wait_time(self, n: int) -> float:
        """
        :param n: The number of successive changes in motion state.
        :return: The minimum wait time for detecting n successive changes in motion state.
        """
        return (
            self.window_timeframe.total_seconds() * n + self.wait_time.total_seconds()
        )

    @property
    def window_timeframe(self) -> timedelta:
        return self.measure_timestep * self.window_size

    @property
    def measure_timestep(self) -> timedelta:
        return self._measure_timestep

    @measure_timestep.setter
    def measure_timestep(self, measure_timestep: timedelta):
        """
        Update the measure timestep and the wait time between calls to the event detector.
        """
        # frames per measure timestep
        self.measure_frame_rate: float = ceil(
            measure_timestep.total_seconds() / self.time_between_frames.total_seconds()
        )
        self._measure_timestep = self.time_between_frames * self.measure_frame_rate
        self.wait_time = self._measure_timestep

    def _init_data_holders(self):
        """
        Initialize the pose, time, and distance data holders.
        """
        # Data
        self.poses: List[Pose] = []
        self.times: List[float] = []

        # Window data
        self.latest_distances: List[List[float]] = []
        self.all_distances: List[List[float]] = []
        self.filtered_distances: Optional[np.ndarray] = None
        self.latest_times: List[float] = []

        # Plotting data
        self.original_distances: List[List[List[float]]] = []
        self.all_filtered_distances: List[np.ndarray] = []
        self.all_times: List[List[float]] = []

    def update_with_latest_motion_data(
        self, current_time: Optional[float] = None
    ) -> Tuple[float, Pose]:
        """
        Update the latest pose and time of the object.
        """
        # loggerdebug.debug(f"Updating with latest motion data")
        latest_pose, latest_time = self.get_current_pose_and_time()
        # loggerdebug.debug(f"current time: {current_time}")
        if current_time is not None:
            latest_time = current_time
        repeat = True
        while repeat:
            try:
                self.data_queue.put_nowait((latest_time, latest_pose))
                repeat = False
                # loggerdebug.debug(f"Put pose {latest_pose} at time {latest_time}")
            except Full:
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.task_done()
                except Empty:
                    pass

        return latest_time, latest_pose

    def get_current_pose_and_time(self) -> Tuple[Pose, float]:
        """
        Get the current pose and time of the object.
        """
        pose = self.tracked_object.global_pose.to_pose()
        return pose, time.time()

    def detect_events(self) -> Optional[List[MotionEvent]]:
        """
        Detect if the object starts or stops moving.

        :return: An instance of the TranslationEvent class that represents the event if the object is moving, else None.
        """
        # loggerdebug.debug(f"Detecting events. {self.window_size}")
        # latest_time, latest_pose = self.update_with_latest_motion_data()
        try:
            self.latest_time, self.latest_pose = self.data_queue.get_nowait()
            self.data_queue.task_done()
        except Empty:
            # loggerdebug.debug(f"No pose in queue. {self.window_size}")
            return
        # loggerdebug.debug(f"Latest pose: {self.latest_pose}")
        # self.latest_pose, self.latest_time = self.get_current_pose_and_time()
        self.poses.append(self.latest_pose)
        self.times.append(self.latest_time)
        if len(self.poses) > 1:

            self.calculate_and_update_latest_distance()
            self._crop_distances_and_times_to_window_size()

        if not self.window_size_reached:
            return
        # loggerdebug.debug(
        #    f"Window size reached, checking for motion state change. {self.window_size}"
        # )
        events: Optional[List[MotionEvent]] = None
        if self.motion_sate_changed:
            self.last_state_change_idx = len(self.all_distances) - 1
            events = [self.update_motion_state_and_create_event()]

        if self.plot_distances:
            self.keep_track_of_history()

        return events

    def update_motion_state_and_create_event(self) -> MotionEvent:
        """
        Update the motion state of the object and create an event.
        :return: An instance of the MotionEvent class that represents the event.
        """
        self.was_moving = not self.was_moving
        self.update_object_motion_state(self.was_moving)
        self.update_start_pose_and_event_time()
        return self.create_event()

    @property
    def motion_sate_changed(self) -> bool:
        """
        Check if the object is moving/has stopped by using the motion detection method.

        :return: A boolean value that indicates if the object motion state has changed.
        """
        if self.was_moving:
            stopped = is_stopped(self.latest_distances, self.stop_distance_threshold)
            return stopped
        else:
            displaced = is_displaced(self.latest_distances, self.distance_threshold)
            return displaced

    def keep_track_of_history(self):
        """
        Keep track of the history of the object.
        """
        self.original_distances.append(self.latest_distances)
        if self.filtered_distances:
            self.all_filtered_distances.append(self.filtered_distances)
        self.all_times.append(self.latest_times)

    @property
    def time_since_last_event(self) -> float:
        return time.time() - self.latest_time

    @abstractmethod
    def update_object_motion_state(self, is_moving: bool) -> None:
        """
        Update the object motion state.

        :param is_moving: A boolean value that represents if the object is moving.
        """
        pass

    def calculate_and_update_latest_distance(self):
        """
        Calculate the latest distance and time between the current pose and the previous pose.
        """
        distance = self.calculate_distance()
        self.all_distances.append(distance)

    @property
    def measure_timestep_passed(self) -> bool:
        """
        :return: True if the measure timestep has passed since the last event.
        """
        return self.time_since_last_event >= self.measure_timestep.total_seconds()

    @property
    def window_size_reached(self) -> bool:
        return len(self.latest_distances) >= self.window_size

    def _crop_distances_and_times_to_window_size(self):
        if len(self.latest_distances) < self.window_size:
            self.latest_distances.append(self.all_distances[-1])
        else:
            self.latest_distances.pop(0)
            self.latest_distances.append(self.all_distances[-1])
        if self.filter:
            self.latest_distances = self.filter.filter_data(
                np.array(self.latest_distances)
            ).tolist()
        self.latest_times = self.times[-self.window_size :]

    def _reset_distances_and_times(self):
        self.latest_distances = []
        self.latest_times = []

    def update_start_pose_and_event_time(self, index: int = -1):
        """
        Update the start pose and event time.

        :param index: The index of the latest pose, and time.
        """
        self.start_pose = self.poses[-self.window_size]
        self.event_time = self.latest_times[-self.window_size]

    def filter_data(self) -> np.ndarray:
        """
        Apply a preprocessing filter to the distances.
        """
        self.filtered_distances = self.filter.filter_data(
            np.array(self.latest_distances)
        )
        return self.filtered_distances

    def create_event(self) -> MotionEvent:
        """
        Create a motion event.

        :return: An instance of the TranslationEvent class that represents the event.
        """
        current_pose, current_time = self.get_current_pose_and_time()
        event_type = self.get_event_type()
        start_pose_stamped = PoseStamped.from_spatial_type(self.start_pose)
        current_pose_stamped = PoseStamped.from_spatial_type(current_pose)
        event = event_type(
            self.tracked_object,
            start_pose_stamped,
            current_pose_stamped,
            timestamp=self.event_time,
        )
        return event

    @abstractmethod
    def calculate_distance(self):
        pass

    @abstractmethod
    def get_event_type(self):
        pass

    def stop(self):
        """
        Stop the event detector.
        """
        # plot the distances
        if self.plot_distances and plt:
            self.plot_and_show_distances()

        if self.plot_distance_windows and plt:
            self.plot_and_show_distance_windows()

        super().stop()

    def _join(self, timeout=None):
        pass

    def plot_and_show_distances(self, plot_filtered: bool = True) -> None:
        """
        Plot the average distances.
        """
        plt.plot(
            [t - self.times[0] for t in self.times],
            self.original_distances[: len(self.times)],
        )
        if plot_filtered and self.all_filtered_distances:
            plt.plot(
                [t - self.times[0] for t in self.times],
                self.all_filtered_distances[: len(self.times)],
            )
        plt.title(
            f"Results of {self.__class__.__name__} for {self.tracked_object.name}"
        )
        plt.show()

    def plot_and_show_distance_windows(self, plot_freq: bool = False) -> None:
        """
        Plot the distances and the frequencies of the distances.

        :param plot_freq: If True, plot the frequencies of the distances as well.
        """
        plot_cols: int = 2 if plot_freq else 1
        for i, window_time in enumerate(self.all_times):
            orig_distances = np.array(self.original_distances[i])
            times = np.array(window_time) - window_time[0]
            fig, axes = plt.subplots(3, plot_cols, figsize=(10, 10))
            self._add_distance_vs_filtered_to_plot(
                orig_distances,
                self.all_filtered_distances[i],
                times,
                axes[:, 0] if plot_freq else axes,
            )
            if plot_freq:
                self._add_frequencies_plot(orig_distances, axes[:, 1])
            plt.show()

    @staticmethod
    def _add_distance_vs_filtered_to_plot(
        distances: np.ndarray,
        filtered_distances: np.ndarray,
        times: np.ndarray,
        axes: np.ndarray,
    ) -> None:
        """
        Add the distances and the filtered distances to the figure.

        :param distances: The original distances.
        :param filtered_distances: The filtered distances.
        :param times: The times.
        :param axes: The axes to plot on.
        """
        ax_labels: List[str] = ["x", "y", "z"]
        for j, ax in enumerate(axes):
            original = distances[:, j]
            if np.mean(original) <= 1e-3:
                continue
            filtered = filtered_distances[:, j]
            ax.plot(times, original, label=f"original_{ax_labels[j]}")
            ax.plot(times[-len(filtered) :], filtered, label=f"filtered_{ax_labels[j]}")
            ax.legend()

    def _add_frequencies_plot(self, distances: np.ndarray, axes: np.ndarray) -> None:
        """
        Add the frequencies plot to the figure.

        :param distances: The distances.
        :param axes: The axes to plot on.
        """
        for j, ax in enumerate(axes):
            xmag = np.fft.fft(distances[:, j])
            freqs = np.fft.fftfreq(len(xmag), d=self.measure_timestep.total_seconds())
            ax.bar(freqs[: len(xmag) // 2], np.abs(xmag)[: len(xmag) // 2], width=0.1)
            ax.legend()


class TranslationDetector(MotionDetector):
    translating_velocity_in_mm_per_second: float = 30
    velocity_threshold: float = translating_velocity_in_mm_per_second * 1e-3
    stop_velocity_in_mm_per_second: float = 3
    stop_velocity_threshold: float = stop_velocity_in_mm_per_second * 1e-3

    def update_object_motion_state(self, is_moving: bool) -> None:
        """
        Update the object motion state.
        """
        self.tracked_object.is_translating = is_moving

    def calculate_distance(self):
        """
        Calculate the Euclidean distance between the latest and current positions of the object.
        """
        # return calculate_euclidean_distance(self.latest_pose.position.to_list(), current_pose.position.to_list())
        pos1 = [
            float(self.poses[-2].x),
            float(self.poses[-2].y),
            float(self.poses[-2].z),
        ]
        pos2 = [
            float(self.poses[-1].x),
            float(self.poses[-1].y),
            float(self.poses[-1].z),
        ]
        translation = calculate_translation(pos1, pos2)
        return translation

    def get_event_type(self):
        return TranslationEvent if self.was_moving else StopTranslationEvent


class RotationDetector(MotionDetector):
    degrees_per_second: float = 10
    velocity_threshold: float = degrees_per_second * np.pi / 180

    def update_object_motion_state(self, moving: bool) -> None:
        """
        Update the object motion state.
        """
        self.tracked_object.is_rotating = moving

    def calculate_distance(self):
        """
        Calculate the angle between the latest and current quaternions of the object
        """
        quat_diff = calculate_quaternion_difference(
            self.poses[-2].to_quaternion().to_list(),
            self.poses[-1].to_quaternion().to_list(),
        )
        # angle = 2 * np.arccos(quat_diff[0])
        euler_diff = list(euler_from_quaternion(quat_diff))
        euler_diff[2] = 0
        return euler_diff

    def get_event_type(self):
        return RotationEvent if self.was_moving else StopRotationEvent
