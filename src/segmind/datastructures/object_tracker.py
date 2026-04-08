from __future__ import annotations


from dataclasses import dataclass
from datetime import timedelta
from threading import RLock
from typing import Callable, Tuple

from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Type, Optional, TYPE_CHECKING, Dict, Set
from segmind import logger, set_logger_level, LogLevel
import numpy as np



if TYPE_CHECKING:
    from .events import Event, EventUnion
    from ..detectors.base import SegmindContext

set_logger_level(LogLevel.DEBUG)

@dataclass
class ObjectTracker:
    """
    Tracks and manages events and movement status of an object.

    The ObjectTracker class offers functionality to monitor, sort, filter, and
    retrieve events associated with an object. It provides a structured way of
    storing and accessing historical events while maintaining thread safety. The
    class also integrates the ability to analyze the movement state of objects
    through its context and body.
    """

    context: Optional[SegmindContext]
    """
    Context that provides relevant configuration and state information.
    """

    body: Optional[Body]
    """
    The body associated with this tracker.
    """

    _event_history: Optional[List[Event]]
    """
    List of events associated with the object.
    """

    _lock: RLock = RLock()
    """ 
    threading.RLock object used for thread-safe access to the object's event history.
    """

    @property
    def current_state(self) -> bool:
        """
        Returns the current state of the object's movement status.

        This property provides a way to check whether the associated object
        is currently moving without directly interacting with the internal
        status storage. It uses the object's context and body information
        to retrieve this status.

        :return: Indicates True if the object is moving, otherwise False.
        :rtype: bool
        """
        with self._lock:
            return self.context.object_moving_status[self.body]


    def add_event(self, event: Event):
        """
        Adds an event to the event history in a thread-safe manner.

        The event is appended to the event history, and the history is
        then sorted based on the timestamp of each event.

        :param event: The event to be added to the history.
        :type event: Event
        """
        with self._lock:
            self._event_history.append(event)
            self._event_history.sort(key=lambda e: e.timestamp)


    def get_event_history(self) -> List[Event]:
        """
        Retrieves the history of events for this instance.

        This method returns a list of events that have been recorded.
        The event history is thread-safe and is accessed under a lock
        to ensure consistent data.

        :return: A list of events recorded in the event history.
        :rtype: List[Event]
        """
        with self._lock:
            return self._event_history


    def clear_event_history(self):
        """
        Clears the event history of the object.

        This method ensures thread-safety while clearing the event history
        to maintain consistency in a multi-threaded environment.

        """
        with self._lock:
            self._event_history.clear()

    def get_latest_event(self) -> Optional[Event]:
        """
        Retrieves the most recent event from the event history.

        :return: The latest event from the event history, or None if the history is empty.
        :rtype: Optional[Event]
        """
        with self._lock:
            try:
                return self._event_history[-1]
            except IndexError:
                return None


    def get_latest_event_of_type(self, event_type: Type[Event]) -> Optional[Event]:
        """
        Retrieves the latest event of the specified type from the event history.

        This method searches through the event history in reverse order to find the
        most recent event that matches the given event type.

        :param event_type: The type of the event to search for.
        :return: The most recent event of the specified type if found, otherwise None.
        """
        with self._lock:
            for event in reversed(self._event_history):
                if isinstance(event, event_type):
                    return event
            return None


    def get_first_event_before(self, timestamp: float) -> Optional[Event]:
        """
        Retrieves the first event that occurred before the specified timestamp.

        This method checks for the first event in the event history that took place
        prior to the given timestamp. If such an event is found, it is returned;
        otherwise, None is returned.

        :param timestamp: The reference timestamp to compare against.
        :return: The first event occurring before the given timestamp, or None if no such event exists.
        """
        with self._lock:
            first_event_index = self.get_index_of_first_event_before(timestamp)
            return self._event_history[first_event_index] if first_event_index is not None else None

    def get_first_event_after(self, timestamp: float) -> Optional[Event]:
        """
        Gets the first event that occurs after the given timestamp.

        Uses the provided timestamp to locate the first event in the event
        history that occurs after the specified time.

        :param timestamp: The reference timestamp, in seconds, after which the event is searched for.
        :type timestamp: float

        :return: The first event occurring after the given timestamp, or None if no such event exists.
        :rtype: Optional[Event]
        """
        with self._lock:
            first_event_index = self.get_index_of_first_event_after(timestamp)
            return self._event_history[first_event_index] if first_event_index is not None else None

    def get_nearest_event_of_type_to_event(self, event: Event, event_type: Type[Event],
                                           tolerance: Optional[timedelta] = None) -> Optional[EventUnion]:
        """
        Returns the nearest event of a specified type to a given event.

        :param event: The reference event.
        :type event: Event
        :param event_type: The type of event to search for.
        :type event_type: Type[Event]
        :param tolerance: Maximum allowed time difference.
        :type tolerance: Optional[timedelta]

        :return: The nearest matching event or None.
        :rtype: Optional[EventUnion]
        """
        return self.get_nearest_event_of_type_to_timestamp(event.timestamp, event_type, tolerance)

    def get_nearest_event_of_type_to_timestamp(self, timestamp: float, event_type: Type[Event],
                                               tolerance: Optional[timedelta] = None) -> Optional[Event]:
        """
        Finds the nearest event of a specified type to a timestamp.

        :param timestamp: Reference timestamp.
        :type timestamp: float
        :param event_type: Event type to search for.
        :type event_type: Type[Event]
        :param tolerance: Maximum allowed time difference.
        :type tolerance: Optional[timedelta]

        :return: Closest event or None.
        :rtype: Optional[Event]
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            type_cond = np.array([isinstance(event, event_type) for event in self._event_history])
            valid_indices = np.where(type_cond)[0]
            if len(valid_indices) > 0:
                time_stamps = time_stamps[valid_indices]
                nearest_event_index = self._get_nearest_index(time_stamps, timestamp, tolerance)
                if nearest_event_index is not None:
                    return self._event_history[valid_indices[nearest_event_index]]

    def get_nearest_event_to(self, timestamp: float, tolerance: Optional[timedelta] = None) -> Optional[Event]:
        """
        Finds the nearest event to a timestamp.

        :param timestamp: Target timestamp.
        :type timestamp: float
        :param tolerance: Optional tolerance.
        :type tolerance: Optional[timedelta]

        :return: Nearest event or None.
        :rtype: Optional[Event]
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            nearest_event_index = self._get_nearest_index(time_stamps, timestamp, tolerance)
            if nearest_event_index is not None:
                return self._event_history[nearest_event_index]

    def _get_nearest_index(self, time_stamps: np.ndarray,
                           timestamp: float, tolerance: Optional[timedelta] = None) -> Optional[int]:
        """
        Finds the nearest index to a timestamp.

        :param time_stamps: Array of timestamps.
        :type time_stamps: np.ndarray
        :param timestamp: Target timestamp.
        :type timestamp: float
        :param tolerance: Optional tolerance.
        :type tolerance: Optional[timedelta]

        :return: Index or None.
        :rtype: Optional[int]
        """
        with self._lock:
            nearest_event_index = np.argmin(np.abs(time_stamps - timestamp))
            if tolerance is not None and abs(time_stamps[nearest_event_index] - timestamp) > tolerance.total_seconds():
                return None
            return nearest_event_index

    def get_nearest_event_to_event_with_conditions(self, event: Event, conditions: Callable[[Event], bool]) -> Optional[
        Event]:
        """
        Finds nearest event to another event matching conditions.

        :param event: Reference event.
        :type event: Event
        :param conditions: Filtering function.
        :type conditions: Callable[[Event], bool]

        :return: Matching event or None.
        :rtype: Optional[Event]
        """
        with self._lock:
            events = self.get_events_sorted_by_nearest_to_event(event)
            found_events = self.get_event_where(conditions, events=[e[0] for e in events])
            return found_events[0] if found_events else None

    def get_events_sorted_by_nearest_to_event(self, event: Event):
        """
        Sorts events by proximity to an event.

        :param event: Reference event.
        :type event: Event

        :return: Sorted events.
        :rtype: list
        """
        return self.get_events_sorted_by_nearest_to_timestamp(event.timestamp)

    def get_events_sorted_by_nearest_to_timestamp(self, timestamp: float) -> List[Tuple[Event, float]]:
        """
        Sorts events by proximity to a timestamp.

        :param timestamp: Reference timestamp.
        :type timestamp: float

        :return: List of (event, time difference).
        :rtype: List[Tuple[Event, float]]
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            time_diff = np.abs(time_stamps - timestamp)
            events_with_time_diff = [(event, dt) for event, dt in zip(self._event_history, time_diff)]
            events_with_time_diff.sort(key=lambda e: e[1])
        return events_with_time_diff

    def get_first_event_of_type_after_event(self, event_type: Type[Event], event: Event) -> Optional[EventUnion]:
        """
        Gets the first event of a type after an event.

        :param event_type: Event type.
        :type event_type: Type[Event]
        :param event: Reference event.
        :type event: Event

        :return: Matching event or None.
        :rtype: Optional[EventUnion]
        """
        return self.get_first_event_of_type_after_timestamp(event_type, event.timestamp)

    def get_first_event_of_type_after_timestamp(self, event_type: Type[Event], timestamp: float) -> Optional[Event]:
        """
        Gets first event of a type after a timestamp.

        :param event_type: Event type.
        :type event_type: Type[Event]
        :param timestamp: Reference timestamp.
        :type timestamp: float

        :return: Matching event or None.
        :rtype: Optional[Event]
        """
        with self._lock:
            start_index = self.get_index_of_first_event_after(timestamp)
            if start_index is not None:
                for event in self._event_history[start_index:]:
                    if isinstance(event, event_type):
                        return event

    def get_first_event_of_type_before_event(self, event_type: Type[Event], event: Event) -> Optional[EventUnion]:
        """
        Gets first event of a type before an event.

        :param event_type: Event type.
        :type event_type: Type[Event]
        :param event: Reference event.
        :type event: Event

        :return: Matching event or None.
        :rtype: Optional[EventUnion]
        """
        return self.get_first_event_of_type_before_timestamp(event_type, event.timestamp)

    def get_first_event_of_type_before_timestamp(self, event_type: Type[Event], timestamp: float) -> Optional[Event]:
        """
        Gets first event of a type before a timestamp.

        :param event_type: Event type.
        :type event_type: Type[Event]
        :param timestamp: Reference timestamp.
        :type timestamp: float

        :return: Matching event or None.
        :rtype: Optional[Event]
        """
        with self._lock:
            start_index = self.get_index_of_first_event_before(timestamp)
            if start_index is not None:
                for event in reversed(self._event_history[:min(start_index + 1, len(self._event_history))]):
                    if isinstance(event, event_type):
                        return event

    def get_index_of_first_event_after(self, timestamp: float) -> Optional[int]:
        """
        Gets index of first event after timestamp.

        :param timestamp: Reference timestamp.
        :type timestamp: float

        :return: Index or None.
        :rtype: Optional[int]
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            try:
                return np.where(time_stamps > timestamp)[0][0]
            except IndexError:
                return None

    def get_index_of_first_event_before(self, timestamp: float) -> Optional[int]:
        """
        Gets index of first event before timestamp.

        :param timestamp: Reference timestamp.
        :type timestamp: float

        :return: Index or None.
        :rtype: Optional[int]
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            try:
                return np.where(time_stamps < timestamp)[0][-1]
            except IndexError:
                return None

    def get_events_between_two_events(self, event1: Event, event2: Event) -> List[Event]:
        """
        Gets events between two events.

        :param event1: Start event.
        :type event1: Event
        :param event2: End event.
        :type event2: Event

        :return: Events between them.
        :rtype: List[Event]
        """
        return [e for e in self.get_events_between_timestamps(event1.timestamp, event2.timestamp)
                if e not in [event1, event2]]

    def get_events_between_timestamps(self, timestamp1: float, timestamp2: float) -> List[Event]:
        """
        Gets events between timestamps.

        :param timestamp1: First timestamp.
        :type timestamp1: float
        :param timestamp2: Second timestamp.
        :type timestamp2: float

        :return: List of events.
        :rtype: List[Event]
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            if timestamp1 > timestamp2:
                timestamp1, timestamp2 = timestamp2, timestamp1
            try:
                indices = np.where(np.logical_and(time_stamps <= timestamp2, time_stamps >= timestamp1))[0]
                events = [self._event_history[i] for i in indices]
                return events
            except IndexError:
                logger.error(f"No events between timestamps {timestamp1}, {timestamp2}")
                return []

    def get_event_where(self, conditions: Callable[[Event], bool], events: Optional[List[Event]] = None) -> List[Event]:
        """
        Filters events by condition.

        :param conditions: Condition function.
        :type conditions: Callable[[Event], bool]
        :param events: Optional event list.
        :type events: Optional[List[Event]]

        :return: Matching events.
        :rtype: List[Event]
        """
        events = events if events is not None else self._event_history
        return [event for event in events if conditions(event)]

    @property
    def time_stamps_array(self) -> np.ndarray:
        return np.array(self.time_stamps)

    @property
    def time_stamps(self) -> List[float]:
        with self._lock:
            return [event.timestamp for event in self._event_history]


class ObjectTrackerFactory:
    """
    Factory class to manage creation and access of ObjectTracker instances.

    This class is used to manage a collection of ObjectTracker instances associated
    with Body objects. It enforces synchronization to ensure thread safety and
    ensures a single ObjectTracker instance per Body object. The factory allows
    retrieving all registered ObjectTracker instances or creating/retrieving a
    specific tracker for a given Body.
    """

    _trackers: Dict[Body, ObjectTracker] = {}
    _lock: RLock = RLock()

    @classmethod
    def get_all_trackers(cls) -> List[ObjectTracker]:
        with cls._lock:
            return list(cls._trackers.values())

    @classmethod
    def get_tracker(cls, obj: Body) -> ObjectTracker:
        with cls._lock:
            if obj not in cls._trackers:
                cls._trackers[obj] = ObjectTracker(body=obj, context=None, _event_history=[])
            return cls._trackers[obj]


