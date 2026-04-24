from __future__ import annotations


from dataclasses import dataclass, field
from datetime import timedelta
from threading import RLock
from typing import Callable, Tuple, ClassVar

from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Type, Optional, TYPE_CHECKING, Dict, Set
from segmind import logger, set_logger_level, LogLevel
import numpy as np



if TYPE_CHECKING:
    from .events import DetectionEvent
    from ..detectors.base import SegmindContext

set_logger_level(LogLevel.DEBUG)

@dataclass
class ObjectEventTracker:
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

    #Check with msc for the history
    _event_history: Optional[List[DetectionEvent]]
    """
    List of events associated with the object.
    """

    _lock: RLock = RLock()
    """ 
    threading.RLock object used for thread-safe access to the object's event history.
    """

    def get_all_events_of_type(self, type_: Type[DetectionEvent], latest_first: bool = True):
        """
        :param type_: Type of event to retrieve.
        :param latest_first: If True, returns events in chronological order (latest first). Otherwise, returns events in reverse chronological order (oldest first).
        :return: List of events of the specified type in chronological order.
        """
        filtered_events = self.get_event_where(lambda e: isinstance(e, type_))
        if latest_first:
            return reversed(filtered_events)
        return filtered_events

    def add_event(self, event: DetectionEvent):
        """
        Adds an event to the event history in a thread-safe manner.

        The event is appended to the event history, and the history is
        then sorted based on the timestamp of each event.

        :param event: The event to be added to the history.
        """
        with self._lock:
            self._event_history.append(event)
            self._event_history.sort(key=lambda e: e.timestamp)


    def get_event_history(self) -> List[DetectionEvent]:
        """
        Retrieves the history of events for this instance.

        This method returns a list of events that have been recorded.
        The event history is thread-safe and is accessed under a lock
        to ensure consistent data.

        :return: A list of events recorded in the event history.
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

    def get_latest_event(self) -> Optional[DetectionEvent]:
        """
        Retrieves the most recent event from the event history.

        :return: The latest event from the event history, or None if the history is empty.
        """
        with self._lock:
            try:
                return self._event_history[-1]
            except IndexError:
                return None


    def get_latest_event_of_type(self, event_type: Type[DetectionEvent]) -> Optional[DetectionEvent]:
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


    def get_first_event_before(self, timestamp: float) -> Optional[DetectionEvent]:
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

    def get_first_event_after(self, timestamp: float) -> Optional[DetectionEvent]:
        """
        Gets the first event that occurs after the given timestamp.

        Uses the provided timestamp to locate the first event in the event
        history that occurs after the specified time.

        :param timestamp: The reference timestamp, in seconds, after which the event is searched for.
        :return: The first event occurring after the given timestamp, or None if no such event exists.
        """
        with self._lock:
            first_event_index = self.get_index_of_first_event_after(timestamp)
            return self._event_history[first_event_index] if first_event_index is not None else None

    def get_nearest_event_of_type_to_event(self, event: DetectionEvent, event_type: Type[DetectionEvent],
                                           tolerance: Optional[timedelta] = None) -> Optional[DetectionEvent]:
        """
        Returns the nearest event of a specified type to a given event.

        :param event: The reference event.
        :param event_type: The type of event to search for.
        :param tolerance: Maximum allowed time difference.

        :return: The nearest matching event or None.
        """
        return self.get_nearest_event_of_type_to_timestamp(event.timestamp, event_type, tolerance)

    def get_nearest_event_of_type_to_timestamp(self, timestamp: float, event_type: Type[DetectionEvent],
                                               tolerance: Optional[timedelta] = None) -> Optional[DetectionEvent]:
        """
        Finds the nearest event of a specified type to a timestamp.

        :param timestamp: Reference timestamp.
        :param event_type: Event type to search for.
        :param tolerance: Maximum allowed time difference.

        :return: Closest event or None.
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

    def get_nearest_event_to(self, timestamp: float, tolerance: Optional[timedelta] = None) -> Optional[DetectionEvent]:
        """
        Finds the nearest event to a timestamp.

        :param timestamp: Target timestamp.
        :param tolerance: Optional tolerance.

        :return: Nearest event or None.
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
        :param timestamp: Target timestamp.
        :param tolerance: Optional tolerance.

        :return: Index or None.
        """
        with self._lock:
            nearest_event_index = np.argmin(np.abs(time_stamps - timestamp))
            if tolerance is not None and abs(time_stamps[nearest_event_index] - timestamp) > tolerance.total_seconds():
                return None
            return nearest_event_index

    def get_nearest_event_to_event_with_conditions(self, event: DetectionEvent, conditions: Callable[[DetectionEvent], bool]) -> Optional[
        DetectionEvent]:
        """
        Finds nearest event to another event matching conditions.

        :param event: Reference event.
        :param conditions: Filtering function.

        :return: Matching event or None.
        """
        with self._lock:
            events = self.get_events_sorted_by_nearest_to_event(event)
            found_events = self.get_event_where(conditions, events=[e[0] for e in events])
            return found_events[0] if found_events else None

    def get_events_sorted_by_nearest_to_event(self, event: DetectionEvent) -> List[Tuple[DetectionEvent, float]]:
        """
        Sorts events by proximity to an event.

        :param event: Reference event.

        :return: Sorted events.
        """
        return self.get_events_sorted_by_nearest_to_timestamp(event.timestamp)

    def get_events_sorted_by_nearest_to_timestamp(self, timestamp: float) -> List[Tuple[DetectionEvent, float]]:
        """
        Sorts events by proximity to a timestamp.

        :param timestamp: Reference timestamp.

        :return: List of (event, time difference).
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            time_diff = np.abs(time_stamps - timestamp)
            events_with_time_diff = [(event, dt) for event, dt in zip(self._event_history, time_diff)]
            events_with_time_diff.sort(key=lambda e: e[1])
        return events_with_time_diff

    def get_first_event_of_type_after_event(self, event_type: Type[DetectionEvent], event: DetectionEvent) -> Optional[DetectionEvent]:
        """
        Gets the first event of a type after an event.

        :param event_type: Event type.
        :param event: Reference event.

        :return: Matching event or None.
        """
        return self.get_first_event_of_type_after_timestamp(event_type, event.timestamp)

    def get_first_event_of_type_after_timestamp(self, event_type: Type[DetectionEvent], timestamp: float) -> Optional[DetectionEvent]:
        """
        Gets first event of a type after a timestamp.

        :param event_type: Event type.
        :param timestamp: Reference timestamp.

        :return: Matching event or None.
        """
        with self._lock:
            start_index = self.get_index_of_first_event_after(timestamp)
            if start_index is not None:
                for event in self._event_history[start_index:]:
                    if isinstance(event, event_type):
                        return event

    def get_first_event_of_type_before_event(self, event_type: Type[DetectionEvent], event: DetectionEvent) -> Optional[DetectionEvent]:
        """
        Gets first event of a type before an event.

        :param event_type: Event type.
        :param event: Reference event.

        :return: Matching event or None.
        """
        return self.get_first_event_of_type_before_timestamp(event_type, event.timestamp)

    def get_first_event_of_type_before_timestamp(self, event_type: Type[DetectionEvent], timestamp: float) -> Optional[DetectionEvent]:
        """
        Gets first event of a type before a timestamp.

        :param event_type: Event type.
        :param timestamp: Reference timestamp.

        :return: Matching event or None.
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

        :return: Index or None.
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

        :return: Index or None.
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            try:
                return np.where(time_stamps < timestamp)[0][-1]
            except IndexError:
                return None

    def get_events_between_two_events(self, event1: DetectionEvent, event2: DetectionEvent) -> List[DetectionEvent]:
        """
        Gets events between two events.

        :param event1: Start event.
        :param event2: End event.

        :return: Events between them.
        """
        return [e for e in self.get_events_between_timestamps(event1.timestamp, event2.timestamp)
                if e not in [event1, event2]]

    def get_events_between_timestamps(self, timestamp1: float, timestamp2: float) -> List[DetectionEvent]:
        """
        Gets events between timestamps.

        :param timestamp1: First timestamp.
        :param timestamp2: Second timestamp.

        :return: List of events.
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
                logger.debug(f"No events between timestamps {timestamp1}, {timestamp2}")
                return []

    def get_event_where(self, conditions: Callable[[DetectionEvent], bool], events: Optional[List[DetectionEvent]] = None) -> List[DetectionEvent]:
        """
        Filters events by condition.

        :param conditions: Condition function.
        :param events: Optional event list.

        :return: Matching events.
        """
        events = events if events is not None else self._event_history
        return [event for event in events if conditions(event)]

    @property
    def time_stamps_array(self) -> np.ndarray:
        """
        Gets the time stamps as a NumPy array.

        :return: NumPy array of time stamps.
        """
        return np.array(self.time_stamps)

    @property
    def time_stamps(self) -> List[float]:
        """
        Gets the time stamps of the events.

        :return: List of time stamps.
        """
        with self._lock:
            return [event.timestamp for event in self._event_history]


# Requires some refactoring to match the SOLID principles.
class ObjectTrackerFactory:
    """
    Factory class to manage creation and access of ObjectTracker instances.

    This class is used to manage a collection of ObjectTracker instances associated
    with Body objects. It enforces synchronization to ensure thread safety and
    ensures a single ObjectTracker instance per Body object. The factory allows
    retrieving all registered ObjectTracker instances or creating/retrieving a
    specific tracker for a given Body.
    """

    _trackers: ClassVar[Dict[Body, ObjectEventTracker]] = {}
    """
    The dictionary mapping Body objects to their associated ObjectTracker instances.
    """

    _lock: ClassVar[RLock] = RLock()
    """
    threading.RLock object used for thread-safe access to the _trackers dictionary.
    """

    @classmethod
    def get_all_trackers(cls) -> List[ObjectEventTracker]:
        with cls._lock:
            return list(cls._trackers.values())

    @classmethod
    def get_tracker(cls, obj: Body) -> ObjectEventTracker:
        with cls._lock:
            if obj not in cls._trackers:
                cls._trackers[obj] = ObjectEventTracker(body=obj, context=None, _event_history=[])
            return cls._trackers[obj]


