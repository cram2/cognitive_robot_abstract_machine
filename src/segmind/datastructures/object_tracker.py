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

        Returns
        -------
        bool
            Indicates True if the object is moving, otherwise False.
        """
        with self._lock:
            return self.context.object_moving_status[self.body]


    def add_event(self, event: Event):
        """
        Adds an event to the event history in a thread-safe manner.

        The event is appended to the event history, and the history is
        then sorted based on the timestamp of each event.

        Parameters:
            event (Event): The event to be added to the history.
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

        Returns:
            List[Event]: A list of events recorded in the event history.
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

        Returns
        -------
        Optional[Event]
            The latest event from the event history, or None if the
            history is empty.
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

        Args:
            event_type (Type[Event]): The type of the event to search for.

        Returns:
            Optional[Event]: The most recent event of the specified type if found,
            otherwise None.
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

        Parameters:
        timestamp (float): The reference timestamp to compare against.

        Returns:
        Optional[Event]: The first event occurring before the given timestamp, or
        None if no such event exists.
        """
        with self._lock:
            first_event_index = self.get_index_of_first_event_before(timestamp)
            return self._event_history[first_event_index] if first_event_index is not None else None

    def get_first_event_after(self, timestamp: float) -> Optional[Event]:
        """
        Gets the first event that occurs after the given timestamp.

        Uses the provided timestamp to locate the first event in the event
        history that occurs after the specified time.

        Parameters:
        timestamp: float
            The reference timestamp, in seconds, after which the event is
            searched for.

        Returns:
        Optional[Event]
            The first event occurring after the given timestamp, or None if
            no such event exists.
        """
        with self._lock:
            first_event_index = self.get_index_of_first_event_after(timestamp)
            return self._event_history[first_event_index] if first_event_index is not None else None

    def get_nearest_event_of_type_to_event(self, event: Event, event_type: Type[Event],
                                           tolerance: Optional[timedelta] = None) -> Optional[EventUnion]:
        """
        Returns the nearest event of a specified type to a given event.

        This method finds and returns the nearest event of a given event type that is
        close to the provided event's timestamp. Optionally, a tolerance can be
        specified to limit the search to a specific time range.

        Parameters:
        event: Event
            The reference event whose timestamp will be used for finding the nearest
            event.
        event_type: Type[Event]
            The type of event to search for.
        tolerance: Optional[timedelta]
            A timedelta object specifying the maximum allowable time difference
            between the reference event's timestamp and the candidate event. If None,
            no tolerance limit will be applied.

        Returns:
        Optional[EventUnion]
            The nearest event of the specified type to the provided event. Returns None
            if no event of the specified type is found within the given tolerance.
        """
        return self.get_nearest_event_of_type_to_timestamp(event.timestamp, event_type, tolerance)

    def get_nearest_event_of_type_to_timestamp(self, timestamp: float, event_type: Type[Event],
                                               tolerance: Optional[timedelta] = None) -> Optional[Event]:
        """
        Finds the nearest event of a specified type to a given timestamp within an optional tolerance.

        This method searches for the event of the specified type in the event history whose
        timestamp is closest to the provided timestamp. An optional tolerance can be specified
        to limit the search to only events within the defined time range. If a matching event is
        found, it returns the event, otherwise returns None.

        Parameters:
            timestamp (float): The reference timestamp to which the nearest event is to be found.
            event_type (Type[Event]): The type of event to search for in the event history.
            tolerance (Optional[timedelta]): The maximum allowed time difference between the
                                              reference timestamp and the event's timestamp. If
                                              None, no tolerance is applied.

        Returns:
            Optional[Event]: The event of the specified type closest to the given timestamp, or
                             None if no such event is found.
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
        Finds the nearest event to the given timestamp.

        This method searches for an event in the stored event history whose
        timestamp is closest to the provided timestamp. An optional tolerance
        can be specified to restrict the maximum difference between the
        timestamp and the nearest event's timestamp.

        Parameters:
            timestamp (float): The target timestamp to which the nearest event
                should be identified.
            tolerance (Optional[timedelta], optional): The maximum allowable
                difference between the given timestamp and the nearest event's
                timestamp. Default is None, meaning no restriction.

        Returns:
            Optional[Event]: The event from the event history that is closest
                to the provided timestamp. Returns None if no such event is
                found or if the tolerance is exceeded.
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            nearest_event_index = self._get_nearest_index(time_stamps, timestamp, tolerance)
            if nearest_event_index is not None:
                return self._event_history[nearest_event_index]

    def _get_nearest_index(self, time_stamps: np.ndarray,
                           timestamp: float, tolerance: Optional[timedelta] = None) -> Optional[int]:
        """
        Finds the nearest index in a sorted array of timestamps to a given timestamp.

        This method determines the index of the timestamp in the provided array that is
        closest to the given timestamp. If a tolerance is specified, the method ensures
        that the nearest timestamp is within the tolerance range. If no timestamp meets
        this condition, the method returns None.

        Parameters:
        time_stamps (np.ndarray): A sorted array of timestamps.
        timestamp (float): The target timestamp to find the nearest value for.
        tolerance (Optional[timedelta]): A tolerance value within which the
            difference between the nearest timestamp and the given timestamp is
            acceptable. Defaults to None.

        Returns:
        Optional[int]: The index of the nearest timestamp if within the allowed
            tolerance, otherwise None.
        """
        with self._lock:
            nearest_event_index = np.argmin(np.abs(time_stamps - timestamp))
            if tolerance is not None and abs(time_stamps[nearest_event_index] - timestamp) > tolerance.total_seconds():
                return None
            return nearest_event_index

    def get_nearest_event_to_event_with_conditions(self, event: Event, conditions: Callable[[Event], bool]) -> Optional[Event]:
        """
        Finds the nearest event to a given event that satisfies the specified conditions.

        The method identifies events sorted by their proximity to the given event and
        applies the provided conditions to filter them. If no event satisfies the
        conditions, None is returned.

        Args:
            event: The reference event to which the proximity is evaluated.
            conditions: A callable function that determines whether an event satisfies
                the specified conditions.

        Returns:
            An Event object that is nearest to the specified event and satisfies the
            conditions, or None if no such event is found.
        """
        with self._lock:
            events = self.get_events_sorted_by_nearest_to_event(event)
            found_events = self.get_event_where(conditions, events=[e[0] for e in events])
            if len(found_events) == 0:
                return None
            else:
                return found_events[0]

    def get_events_sorted_by_nearest_to_event(self, event: Event):
        """
        Gets events sorted by their proximity to a specified event.

        This method retrieves events and sorts them based on their
        closeness in time to the provided event's timestamp. It
        uses the `event.timestamp` attribute to determine the
        proximity of other events.

        Args:
            event (Event): The event whose timestamp will be used to
            calculate proximity for sorting.

        Returns:
            list: A list of events sorted by their closeness to the
            specified event's timestamp.
        """
        return self.get_events_sorted_by_nearest_to_timestamp(event.timestamp)

    def get_events_sorted_by_nearest_to_timestamp(self, timestamp: float) -> List[Tuple[Event, float]]:
        """
        Returns a list of events sorted by their temporal proximity to the given timestamp.

        This method computes the absolute difference between the provided timestamp and
        each event's timestamp, then sorts the events in ascending order of those
        differences. It ensures thread safety while accessing shared data and the
        returned list contains tuples where each tuple consists of an event and its
        corresponding time difference.

        Parameters:
            timestamp (float): The reference timestamp to evaluate temporal proximity
                               for the events.

        Returns:
            List[Tuple[Event, float]]: A list of tuples where each tuple contains an
                                       event and the absolute time difference between
                                       the event's timestamp and the provided timestamp.
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            time_diff = np.abs(time_stamps - timestamp)
            events_with_time_diff = [(event, dt) for event, dt in zip(self._event_history, time_diff)]
            events_with_time_diff.sort(key=lambda e: e[1])
        return events_with_time_diff

    def get_first_event_of_type_after_event(self, event_type: Type[Event], event: Event) -> Optional[EventUnion]:
        """
        Gets the first event of a specified type that occurs after the given event.

        This method searches for an event of the specified type that has a timestamp
        occurring after the given event's timestamp. If no such event exists, it
        returns None.

        Args:
            event_type: The type of event to search for.
            event: The reference event after which the search is performed.

        Returns:
            The first event of the specified type occurring after the given event, or
            None if no matching event is found.
        """
        return self.get_first_event_of_type_after_timestamp(event_type, event.timestamp)

    def get_first_event_of_type_after_timestamp(self, event_type: Type[Event], timestamp: float) -> Optional[Event]:
        """
        Retrieves the first event of a specified type that occurs after a given timestamp.

        Searches through an event history starting from the first event that occurs
        after the provided timestamp. Returns the first event of the specified type
        if found; otherwise, returns None.

        Args:
            event_type (Type[Event]): The type of event to search for.
            timestamp (float): The timestamp after which the search should begin.

        Returns:
            Optional[Event]: The first event of the specified type occurring after
            the given timestamp, or None if no such event exists.
        """
        with self._lock:
            start_index = self.get_index_of_first_event_after(timestamp)
            if start_index is not None:
                for event in self._event_history[start_index:]:
                    if isinstance(event, event_type):
                        return event

    def get_first_event_of_type_before_event(self, event_type: Type[Event], event: Event) -> Optional[EventUnion]:
        """
        Returns the first event of a specified type that occurred before a given event.

        This method retrieves the earliest event of a specified type that occurred prior to the timestamp of the provided
        event. If no such event is found, the method returns None.

        Parameters:
        event_type: Type[Event]
            The type of event to search for.
        event: Event
            The reference event to determine the timestamp before which the search is conducted.

        Returns:
        Optional[EventUnion]
            The first event of the specified type found before the given event, or None if no such event exists.
        """
        return self.get_first_event_of_type_before_timestamp(event_type, event.timestamp)

    def get_first_event_of_type_before_timestamp(self, event_type: Type[Event], timestamp: float) -> Optional[Event]:
        """
        Retrieves the first event of a specified type that occurred before the given
        timestamp. This function searches the event history in reverse order, starting
        from the most recent event before the provided timestamp.

        Parameters:
        event_type: Type[Event]
            The type of event to look for in the event history.
        timestamp: float
            The reference timestamp. The function looks for the first event that
            occurred before this time.

        Returns:
        Optional[Event]
            The first event of the specified type that occurred before the given
            timestamp, or None if no such event is found.
        """
        with self._lock:
            start_index = self.get_index_of_first_event_before(timestamp)
            if start_index is not None:
                for event in reversed(self._event_history[:min(start_index+1, len(self._event_history))]):
                    if isinstance(event, event_type):
                        return event

    def get_index_of_first_event_after(self, timestamp: float) -> Optional[int]:
        """
        Returns the index of the first event occurring after the specified timestamp.

        This method searches through an array of event timestamps and identifies the
        index of the first event that occurs strictly after the provided timestamp. If
        no such event is found, it returns None.

        Parameters:
        timestamp: float
            The timestamp to compare events against.

        Returns:
        Optional[int]
            The index of the first event occurring after the specified timestamp, or
            None if no such event exists.
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            try:
                return np.where(time_stamps > timestamp)[0][0]
            except IndexError:
                logger.error(f"No events after timestamp {timestamp}")
                return None

    def get_index_of_first_event_before(self, timestamp: float) -> Optional[int]:
        """
        Finds the index of the first event that occurred before the given timestamp.

        This method iterates through the `time_stamps_array` to check if there are
        timestamps that are less than the specified `timestamp`. It returns the
        index of the most recent such event or `None` if no such event exists.

        Args:
            timestamp: Timestamp to compare against the elements in the
            `time_stamps_array`.

        Returns:
            The index of the first event occurring before the provided
            timestamp or `None` if no such event is found.
        """
        with self._lock:
            time_stamps = self.time_stamps_array
            try:
                return np.where(time_stamps < timestamp)[0][-1]
            except IndexError:
                logger.error(f"No events before timestamp {timestamp}")
                return None

    def get_events_between_two_events(self, event1: Event, event2: Event) -> List[Event]:
        """
        Returns a list of events that occur between two specified events, excluding the boundary events.

        This method identifies all the events that have timestamps falling between the timestamps
        of the two given events. The returned list excludes the starting and ending events provided
        as inputs.

        Parameters:
        event1 (Event): The starting event. Its timestamp defines the lower bound of the time range.
        event2 (Event): The ending event. Its timestamp defines the upper bound of the time range.

        Returns:
        List[Event]: A list of Event objects that are between the timestamps of event1 and event2,
        excluding event1 and event2 themselves.
        """
        return [e for e in self.get_events_between_timestamps(event1.timestamp, event2.timestamp)
                if e not in [event1, event2]]

    def get_events_between_timestamps(self, timestamp1: float, timestamp2: float) -> List[Event]:
        """
        Retrieves events between two specified timestamps.

        This method searches within the stored event history for events whose
        timestamps fall within the specified range. The timestamps are evaluated
        in ascending order, so the order of the input timestamps does not matter.

        Parameters:
            timestamp1 (float): The first timestamp defining the range.
            timestamp2 (float): The second timestamp defining the range.

        Returns:
            List[Event]: A list of events occurring between the specified timestamps.
            If no events exist in the given range, an empty list is returned.
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
        Filters events based on a given condition.

        This method iterates through a list of events and checks
        each event against a provided condition. It returns a list
        containing all events that meet the specified condition. If no
        event list is provided, the method operates on the default event
        history.

        Parameters:
        conditions (Callable[[Event], bool]): A callable function that takes
            an Event object and returns a boolean indicating whether the
            event meets the condition.
        events (Optional[List[Event]]): A list of events to filter. If None,
            the default event history is used.

        Returns:
        List[Event]: A list of events that satisfy the specified condition.
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


