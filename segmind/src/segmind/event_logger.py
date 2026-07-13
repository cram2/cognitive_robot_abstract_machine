from __future__ import annotations

import logging
import os
import queue
import threading
from collections import UserDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from os.path import dirname, abspath
from threading import RLock

from segmind.datastructures.event_plotter import EventPlotter
from segmind.datastructures.events import DetectionEvent, EventWithTrackedObjects
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from typing_extensions import List, Optional, Type, Callable, Tuple
from typing import ClassVar

logger = logging.getLogger(__name__)


ConditionFunction = Callable[[DetectionEvent], bool]
CallbackFunction = Callable[[DetectionEvent], None]


class EventCallbacks(UserDict):
    """
    A dictionary that maps event types to a list of tuples each has a condition and a callback, the callback will be called when the event occurs and the condition is met.
    This modifies the setitem such that if a class or its subclass is added, the callback is also added to the subclass.
    """

    def __setitem__(self, key: Type[DetectionEvent], value: List[Tuple[ConditionFunction, CallbackFunction]]):
        if key not in self:
            super().__setitem__(key, value)
        else:
            self[key].extend(value)
        for subclass in key.__subclasses__():
            self.__setitem__(subclass, value)


@dataclass
class EventLogger:
    """
    A class that logs events that are happening in the simulation.
    """

    annotate_events: bool = field(default=False)
    """
    A boolean indicating whether events should be annotated.
    """
    events_to_annotate: List[Type[DetectionEvent]] = field(default=None)
    """
    A list of event types that should be annotated if annotation is enabled.
    """
    current_logger: ClassVar[Optional[EventLogger]] = None
    """
    A singleton instance of the event logger.
    """
    event_callbacks: EventCallbacks = field(default_factory=EventCallbacks)
    """
    A dictionary that maps event types to a list of callbacks that should be called when the event occurs.
    """
    timeline: List[DetectionEvent] = field(default_factory=list, init=False)
    """
    A list of all events.
    """
    event_queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    """
    A queue of events that should be processed.
    """
    timeline_lock: RLock = field(default_factory=RLock, init=False)
    """
    A lock for the timeline.
    """
    event_callbacks_lock: RLock = field(default_factory=RLock, init=False)
    """
    A lock for the event callbacks.
    """
    annotation_queue: Optional[queue.Queue] = field(default=None, init=False)
    """
    A queue for annotation events.
    """
    annotation_thread: Optional[EventAnnotationThread] = field(default=None, init=False)
    """
    A thread for annotating events.
    """

    def __post_init__(self):
        """
        Initialize the EventLogger.
        """
        if self.annotate_events:
            self.annotation_queue = queue.Queue()
            self.annotation_thread = EventAnnotationThread(self)
            self.annotation_thread.start()
        if EventLogger.current_logger is None:
            EventLogger.current_logger = self

    def reset(self):
        self.timeline = []
        self.event_queue = queue.Queue()
        for obj_tracker in ObjectTrackerFactory.get_all_trackers():
            obj_tracker.reset()

    def add_callback(self, event_type: Type[DetectionEvent], callback: CallbackFunction, condition: Optional[ConditionFunction] = None) -> None:
        """
        Add a callback for an event type.

        :param event_type: The type of the event.
        :param callback: The callback to add.
        """
        if condition is None:
            condition = lambda event: True
        with self.event_callbacks_lock:
            self.event_callbacks[event_type] = [(condition, callback)]

    def log_event(self, event: DetectionEvent, factory: ObjectTrackerFactory):
        """
        Log an event, ignoring it if it is already in the timeline.

        :param event: The event to log.
        :param factory: The object tracker factory.
        """
        with self.timeline_lock:
            if event in self.timeline:
                return
            self.timeline.append(event)
        self.update_object_trackers_with_event(event, factory)
        self.event_queue.put(event)
        self.annotate_scene_with_event(event)
        self.call_event_callbacks(event)

    def call_event_callbacks(self, event: DetectionEvent) -> None:
        """
        Call the callbacks that are registered for the event type.

        :param event: The event to call the callbacks for.
        """
        with self.event_callbacks_lock:
            if type(event) in self.event_callbacks:
                for condition, callback in self.event_callbacks[type(event)]:
                    if condition(event):
                        callback(event)

    def annotate_scene_with_event(self, event: DetectionEvent) -> None:
        """
        Annotate the scene with the event.

        :param event: The event to annotate the scene with.
        """

        if self.events_to_annotate is not None and (type(event) in self.events_to_annotate):
            logger.debug(f"Logging event: {event}")
            if self.annotation_thread is not None:
                self.annotation_queue.put(event)
                

    @staticmethod
    def update_object_trackers_with_event(event: DetectionEvent, factory: ObjectTrackerFactory) -> None:
        """
        Update the event object trackers with the event.

        :param event: The event to update the object trackers with.
        :param factory: The object tracker factory.
        """
        if isinstance(event, EventWithTrackedObjects):
            event.update_object_trackers_with_event(factory)

    def plot_events(self, show: bool = True, save_path: Optional[str] = None):
        """
        Plot all events that have been logged in a timeline.
        """
        plotter = EventPlotter()
        plotter.plot(self.get_events(), show=show, save_path=save_path)

    def print_events(self):
        """
        Print all events that have been logged.
        """
        logger.debug("Events:")
        logger.debug(self.__str__())

    def get_events(self) -> List[DetectionEvent]:
        """
        Get all events that have been logged.
        """
        with self.timeline_lock:
            events = self.timeline.copy()
        return events

    def get_next_event(self):
        """
        Get the next event from the event queue.
        """
        try:
            event = self.event_queue.get(block=False)
            self.event_queue.task_done()
            return event
        except queue.Empty:
            return None

    def join(self):
        """
        Wait for all events to be processed and all annotations to be added.
        """
        if self.annotation_thread is not None:
            self.annotation_thread.stop()
            self.annotation_thread.join()
            while self.annotation_queue.unfinished_tasks > 0:
                event = self.annotation_queue.get_nowait()
                logger.debug(f"Left out annotation for event: {event}")
                self.annotation_queue.task_done()
            self.annotation_queue.join()
        self.event_queue.join()

    def __str__(self):
        return '\n'.join([str(event) for event in self.get_events()])


class EventAnnotationThread(threading.Thread):
    def __init__(self, logger: EventLogger):
        super().__init__()
        self.logger = logger
        #self.current_annotations: List[TextAnnotation] = []
        self.kill_event = threading.Event()

    def stop(self):
        self.kill_event.set()
