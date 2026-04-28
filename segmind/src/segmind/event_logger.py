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

from segmind.datastructures.events import DetectionEvent, EventWithTrackedObjects
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Optional, Dict, Type, Callable, Tuple
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
    timeline_per_thread: Dict[str, List[DetectionEvent]] = field(default_factory=dict, init=False)
    """
    A dictionary that maps thread names to a list of events.
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
        self.timeline_per_thread = {}
        for obj_tracker in ObjectTrackerFactory.get_all_trackers():
            obj_tracker.reset()

    def add_callback(self, event_type: Type[DetectionEvent], callback: CallbackFunction, condition: Optional[ConditionFunction] = None) -> None:
        """
        Add a callback for an event type.

        :param event_type: The type of the event.
        :param callback: The callback to add.
        """
        condition = lambda event: True if condition is None else condition
        with self.event_callbacks_lock:
            self.event_callbacks[event_type] = [(condition, callback)]

    def log_event(self, event: DetectionEvent):
        if self.is_event_in_timeline(event):
            return
        self.update_object_trackers_with_event(event)
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
    def update_object_trackers_with_event(event: DetectionEvent) -> None:
        """
        Update the event object trackers with the event.

        :param event: The event to update the object trackers with.
        """
        if isinstance(event, EventWithTrackedObjects):
            event.update_object_trackers_with_event()

    def add_event_to_timeline_of_thread(self, event: DetectionEvent):
        """
        Add an event to the timeline of the detector thread.

        :param event: The event to add.
        """
        thread_id = str(event.detector_thread_id)
        with self.timeline_lock:
            if thread_id not in self.timeline_per_thread:
                self.timeline_per_thread[thread_id] = []
            self.timeline_per_thread[thread_id].append(event)
            self.timeline.append(event)

    def is_event_in_timeline(self, event: DetectionEvent) -> bool:
        """
        Check if an event is already in the timeline.

        :param event: The event to check.
        :return: True if the event is in the timeline, False otherwise.
        """
        with self.timeline_lock:
            if event in self.timeline:
                return True
            else:
                self.add_event_to_timeline_of_thread(event)


    #refactor further and create a new EventPlotter class
    def plot_events(self, show: bool = True, save_path: Optional[str] = None):
        """
        Plot all events that have been logged in a timeline.

        :param show: whether to show the plot (disable if running from docker, use the written file instead by setting
        save_path to the save path you prefer).
        :param save_path: the html plot will be save the given path, if not provided, it will not be saved.
        """
        logger.debug("Plotting events:")
        # construct a dataframe with the events
        import pandas as pd
        import plotly.express as px

        data_dict = defaultdict(list)
        for event in self.timeline:
            start_timestamp = event.timestamp
            if isinstance(start_timestamp, datetime):
                start_timestamp = start_timestamp.timestamp()
            
            end_timestamp = getattr(event, 'end_timestamp', None)
            if end_timestamp is None:
                end_timestamp = start_timestamp + 0.1
            elif isinstance(end_timestamp, datetime):
                end_timestamp = end_timestamp.timestamp()

            data_dict['start'].append(start_timestamp)
            data_dict['end'].append(end_timestamp)
            data_dict['event'].append(event.__class__.__name__)

            tracked_objects = getattr(event, 'tracked_objects', [])
            object_name = ", ".join([obj.name for obj in tracked_objects]) if tracked_objects else "None"
            data_dict['object'].append(object_name)

            obj_type = ", ".join([obj.obj_type.name for obj in tracked_objects if isinstance(obj, Body)]) if tracked_objects else "None"
            data_dict['obj_type'].append(obj_type)

            with_object = getattr(event, 'with_object', None)
            data_dict['with_object'].append(with_object.name if with_object else None)
            data_dict['with_obj_type'].append(with_object.obj_type.name if isinstance(with_object, Body) else None)

        if not data_dict['start']:
            logger.debug("No events to plot.")
            return

        # subtract the start time from all timestamps
        min_start = min(data_dict['start'])
        data_dict['start'] = [x - min_start for x in data_dict['start']]
        data_dict['end'] = [x - min_start for x in data_dict['end']]

        df = pd.DataFrame(data_dict)

        fig = px.timeline(
            df,
            x_start=pd.to_datetime(df['start'], unit='s'),
            x_end=pd.to_datetime(df['end'], unit='s'),
            y='event',
            color='event',
            hover_data={'object': True, 'obj_type': True, 'with_object': True, 'with_obj_type': True},
            title="Events Timeline"
        )

        fig.update_xaxes(tickformat='%S')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_layout(
            font_family="Courier New",
            font_color="black",
            font_size=20,
            title_font_family="Times New Roman",
            title_font_color="black",
            title_font_size=30,
            legend_title_font_color="black",
            legend_title_font_size=24,
        )

        if show:
            fig.show()
        if save_path:
            if not os.path.exists(dirname(save_path)):
                os.makedirs(dirname(save_path))
            if not save_path.endswith('.html'):
                save_path += '.html'
            file_path = abspath(save_path)
            fig.write_html(file_path)
            logger.debug(f"Plot saved to {file_path}")

    def print_events(self):
        """
        Print all events that have been logged.
        """
        logger.debug("Events:")
        logger.debug(self.__str__())

    def get_events_per_thread(self) -> Dict[str, List[DetectionEvent]]:
        """
        Get all events that have been logged.
        """
        with self.timeline_lock:
            events = self.timeline_per_thread.copy()
        return events

    def get_events(self) -> List[DetectionEvent]:
        """
        Get all events that have been logged.
        """
        with self.timeline_lock:
            events = self.timeline.copy()
        return events

    def get_latest_event_of_detector_for_object(self, detector_prefix: str, obj: Body) -> Optional[DetectionEvent]:
        """
        Get the latest of event of the thread that has the given prefix and object name in its id.

        :param detector_prefix: The prefix of the thread id.
        :param obj: The object that should have its name in the thread id.
        """
        thread_id = self.find_thread_with_prefix_and_object(detector_prefix, obj.name)
        return self.get_latest_event_of_thread(thread_id)

    def get_nearest_event_of_detector_for_object(self, detector_prefix: str, obj: Body,
                                                 timestamp: float) -> Optional[DetectionEvent]:
        """
        Get the nearest event of the thread that has the given prefix and object name in its id.

        :param detector_prefix: The prefix of the thread id.
        :param obj: The object that should have its name in the thread id.
        :param timestamp: The timestamp of the event.
        """
        thread_id = self.find_thread_with_prefix_and_object(detector_prefix, obj.name)
        return self.get_nearest_event_of_thread(thread_id, timestamp)

    def find_thread_with_prefix_and_object(self, prefix: str, object_name: str) -> Optional[str]:
        """
        Find the thread id that has the given prefix and object name in its id.

        :param prefix: The prefix of the thread id.
        :param object_name: The object name that should be in the thread id.
        :return: The id of the thread or None if no such thread
        """
        with self.timeline_lock:
            thread_id = [thread_id for thread_id in self.timeline_per_thread.keys() if thread_id.startswith(prefix) and
                         object_name in thread_id]
        return None if len(thread_id) == 0 else thread_id[0]

    def get_nearest_event_of_thread(self, thread_id: str, timestamp: float) -> Optional[DetectionEvent]:
        """
        Get the nearest event of the thread with the given id.

        :param thread_id: The id of the thread.
        :param timestamp: The timestamp of the event.
        :return: The nearest event of the thread or None if no such thread.
        """
        with self.timeline_lock:
            if thread_id not in self.timeline_per_thread:
                return None
            all_event_timestamps = [(event, event.timestamp) for event in self.timeline_per_thread[thread_id]]
            return min(all_event_timestamps, key=lambda x: abs(x[1] - timestamp))[0]

    def get_latest_event_of_thread(self, thread_id: str) -> Optional[DetectionEvent]:
        """
        Get the latest event of the thread with the given id.

        :param thread_id: The id of the thread.
        :return: The latest event of the thread or None if no such thread.
        """
        with self.timeline_lock:
            if thread_id not in self.timeline_per_thread:
                return None
            return self.timeline_per_thread[thread_id][-1]

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
