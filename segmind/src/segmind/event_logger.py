from __future__ import annotations

import os
import queue
import threading
from collections import UserDict, defaultdict
from threading import RLock
import time
from datetime import timedelta
from os.path import dirname, abspath
import re

from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Optional, Dict, Type, TYPE_CHECKING, Callable, Tuple
from .datastructures.events import Event, EventUnion, EventWithTrackedObjects, EventWithTwoTrackedObjects, PickUpEvent, \
    InsertionEvent
from .datastructures.mixins import HasPrimaryTrackedObject
from .datastructures.object_tracker import ObjectTrackerFactory
from .utils import text_to_speech

from segmind import set_logger_level, LogLevel, logger


set_logger_level(LogLevel.DEBUG)

ConditionFunction = Callable[[EventUnion], bool]
CallbackFunction = Callable[[EventUnion], None]


class EventCallbacks(UserDict):
    """
    A dictionary that maps event types to a list of tuples each has a condition and a callback, the callback will be called when the event occurs and the condition is met.
    This modifies the setitem such that if a class or its subclass is added, the callback is also added to the subclass.
    """

    def __setitem__(self, key: Type[Event], value: List[Tuple[ConditionFunction, CallbackFunction]]):
        if key not in self:
            super().__setitem__(key, value)
        else:
            self[key].extend(value)
        for subclass in key.__subclasses__():
            self.__setitem__(subclass, value)


class EventLogger:
    """
    A class that logs events that are happening in the simulation.
    """

    current_logger: Optional[EventLogger] = None
    """
    A singleton instance of the event logger.
    """
    event_callbacks: EventCallbacks = EventCallbacks()
    """
    A dictionary that maps event types to a list of callbacks that should be called when the event occurs.
    """

    def __init__(self, annotate_events: bool = False, events_to_annotate: List[Type[Event]] = None):
        """
        Initialize the EventLogger.

        :param annotate_events: A boolean indicating whether events should be annotated.
        :param events_to_annotate: A list of event types that should be annotated if annotation is enabled.
        """
        self.timeline_per_thread = {}
        self.timeline = []
        self.event_queue = queue.Queue()
        self.timeline_lock: RLock = RLock()
        self.event_callbacks_lock: RLock = RLock()
        self.annotate_events = annotate_events
        self.events_to_annotate = events_to_annotate
        if annotate_events:
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

    def add_callback(self, event_type: Type[Event], callback: CallbackFunction, condition: Optional[ConditionFunction] = None) -> None:
        """
        Add a callback for an event type.

        :param event_type: The type of the event.
        :param callback: The callback to add.
        """
        condition = lambda event: True if condition is None else condition
        with self.event_callbacks_lock:
            self.event_callbacks[event_type] = [(condition, callback)]

    def log_event(self, event: Event):
        if self.is_event_in_timeline(event):
            return
        self.update_object_trackers_with_event(event)
        self.event_queue.put(event)
        self.annotate_scene_with_event(event)
        self.call_event_callbacks(event)

    def call_event_callbacks(self, event: Event) -> None:
        """
        Call the callbacks that are registered for the event type.

        :param event: The event to call the callbacks for.
        """
        with self.event_callbacks_lock:
            if type(event) in self.event_callbacks:
                for condition, callback in self.event_callbacks[type(event)]:
                    if condition(event):
                        callback(event)

    def annotate_scene_with_event(self, event: Event) -> None:
        """
        Annotate the scene with the event.

        :param event: The event to annotate the scene with.
        """

        if self.events_to_annotate is not None and (type(event) in self.events_to_annotate):
            logger.debug(f"Logging event: {event}")
            if self.annotation_thread is not None:
                self.annotation_queue.put(event)
                

    @staticmethod
    def update_object_trackers_with_event(event: Event) -> None:
        """
        Update the event object trackers with the event.

        :param event: The event to update the object trackers with.
        """
        if isinstance(event, EventWithTrackedObjects):
            event.update_object_trackers_with_event()

    def add_event_to_timeline_of_thread(self, event: Event) -> None:
        """
        Add an event to the timeline of the detector thread.
        :param event: The event to add.
        """
        thread_id = event.detector_thread_id
        with self.timeline_lock:
            if thread_id not in self.timeline_per_thread:
                self.timeline_per_thread[thread_id] = []
            self.timeline_per_thread[thread_id].append(event)
            self.timeline.append(event)

    def is_event_in_timeline(self, event: Event) -> bool:
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
        import plotly.graph_objects as go

        data_dict = defaultdict(list)
        for tracker in ObjectTrackerFactory.get_all_trackers():
            for event in tracker.get_event_history():
                end_timestamp = event.timestamp + timedelta(seconds=0.1).total_seconds()
                if hasattr(event, 'end_timestamp') and event.end_timestamp is not None:
                    end_timestamp = max(event.end_timestamp, end_timestamp)
                data_dict['end'].append(end_timestamp)
                data_dict['start'].append(event.timestamp)
                data_dict['event'].append(event.__class__.__name__)
                obj = event.tracked_object if isinstance(event, HasPrimaryTrackedObject) else tracker.obj
                data_dict['object'].append(obj.name)
                if isinstance(obj, Body):
                    data_dict['obj_type'].append(obj.obj_type.name)
                elif isinstance(obj, Body):
                    data_dict['obj_type'].append(f'Link of {obj.parent_entity.obj_type}')
                if isinstance(event, EventWithTwoTrackedObjects) and event.with_object is not None:
                    with_object = event.with_object
                    data_dict['with_object'].append(with_object.name)
                    if isinstance(with_object, Body):
                        data_dict['with_obj_type'].append(with_object.obj_type.name)
                    elif isinstance(with_object, Body):
                        data_dict['with_obj_type'].append(f'Link of {with_object.parent_entity.obj_type}')
                else:
                    data_dict['with_object'].append(None)
                    data_dict['with_obj_type'].append(None)
        if len(data_dict['start']) == 0:
            logger.debug("No events to plot.")
            return
        # subtract the start time from all timestamps
        min_start = min(data_dict['start'])
        data_dict['start'] = [x - min_start for x in data_dict['start']]
        data_dict['end'] = [x - min_start for x in data_dict['end']]
        tickvals = []
        prev_val = None
        range_val = max(data_dict['end']) - min(data_dict['start'])
        per_tick_range = range_val / 10
        for val in sorted(data_dict['start']):
            if prev_val is None:
                tickvals.append(val)
                prev_val = val
                continue
            if abs(val - prev_val) > per_tick_range:
                tickvals.append(val)
                prev_val = val
            else:
                tickvals.append("")
        df = pd.DataFrame(data_dict)

        fig = go.Figure()

        fig = px.timeline(df, x_start=pd.to_datetime(df[f'start'], unit='s'),
                          x_end=pd.to_datetime(df[f'end'], unit='s'),
                          y=f'event',
                          color=f'event',
                          hover_data={'object': True, 'obj_type': True, 'with_object': True, 'with_obj_type': True},
                          # text=f'object',
                          title=f"Events Timeline")
        tick_vals = [pd.to_datetime(x, unit='s') if x != "" else x for x in tickvals]
        fig.update_xaxes(tickvals=tick_vals, tickformat='%S')
        # fig.update_xaxes(tickvals=pd.to_datetime(df[f'start'], unit='s'), tickformat='%S')
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

    def get_events_per_thread(self) -> Dict[str, List[Event]]:
        """
        Get all events that have been logged.
        """
        with self.timeline_lock:
            events = self.timeline_per_thread.copy()
        return events

    def get_events(self) -> List[Event]:
        """
        Get all events that have been logged.
        """
        with self.timeline_lock:
            events = self.timeline.copy()
        return events

    def get_latest_event_of_detector_for_object(self, detector_prefix: str, obj: Body) -> Optional[Event]:
        """
        Get the latest of event of the thread that has the given prefix and object name in its id.

        :param detector_prefix: The prefix of the thread id.
        :param obj: The object that should have its name in the thread id.
        """
        thread_id = self.find_thread_with_prefix_and_object(detector_prefix, obj.name)
        return self.get_latest_event_of_thread(thread_id)

    def get_nearest_event_of_detector_for_object(self, detector_prefix: str, obj: Body,
                                                 timestamp: float) -> Optional[EventUnion]:
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

    def get_nearest_event_of_thread(self, thread_id: str, timestamp: float) -> Optional[EventUnion]:
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

    def get_latest_event_of_thread(self, thread_id: str) -> Optional[Event]:
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

    def run(self):
        while not self.kill_event.is_set() or not self.logger.annotation_queue.empty():
            try:
                event = self.logger.annotation_queue.get_nowait()
                self.logger.annotation_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)
                continue
            obj_name_map = {"montessori_object_6": "Cylinder",
                            "montessori_object_3": "Cube",
                            "montessori_object_5": "Cuboid",
                            "montessori_object_2": "Triangle", }
            try:
                if isinstance(event, PickUpEvent) and event.tracked_object.name in obj_name_map:
                    text_to_speech(f"The {obj_name_map[event.tracked_object.name]} was picked")
                elif isinstance(event, InsertionEvent) and event.tracked_object.name in obj_name_map:
                    hole_name = event.through_hole.name.replace('_', ' ').strip()
                    hole_name = re.sub(r'\d+', '', hole_name).strip()
                    text_to_speech(f"The {obj_name_map[event.tracked_object.name]} was inserted into the {hole_name}")
                event.annotate()
            except Exception as e:
                logger.debug(f"Error annotating event {event}: {e}")
                continue

    def stop(self):
        self.kill_event.set()
