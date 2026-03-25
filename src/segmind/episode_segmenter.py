import datetime
import threading
from os.path import dirname
from pathlib import Path

from pandas.core.internals import DataManager

from giskardpy.executor import Pacer, SimulationPacer
from krrood.symbolic_math.symbolic_math import FloatVariable
from segmind import logger, set_logger_level, LogLevel
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from .datastructures.object_tracker import ObjectTracker
from .detectors.atomic_event_detectors_nodes import SegmindContext
from .detectors.coarse_event_detectors import *
from .detectors.spatial_relation_detector import SpatialRelationDetector
from .episode_player import EpisodePlayer
from .event_logger import EventLogger

set_logger_level(LogLevel.DEBUG)


class EpisodeSegmenter(ABC):

    def __init__(
        self,
        episode_player: EpisodePlayer,
        world: Optional[World] = None,
        detectors_to_start: Optional[List[Type[DetectorWithStarterEvent]]] = None,
        initial_detectors: Optional[List[Type[AtomicEventDetector]]] = None,
        annotate_events: bool = False,
        plot_timeline: bool = False,
        show_plots: bool = False,
        plot_save_path: Optional[str] = None,
    ):
        """
        Initializes the EpisodeSegmenter class.

        :param episode_player: The thread that plays the episode and generates the motion.
        :param detectors_to_start: The list of event detectors that should be started.
        :param annotate_events: A boolean value that indicates if the events should be annotated.
        :param plot_timeline: A boolean value that indicates if the events timeline should be plotted.
        :param show_plots: A boolean value that indicates if the plots should be shown.
        :param plot_save_path: The path where the plots should be saved.
        """
        self._detectors_to_start: List[Type[DetectorWithStarterEvent]] = []
        self._initial_detectors: List[Type[AtomicEventDetector]] = []
        self.episode_player: EpisodePlayer = episode_player
        self.world: World = world
        self.logger = EventLogger(annotate_events)
        self.detectors_to_start: List[Type[DetectorWithStarterEvent]] = (
            detectors_to_start if detectors_to_start else []
        )
        self.initial_detectors: List[Type[AtomicEventDetector]] = (
            initial_detectors if initial_detectors else []
        )
        self.objects_to_avoid = ["particle", "floor", "kitchen"]
        self.starter_event_to_detector_thread_map: Dict[
            Tuple[Event, Type[DetectorWithStarterEvent]], DetectorWithStarterEvent
        ] = {}
        self.detector_threads_list: List[EventDetectorUnion] = []
        self.object_trackers: Dict[Body, ObjectTracker] = {}
        self.plot_timeline = plot_timeline
        self.show_plots = show_plots
        self.plot_save_path = plot_save_path
        self.kill_event: threading.Event = threading.Event()

    @property
    def detectors_to_start(self) -> List[Type[DetectorWithStarterEvent]]:
        """
        :return: The list of event detectors that should be started.
        """
        return self._detectors_to_start

    @detectors_to_start.setter
    def detectors_to_start(self, detectors: List[Type[DetectorWithStarterEvent]]):
        """
        Set the list of event detectors that should be started.

        :param detectors: The list of event detectors to set.
        """
        self._detectors_to_start = detectors
        self.update_events_to_annotate()

    @property
    def initial_detectors(self) -> List[Type[AtomicEventDetector]]:
        """
        :return: The list of initial event detectors that should be started.
        """
        return self._initial_detectors

    @initial_detectors.setter
    def initial_detectors(self, detectors: List[Type[AtomicEventDetector]]):
        """
        Set the list of initial event detectors that should be started.

        :param detectors: The list of initial event detectors to set.
        """
        self._initial_detectors = detectors
        self.update_events_to_annotate()

    def update_events_to_annotate(self):
        event_types = {
            event_type
            for detector in self.detectors_to_start + self.initial_detectors
            for event_type in detector.event_types()
        }
        self.logger.events_to_annotate = list(event_types)

    def reset(self, reset_logger: bool = True) -> None:
        # self.join_detectors()
        self.starter_event_to_detector_thread_map = {}
        self.detector_threads_list = []
        time.sleep(0.5)
        if reset_logger:
            self.logger.reset()

    def join_detectors(self, atomic_only: bool = False) -> None:
        atomic_detectors = [
            detector
            for detector in self.detector_threads_list
            if not isinstance(
                detector, (DetectorWithStarterEvent, SpatialRelationDetector)
            )
        ]
        if atomic_only:
            non_atomic_detectors = []
        else:
            non_atomic_detectors = [
                detector
                for detector in self.detector_threads_list
                if isinstance(detector, DetectorWithStarterEvent)
            ]
        for detector_thread in atomic_detectors + non_atomic_detectors:
            detector_thread.stop()
            logger.debug(f"Joining {detector_thread.thread_id}, {detector_thread.name}")
            detector_thread.join()

    def start(self) -> None:
        """
        Start the episode player and the event detectors.
        """
        logger.debug(
            f"Starting episode segmenter with detectors {self.detectors_to_start} and initial detectors {self.initial_detectors}."
        )
        self.start_episode_player_and_wait_till_ready()
        self.run_event_detectors()

    def stop(self) -> None:
        self.kill_event.set()

    def start_episode_player_and_wait_till_ready(self) -> None:
        """
        Start the Episode player thread, and waits until the thread signals that it is
        ready (e.g., the replay environment is initialized with all objects in starting poses).
        """
        if not self.episode_player.is_alive():
            # ToDO: Why run works but start does not???
            self.episode_player.start()
        while not self.episode_player.ready:
            time.sleep(0.1)

    def run_event_detectors_sequential(self, tracked_objs: List[Body] = []):
        for obj in tracked_objs:
            translation_detector = TranslationDetector(
                logger=self.logger, tracked_object=obj
            )

    def run_event_detectors(self) -> None:
        """
        Run the event detectors on the motion generated by the motion generator thread.
        """
        self.create_detector_and_start_it(
            NewObjectDetector, avoid_objects=self.avoid_object
        )
        self.run_initial_event_detectors()
        closed_threads = False

        while (not closed_threads) or (self.logger.event_queue.unfinished_tasks > 0):
            if (
                not self.episode_player.is_alive() or self.kill_event.is_set()
            ) and not closed_threads:
                self.join_detectors(atomic_only=True)
                closed_threads = True

            next_event = self.logger.get_next_event()

            if next_event is None:
                time.sleep(0.01)
                continue

            self.process_event(next_event)

        if self.plot_timeline:
            self.logger.plot_events(show=self.show_plots, save_path=self.plot_save_path)

        self.join_detectors()
        self.join()

    def process_event(self, event: EventUnion) -> None:
        """
        Process the event generated by the event logger, and start the detector threads that are triggered by the event.

        :param event: The event that should be processed.
        """
        self._process_event(event)
        self.start_triggered_detectors(event)

    def start_triggered_detectors(self, event: EventUnion) -> None:
        """
        Start the detector threads that are triggered by the event.

        :param event: The event that triggers the detector threads.
        """
        for event_detector in self.detectors_to_start:
            if event_detector.start_condition_checker(event):
                self.start_detector_thread_for_starter_event(event, event_detector)

    @abstractmethod
    def _process_event(self, event: Event) -> None:
        """
        Process the event generated by the event logger.

        :param event: The event that should be processed.
        """
        pass

    def run_initial_event_detectors(self) -> None:
        """
        Run the initial event detectors on the episode played by the episode player thread.
        """
        for detector in self.initial_detectors:
            self.create_detector_and_start_it(detector)
        self._run_initial_event_detectors()

    @abstractmethod
    def _run_initial_event_detectors(self) -> None:
        """
        Run the initial event detectors on the episode played by the episode player thread.
        """
        pass

    def run_objects_initial_event_detectors(self) -> None:
        """
        Start the motion detection threads for the objects in the world, and imagine supports for objects
        that seem to be supported but have no actual support detected.
        """
        self.episode_player.pause()
        set_of_objects = set()
        for obj in self.episode_player.world.bodies:
            if not self.avoid_object(obj):
                set_of_objects.add(obj)
                # if not check_if_object_is_supported(obj):
                #     logdebug(f"Object {obj.name} is not supported.")
                #     if World.current_world.conf.supports_spawning:
                #         Imaginator.imagine_support_for_object(obj)
                #         logdebug(f"Imagined support for object {obj.name}.")
        for obj in set_of_objects:
            self.start_motion_threads_for_object(obj)
            self.start_contact_threads_for_object(obj)
        self.episode_player.resume()

    def update_tracked_objects(self, event: EventUnion) -> None:
        """
        Update the tracked objects based on the event, for example a contact event would reveal new objects that should
        be tracked when an already tracked object comes into contact with a new object.

        :param event: The event that was triggered.
        """
        involved_objects = self.get_involved_bodies(event)
        if not involved_objects:
            return
        logger.debug(f"Involved objects: {[obj.name for obj in involved_objects]}")
        for obj in involved_objects:
            if self.avoid_object(obj):
                continue
            if obj in self.object_trackers.keys():
                continue
            if obj not in self.object_trackers.keys():
                logger.debug(f"New object {obj.name}")
                self.object_trackers[obj] = ObjectTracker(obj)
                self.start_tracking_threads_for_new_object_and_event(obj, event)

    @abstractmethod
    def start_tracking_threads_for_new_object_and_event(
        self, new_object: Body, event: EventUnion
    ):
        """
        Start the tracking threads for the new object, these threads are used to track the object's motion or contacts
         for example.

        :param new_object: The new object that should be tracked.
        :param event: The event that triggered the tracking.
        """
        pass

    def get_involved_bodies(self, event: EventUnion) -> list[Body] | None:
        """
        Get the bodies involved in the event.

        :param event: The event that involves the objects.
        :return: A list of Object instances that are involved in the event.
        """
        if isinstance(event, EventWithTrackedObjects):
            return event.involved_bodies

    def avoid_object(self, obj: Body) -> bool:
        """
        Check if the object should be avoided.

        :param obj: The object to check.
        :return: True if the object should be avoided, False otherwise.
        """
        logger.debug(f"Checking if object {obj.name} should be avoided.")
        return not type(obj.parent_connection) == Connection6DoF
        # return ((obj.is_an_environment or issubclass(obj.ontology_concept, (Supporter, Location)) or
        #         any([k in obj.name.lower() for k in self.objects_to_avoid])) or
        #        (isinstance(obj, Body) and self.avoid_object(obj.parent_entity)))

    def start_motion_threads_for_object(
        self, obj: Body, event: Optional[NewObjectEvent] = None
    ) -> None:
        """
        Start the motion detection threads for the object.

        :param obj: The Object instance for which the motion detection threads are started.
        :param event: The NewObjectEvent instance that represents the creation of the object.
        """
        for detector in (TranslationDetector,):  # RotationDetector):
            self.create_detector_and_start_it(
                detector,
                tracked_object=obj,
                starter_event=event,
                time_between_frames=self.time_between_frames,
            )

    def start_contact_threads_for_object(
        self, obj: Body, event: Optional[CloseContactEvent] = None
    ) -> None:
        """
        Start the contact threads for the object and updates the tracked objects.

        :param obj: The Object instance for which the contact threads are started.
        :param event: The ContactEvent instance that represents the contact event with the object.
        """
        for detector in (ContactDetector, LossOfContactDetector):
            self.create_detector_and_start_it(
                detector, tracked_object=obj, starter_event=event
            )

    def start_detector_thread_for_starter_event(
        self, starter_event: EventUnion, detector_type: TypeEventDetectorUnion
    ):
        """
        Start the detector thread for the given starter event.

        :param starter_event: The event that starts the detector thread.
        :param detector_type: The type of the detector.
        """
        if not self.is_detector_redundant(detector_type, starter_event):
            if detector_type == PlacingDetector:
                logger.debug(
                    f"new placing detector for object {starter_event.tracked_object.name}"
                )
            self.create_detector_and_start_it(
                detector_type, starter_event=starter_event
            )

    @staticmethod
    def ask_now(case_dict):
        detector_type = case_dict["detector_type"]
        starter_event = case_dict["starter_event"]
        self_ = case_dict["self_"]
        output_ = case_dict["output_"]
        if issubclass(detector_type, GeneralPickUpDetector):
            if isinstance(starter_event, LossOfContactEvent):
                pick_up_detectors = [
                    detector
                    for (
                        _,
                        _,
                    ), detector in self_.starter_event_to_detector_thread_map.items()
                    if isinstance(detector, GeneralPickUpDetector)
                ]
                if len(pick_up_detectors) > 0 and not output_:
                    return True
        return False

    redundant_detector_rdr = RDRDecorator(
        f"{dirname(__file__)}/rdrs",
        (bool,),
        True,
        fit=False,
        fitting_decorator=EpisodePlayer.pause_resume,
        ask_now=ask_now,
    )

    @redundant_detector_rdr.decorator
    def is_detector_redundant(
        self, detector_type: TypeEventDetectorUnion, starter_event: EventUnion
    ) -> bool:
        """
        Check if the detector is redundant.

        :param detector_type: The type of the detector.
        :param starter_event: The event that starts the detector thread.
        :return: True if the detector is redundant, False otherwise.
        """
        if (
            starter_event,
            detector_type,
        ) in self.starter_event_to_detector_thread_map.keys():
            detector = self.starter_event_to_detector_thread_map[
                (starter_event, detector_type)
            ]
            if detector.is_alive() or (detector.detected_before and detector.run_once):
                return True
        return False

    def create_detector_and_start_it(
        self,
        detector_type: TypeEventDetectorUnion,
        tracked_object: Optional[Body] = None,
        starter_event: Optional[EventUnion] = None,
        *detector_args,
        **detector_kwargs,
    ) -> None:
        """
        Start and add an event detector to the detector threads.

        :param detector_type: The event detector to be started and added.
        :param tracked_object: The object to be tracked by the detector.
        :param starter_event: The event that starts the detector thread.
        :param detector_args: The positional arguments to be passed to the detector constructor.
        :param detector_kwargs: The keyword arguments to be passed to the detector constructor.
        """
        detector_kwargs = self.get_detector_args(
            detector_type,
            tracked_object=tracked_object,
            starter_event=starter_event,
            **detector_kwargs,
        )
        detector_kwargs["episode_player"] = self.episode_player
        detector_kwargs["world"] = self.episode_player.world
        detector_kwargs["logger"] = self.logger
        detector = detector_type(**detector_kwargs)
        self.starter_event_to_detector_thread_map[(starter_event, detector_type)] = (
            detector
        )
        self.start_and_add_detector(detector)

    def start_and_add_detector(self, detector: EventDetectorUnion) -> None:
        """
        Start and add an event detector to the detector threads.

        :param detector: The event detector to be started and added.
        """
        detector.start()
        self.detector_threads_list.append(detector)
        logger.debug(f"Created {type(detector).__name__}")
        if (
            isinstance(detector, DetectorWithStarterEvent)
            and detector.starter_event is not None
        ):
            logger.debug(f"For starter event {detector.starter_event}")

    @staticmethod
    def get_detector_args(
        detector_type: TypeEventDetectorUnion,
        tracked_object: Optional[Body] = None,
        starter_event: Optional[EventUnion] = None,
        **other_detector_kwargs,
    ):
        """
        Get the detector arguments from the tracked object and/or the starter event.

        :param detector_type: The type of the detector.
        :param tracked_object: The object to be tracked by the detector.
        :param starter_event: The event that starts the detector thread.
        :param other_detector_args: The positional arguments to be passed to the detector constructor.
        """
        tracked_object = (
            tracked_object
            if tracked_object is not None
            else getattr(starter_event, "tracked_object", None)
        )
        if tracked_object and issubclass(
            detector_type, (DetectorWithTrackedObject, DetectorWithTwoTrackedObjects)
        ):
            other_detector_kwargs["tracked_object"] = tracked_object
        if starter_event and issubclass(detector_type, DetectorWithStarterEvent):
            other_detector_kwargs["starter_event"] = starter_event
        return other_detector_kwargs

    @property
    def time_between_frames(self) -> datetime.timedelta:
        """
        :return: The time between frames of the episode player.
        """
        return self.episode_player.time_between_frames

    def join(self):
        """
        Join all the threads.
        """
        # self.logger.debug_events()
        self.logger.join()
        logger.debug("All threads joined.")


class AgentEpisodeSegmenter(EpisodeSegmenter):
    """
    The AgentBasedEpisodeSegmenter class is used to segment motions into activities (e.g. PickUp) by tracking the
     events that are relevant to the agent for example contact events of the hands or robot.
    """

    def start_tracking_threads_for_new_object_and_event(
        self, new_object: Body, event: Optional[CloseContactEvent] = None
    ):
        logger.debug(
            f"Creating contact and motion threads for object {new_object.name}"
        )
        self.start_contact_threads_for_object(new_object, event)
        self.start_motion_threads_for_object(new_object, event)

    def _process_event(self, event: Event) -> None:
        if isinstance(event, CloseContactEvent):
            self.update_tracked_objects(event)

    def _run_initial_event_detectors(self) -> None:
        """
        Start the contact threads for the agents.
        """
        agents = self.get_agents()
        for agent in agents:
            self.object_trackers[agent] = ObjectTracker(agent)
            self.start_contact_threads_for_object(agent)

    @staticmethod
    def get_agents() -> List[Body]:
        """
        :return: A list of Object instances that represent the available agents in the world.
        """
        return [
            obj
            for obj in World.current_world.objects
            if issubclass(obj.obj_type, Agent)
        ]


class NoAgentEpisodeSegmenter(EpisodeSegmenter):
    """
    The NoAgentEpisodeSegmenter class is used to segment episodes into activities (e.g. PickUp) by tracking the
     events that are relevant to the objects in the world with the lack of an agent in the episode.
    """

    def start_tracking_threads_for_new_object_and_event(
        self, new_object: Body, event: EventUnion
    ):
        pass

    def get_involved_bodies(self, event: EventUnion) -> List[Body]:
        return []

    def _process_event(self, event: EventUnion) -> None:
        pass

    def _run_initial_event_detectors(self) -> None:
        """
        Start the motion detection threads for the objects in the world.
        """
        self.run_objects_initial_event_detectors()


@dataclass
class EpisodeSegmenterExecutor:
    context: SegmindContext
    player: EpisodePlayer | None = None
    pacer: Pacer = field(default_factory=SimulationPacer)
    statechart: DetectorStateChart = field(init=False)
    _control_cycle_index: int = field(init=False)
    _time_variable: FloatVariable = field(init=False)



    def compile(self, statechart: DetectorStateChart):
        self.statechart = statechart
        self.control_cycles = 0
        self.statechart.compile(self.context)
        # self.context.collision_manager.update_collision_matrix()
        # do one tick to immediately active nodes whose start condition is constant true.
        self.fill_holes()
        self.statechart.tick(self.context)
        if self.player:
            self.player.start()


    def fill_holes(self):
        for o in self.context.world.bodies:
            if "hole" in o.name.name:
                self.context.holes.append(o)

    def tick(self):
        #self.player.pause()
        #self.control_cycles += 1
        if hasattr(self.context, "tick_count"):
            self.context.tick_count += 1
        self.statechart.tick(self.context)
        #self.player.resume()
        # ToDo: Here we need to add the state model updates.

    def tick_until_end(self, timeout: int = 1_000):
        """
        Calls tick until is_end_motion() returns True.
        :param timeout: Max number of ticks to perform.
        #ToDo: So in the Dataplayer thread we can add an EndMotion Node and that will trigger the end.
        """
        try:
            for i in range(timeout):
                if self.player.is_alive():
                    self.tick()
                    self.pacer.sleep()
                else:
                    return
            raise TimeoutError("Timeout reached while waiting for end of motion.")
        finally:
            self.statechart.cleanup_nodes(context=self.context)
            self.context.cleanup()

    def spawn_scene(self, models_dir):
        directory = Path(models_dir)
        urdf_files = [f.name for f in directory.glob("*.urdf")]
        for file in urdf_files:
            file_path = models_dir + file
            obj_name = Path(file).stem

            if obj_name == "iCub":
                continue
            try:
                if obj_name == "scene":
                    obj_world = URDFParser.from_file(file_path).parse()
                    world_C_scene = FixedConnection(
                        parent=self.context.world.root, child=obj_world.root
                    )
                    with self.context.world.modify_world():
                        self.context.world.merge_world(obj_world, world_C_scene)
                else:
                    obj_world = URDFParser.from_file(file_path).parse()
                    with self.context.world.modify_world():
                        self.context.world.merge_world(obj_world)

            except Exception as e:
                # import pdb
                # pdb.set_trace()

                continue
