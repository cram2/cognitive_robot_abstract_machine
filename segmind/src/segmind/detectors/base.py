from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Any, Sequence, Tuple, Type

from giskardpy.motion_statechart.context import (
    MotionStatechartContext,
    ContextExtension,
)
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.datastructures.events import MotionEvent, DetectionEvent, RotationEvent
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.event_logger import EventLogger
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class DetectorStateChart(MotionStatechart):
    """
    Statechart responsible for running the different motion detectors.

    Currently acts as a container for the detectors and inherits the
    functionality from MotionStatechart.
    """


IndexedBodyPairs = Dict[Body, Set[Body]]
"""
Type hint for dictionaries mapping bodies to sets of bodies
"""


@dataclass
class SegmindContext(ContextExtension):
    """
    Context object shared across the motion statechart detectors.

    Stores the latest detected contact and support relationships
    between bodies in the simulation as well as the event logger.
    """

    latest_contact_bodies: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies it is currently in contact with.
    """

    latest_support: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently support it.
    """

    latest_grasps: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the tool frames that currently have hold of it.
    """

    latest_containments: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently contain it.
    """

    latest_motion_events: Dict[Body, MotionEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active motion event, if any.
    """

    latest_rotation_events: Dict[Body, RotationEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active rotation event, if any.
    """

    logger: EventLogger = field(default_factory=EventLogger)
    """
    The event logger used to record detected events.
    """

    spent_interaction_events: set[Any] = field(default_factory=set)
    """
    The events already taken as evidence of an interaction, per detector, so that none
    of them is counted twice: a hand that loses its grip and takes hold again has not
    picked the object up a second time.
    """

    placing_pairs: set[Any] = field(default_factory=set)
    """
    Set of placing pairs, to avoid duplicate events
    """

    holes: List[Aperture] = field(default_factory=list)
    """
    List of bodies that can be considered holes
    """

    insertion_pairs: set[Any] = field(default_factory=set)
    """
    List of insertion pairs, to avoid duplicate events
    """

    tracker_registry: ObjectTrackerFactory = field(default_factory=ObjectTrackerFactory)
    """
    The object tracker registry.    
    """


@dataclass(repr=False, eq=False)
class AbstractDetector(MotionStatechartNode, ABC):
    """
    Abstract base class for all detectors.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    exclude_robot: bool = field(kw_only=True, default=True)
    """
    Whether every body of every robot is left out of what a tracked object is checked
    against.

    A run reads what happens in the scene, and a robot carrying an object touches it
    throughout; what the robot does with it is read from the grasp instead, which asks
    about the hand directly and so is unaffected by this.
    """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Executes one update cycle of the detector.

        Determines the objects that should be checked for contacts,
        computes new contact relationships, and triggers events if
        contact changes are detected.

        :param context: The current motion statechart context.
        :return: ObservationStateValues.TRUE if events were triggered,
        otherwise ObservationStateValues.FALSE.
        """
        segmind_context_extension = context.require_extension(SegmindContext)

        objects_to_check = (
            [self.tracked_object]
            if self.tracked_object
            else [
                body
                for body in context.world.bodies
                if type(body.parent_connection) is Connection6DoF
            ]
        )
        events = self.update_context_and_events(
            context, segmind_context_extension, objects_to_check
        )
        for e in events:
            segmind_context_extension.logger.log_event(
                e, segmind_context_extension.tracker_registry
            )
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    @classmethod
    def get_required_detector_types(cls) -> Tuple[Type[AbstractDetector], ...]:
        """
        :return: The kinds of detector whose events this one is read from. A run using
            this detector uses them too.
        """
        return ()

    @classmethod
    def create_for_run(
        cls,
        watched_bodies: Sequence[Body],
        detector_types: Sequence[Type[AbstractDetector]],
    ) -> List[AbstractDetector]:
        """
        The detectors of this kind a run ticks: one for each body it watches.

        :param watched_bodies: The bodies the run watches.
        :param detector_types: Every kind of detector the run ticks.
        """
        return [cls(tracked_object=body) for body in watched_bodies]

    @staticmethod
    def remember_new_relations(
        remembered: IndexedBodyPairs, holding: IndexedBodyPairs
    ) -> IndexedBodyPairs:
        """
        Remember the relations that hold now and were not remembered before.

        :param remembered: The relations detected so far, per body; the new ones are
            added to it.
        :param holding: The relations that hold now, per body.
        :return: The relations that are new, per body.
        """
        new: IndexedBodyPairs = {}
        for body, relations in holding.items():
            added = relations - remembered.get(body, set())
            if not added:
                continue
            remembered.setdefault(body, set()).update(added)
            new[body] = added
        return new

    @staticmethod
    def forget_lost_relations(
        remembered: IndexedBodyPairs, holding: IndexedBodyPairs, bodies: List[Body]
    ) -> IndexedBodyPairs:
        """
        Forget the relations of ``bodies`` that no longer hold.

        Only the relations of ``bodies`` are judged, so a detector never declares lost a
        relation of a body it did not check.

        :param remembered: The relations detected so far, per body; the lost ones are
            removed from it.
        :param holding: The relations that hold now, per body.
        :param bodies: The bodies whose relations were checked.
        :return: The relations that were lost, per body.
        """
        lost: IndexedBodyPairs = {}
        for body in bodies:
            relations = remembered.get(body)
            if not relations:
                continue
            gone = relations - holding.get(body, set())
            if not gone:
                continue
            relations -= gone
            if not relations:
                remembered.pop(body)
            lost[body] = gone
        return lost

    @staticmethod
    def bodies_outside_end_effectors(world: World) -> List[Body]:
        """
        The collidable bodies of ``world`` that are part of no end effector.

        An end effector holds what it grasps; it is not what objects rest on or are
        contained in.
        """
        end_effector_bodies = {
            body
            for end_effector in world.get_semantic_annotations_by_type(EndEffector)
            for body in end_effector.bodies
        }
        return [
            body
            for body in world.bodies_with_collision
            if body not in end_effector_bodies
        ]

    def get_relation(
        self,
        context: MotionStatechartContext,
        tracked_objects: List[Body],
        predicate,
        candidates: Optional[List[Body]] = None,
    ) -> Dict[Body, Set[Body]]:
        """
        Get the relation between tracked objects.

        :param context: The context containing world information.
        :param tracked_objects: List of bodies to check for contact changes.
        :param predicate: Function that returns true if the objects are related.
        :param candidates: The bodies a tracked object may be related to; every
            collidable body of the world when not given. The robot is left out of them
            unless :attr:`exclude_robot` says otherwise.
        :return: Dictionary mapping bodies to sets of related bodies.
        """

        related_bodies: Dict[Body, Set[Body]] = {}
        if candidates is None:
            candidates = context.world.bodies_with_collision
        if self.exclude_robot:
            robot_bodies = set(context.world.robot_bodies_with_collision)
            candidates = [body for body in candidates if body not in robot_bodies]
        for tracked_object in tracked_objects:
            for body in candidates:
                if body is tracked_object:
                    continue
                if predicate(tracked_object, body):
                    related_bodies.setdefault(tracked_object, set()).add(body)
        return related_bodies

    @abstractmethod
    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Core detection logic that updates the internal state and identifies new events.

        This method is called during every tick of the detector. Implementations should
        examine the current state of the world (via the context) for the given
        `tracked_objects`, update the relevant fields in `context` (e.g.,
        `latest_contact_bodies`, `latest_support`), and return a list of any
        `DetectionEvent`s that occurred since the last update.

        Specific implementations may detect:
        * State changes: e.g., a new contact (ContactEvent) or loss of contact.
        * Continuous processes: e.g., ongoing motion or containment.
        * Complex interactions: e.g., insertion or picking up objects.

        :param context: The shared SegmindContext containing the world state,
                        history of relationships, and the event logger.
        :param segmind_context: The SegmindContext extension containing additional states.
        :param tracked_objects: A list of bodies that this detector should focus on
                                during this update cycle.
        :return: A list of DetectionEvent objects representing the events detected
                 in this cycle. Returns an empty list if no events were found.
        """
        pass


@dataclass(repr=False, eq=False)
class EventCombiningDetector(AbstractDetector, ABC):
    """
    A detector concluding from the events other detectors detected, over every body
    those watch.
    """

    @classmethod
    def create_for_run(
        cls,
        watched_bodies: Sequence[Body],
        detector_types: Sequence[Type[AbstractDetector]],
    ) -> List[AbstractDetector]:
        """
        The detectors of this kind a run ticks: one, for all the bodies it watches.
        """
        return [cls()]
