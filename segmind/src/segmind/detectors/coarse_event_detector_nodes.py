from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Type

from typing_extensions import Hashable
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    GraspEvent,
    LossOfGraspEvent,
    SupportEvent,
    DetectionEvent,
    PlacingEvent,
    TranslationEvent,
    LossOfSupportEvent,
    PickUpEvent,
    StopTranslationEvent,
    ContactEvent,
    ContainmentEvent,
    InsertionEvent,
    EventWithTrackedObjects,
)
from semantic_digital_twin.world_description.world_entity import Body
from segmind.detectors.atomic_event_detectors_nodes import TranslationDetector
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.detectors.agent_event_detector_nodes import (
    GraspDetector,
    LossOfGraspDetector,
)


@dataclass
class AbstractInteractionDetector(AbstractDetector):
    """
    Abstract base class for interaction-based detectors.

    Provides shared functionality for monitoring interactions of
    bodies and generating events when detected.
    """

    shift_threshold: timedelta = timedelta(seconds=15)
    """
    The threshold for the time difference between two events to be considered an interaction.
    """

    def runs_beside(self, detector_type: Type[AbstractDetector]) -> bool:
        """
        Whether a detector of ``detector_type`` runs in the same statechart as this one.

        What a run watches for is what it can conclude from: where nothing detects an
        agent taking hold of things, an interaction has to be read from the object's
        own motion instead.

        :param detector_type: The kind of detector looked for.
        :return: True when one of that kind runs beside this detector.
        """
        statechart = self._motion_statechart
        return statechart is not None and any(
            isinstance(node, detector_type) for node in statechart.nodes
        )

    @abstractmethod
    def interaction_key(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> Hashable:
        """
        What an interaction concluded from ``primary`` and ``secondary`` shares with any
        other conclusion of the same interaction, so that it is concluded only once.
        """

    @property
    @abstractmethod
    def primary_event_type(self) -> Type[EventWithTrackedObjects]:
        """
        The kind of event the interaction is concluded from, which happens to the
        tracked object.
        """

    @property
    @abstractmethod
    def secondary_event_type(self) -> Type[EventWithTrackedObjects]:
        """
        The kind of event that has to happen close in time to the primary one for the
        interaction to be concluded.
        """

    @abstractmethod
    def make_event(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> DetectionEvent:
        """
        The interaction concluded from ``primary`` and ``secondary``.
        """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Concludes the interactions that the events logged so far amount to.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information
            required to track events.
        :param tracked_objects: The bodies checked this tick.
        :return: The interactions not concluded before.
        """
        return self._find_interaction_events(segmind_context)

    def _find_interaction_events(
        self, segmind_context: SegmindContext
    ) -> List[DetectionEvent]:
        """
        Scans logged events for correlated pairs of primary and secondary events and
        emits a detection event for each new, unseen pairing.

        For each secondary event, this method searches for a primary event on the same
        tracked object whose timestamp is within :attr:`shift_threshold`. A secondary
        event is evidence of one interaction only, so a later primary cannot conclude a
        second interaction from the same one. If such a pair
        is found and has not been recorded in ``segmind_context.placing_pairs`` before,
        the pair is registered and a detection event is produced via :meth:`make_event`.

        :param segmind_context: The shared context holding the event logger and
            previously seen interaction pairs.
        :return: List of newly detected interaction events.
        """
        primary_events = [
            event
            for event in segmind_context.logger.get_events()
            if isinstance(event, self.primary_event_type)
        ]
        secondary_events = [
            event
            for event in segmind_context.logger.get_events()
            if isinstance(event, self.secondary_event_type)
        ]

        events = []
        by_object = defaultdict(list)
        for primary_event in primary_events:
            by_object[primary_event.tracked_object].append(primary_event)

        spent = segmind_context.spent_interaction_events
        for secondary in secondary_events:
            if (type(self), secondary) in spent:
                continue
            for primary in by_object.get(secondary.tracked_object, []):
                if abs(secondary.timestamp - primary.timestamp) >= self.shift_threshold:
                    continue

                key = self.interaction_key(primary, secondary)
                if key in segmind_context.placing_pairs:
                    continue

                segmind_context.placing_pairs.add(key)
                spent.add((type(self), secondary))
                events.append(self.make_event(primary, secondary))
                break

        return events


@dataclass
class PlacingDetector(AbstractInteractionDetector):
    """
    Reports an object being put down.

    Where an agent letting go of things is watched for, a placing is where it let this
    one go; otherwise it is the object coming to rest on a surface, which is all a run
    with no agent in it has to go on.

    This class is typically used to analyze specific event types, such as stop
    motion and support events, and identify correlations that form the basis
    of new placing events. By ensuring that placing events are uniquely paired,
    the class helps maintain consistency and prevent duplication of events.
    """

    @classmethod
    def get_required_detector_types(cls) -> Tuple[Type[AbstractDetector], ...]:
        return (SupportDetector, TranslationDetector)

    def interaction_key(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> Hashable:
        """
        A placing is the object coming to rest on a surface, concluded once per surface.
        """
        return PlacingEvent, secondary.tracked_object, secondary.with_object

    @property
    def primary_event_type(self) -> Type[EventWithTrackedObjects]:
        """
        The agent letting go of the object where an agent is watched for, otherwise the
        object coming to a stop.
        """
        return (
            LossOfGraspEvent
            if self.runs_beside(LossOfGraspDetector)
            else StopTranslationEvent
        )

    @property
    def secondary_event_type(self) -> Type[EventWithTrackedObjects]:
        return SupportEvent

    def make_event(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> DetectionEvent:
        return PlacingEvent(
            tracked_object=primary.tracked_object, with_object=secondary.with_object
        )


@dataclass
class PickUpDetector(AbstractInteractionDetector):
    """
    Reports an object being picked up.

    Where an agent taking hold of things is watched for, a pick-up is that agent lifting
    this one off what held it up, one per grasp; otherwise it is the object moving after
    losing its support, which is all a run with no agent in it has to go on.

    The PickUpDetector class determines if a "pickup" event has occurred by analyzing
    contextual events such as TranslationEvent and LossOfSupportEvent. It ensures
    that such events are detected and processed by checking their timestamps and
    associating them with corresponding objects. The resulting detected events are
    then returned. This class interfaces with a logger to gather the needed event
    data and uses a context to manage event pairs and thresholds.
    """

    @classmethod
    def get_required_detector_types(cls) -> Tuple[Type[AbstractDetector], ...]:
        return (SupportDetector, TranslationDetector)

    def interaction_key(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> Hashable:
        """
        A pick-up is one lift of the object, however many supports it loses at once.
        """
        return PickUpEvent, primary

    @property
    def primary_event_type(self) -> Type[EventWithTrackedObjects]:
        """
        The agent taking hold of the object where an agent is watched for, otherwise the
        object moving.
        """
        return GraspEvent if self.runs_beside(GraspDetector) else TranslationEvent

    @property
    def secondary_event_type(self) -> Type[EventWithTrackedObjects]:
        return LossOfSupportEvent

    def make_event(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> DetectionEvent:
        return PickUpEvent(tracked_object=primary.tracked_object)
