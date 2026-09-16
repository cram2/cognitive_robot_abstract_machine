from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Callable, Any, Type

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
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.grasp_detector_nodes import (
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

    def _find_interaction_events(
        self,
        segmind_context: SegmindContext,
        primary_event_type: type,
        secondary_event_type: type,
        make_event: Callable[[Any, Any], DetectionEvent],
    ) -> List[DetectionEvent]:
        """
        Scans logged events for correlated pairs of primary and secondary event types
        and emits a detection event for each new, unseen pairing.

        For each secondary event, this method searches for a primary event on the same
        tracked object whose timestamp is within :attr:`shift_threshold`. A secondary
        event is evidence of one interaction only, so a later primary cannot conclude a
        second interaction from the same one. If such a pair
        is found and has not been recorded in ``segmind_context.placing_pairs`` before,
        the pair is registered and a detection event is produced via ``make_event``.

        :param segmind_context: The shared context holding the event logger and
            previously seen interaction pairs.
        :param primary_event_type: The event type to use as the primary signal
            (e.g. ``StopTranslationEvent`` for placing, ``TranslationEvent`` for pickup).
        :param secondary_event_type: The event type to correlate against the primary
            (e.g. ``SupportEvent`` for placing, ``LossOfSupportEvent`` for pickup).
        :param make_event: Factory called with ``(primary, secondary)`` to produce the
            outgoing :class:`DetectionEvent`. Only called once per unique pair.
        :return: List of newly detected interaction events.
        """
        primary_events = [
            e
            for e in segmind_context.logger.get_events()
            if isinstance(e, primary_event_type)
        ]
        secondary_events = [
            e
            for e in segmind_context.logger.get_events()
            if isinstance(e, secondary_event_type)
        ]

        events = []
        by_object = defaultdict(list)
        for e in primary_events:
            by_object[e.tracked_object].append(e)

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
                events.append(make_event(primary, secondary))
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

    def interaction_key(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> Hashable:
        """
        A placing is the object coming to rest on a surface, concluded once per surface.
        """
        return PlacingEvent, secondary.tracked_object, secondary.with_object

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        obj: List[Body],
    ) -> List[DetectionEvent]:
        """
        Updates the system context with new placing event instances based on past
        actions logged in the system. It analyzes and filters specific event types
        to detect new events that can be generated. This function ensures distinct
        events are created by maintaining exclusivity through a pairing mechanism.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param obj: List of bodies to analyze for potential placing events.
        :return: List of generated placing events based on observed interactions.
        """
        return self._find_interaction_events(
            segmind_context,
            primary_event_type=(
                LossOfGraspEvent
                if self.runs_beside(LossOfGraspDetector)
                else StopTranslationEvent
            ),
            secondary_event_type=SupportEvent,
            make_event=lambda primary, secondary: PlacingEvent(
                tracked_object=primary.tracked_object,
                with_object=secondary.with_object,
            ),
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

    def interaction_key(
        self, primary: EventWithTrackedObjects, secondary: EventWithTrackedObjects
    ) -> Hashable:
        """
        A pick-up is one lift of the object, however many supports it loses at once.
        """
        return PickUpEvent, primary

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        obj: List[Body],
    ) -> List[DetectionEvent]:
        """
        Updates the context and generates a list of events based on translation events and loss
        of support events. The method identifies pairs of related events that are close in
        timestamp, determines if they are exclusive, and creates a new event when conditions
        are met.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param obj: List of bodies to analyze for potential pickup events.
        :return: List of generated pickup events based on observed interactions.
        """
        return self._find_interaction_events(
            segmind_context,
            primary_event_type=(
                GraspEvent if self.runs_beside(GraspDetector) else TranslationEvent
            ),
            secondary_event_type=LossOfSupportEvent,
            make_event=lambda primary, secondary: PickUpEvent(
                tracked_object=primary.tracked_object,
            ),
        )
