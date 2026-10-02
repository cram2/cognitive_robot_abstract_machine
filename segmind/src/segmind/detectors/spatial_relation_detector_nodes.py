from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Type

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    DetectionEvent,
    SupportEvent,
    LossOfSupportEvent,
    ContainmentEvent,
    LossOfContainmentEvent,
    ContactEvent,
    InsertionEvent,
)

from semantic_digital_twin.reasoning.predicates import is_supported_by, InsideOf
from semantic_digital_twin.world_description.world_entity import Body

from segmind.detectors.atomic_event_detectors_nodes import ContactDetector
from segmind.detectors.base import (
    AbstractDetector,
    EventCombiningDetector,
    SegmindContext,
)


@dataclass(eq=False, repr=False)
class SupportDetector(AbstractDetector):
    """
    Detects supports being established and being lost.

    A support is one body resting on another. The detector reports a
    :class:`SupportEvent` when a body starts resting on something and a
    :class:`LossOfSupportEvent` when it stops resting on something it rested on.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        objects_to_check: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects newly established and newly lost support relationships.

        A held object is carried rather than resting, so what it brushes on the way does
        not become something it rests on. What already holds it up is left alone, so
        taking hold of something that still stands on a surface does not take that
        surface away from it.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param objects_to_check: Bodies that should be evaluated for supports.
        :return: The supports established, then the supports lost.
        """
        supports_now = self.get_relation(
            context,
            objects_to_check,
            is_supported_by,
            candidates=self.bodies_outside_end_effectors(context.world),
        )
        latest_supports = segmind_context.latest_support
        not_held = {
            body: supporters
            for body, supporters in supports_now.items()
            if not segmind_context.latest_grasps.get(body)
        }
        new_supports = self.remember_new_relations(latest_supports, not_held)
        lost_supports = self.forget_lost_relations(
            latest_supports, supports_now, objects_to_check
        )
        return [
            SupportEvent(tracked_object=body, with_object=supporter)
            for body, supporters in new_supports.items()
            for supporter in supporters
        ] + [
            LossOfSupportEvent(tracked_object=body, with_object=supporter)
            for body, supporters in lost_supports.items()
            for supporter in supporters
        ]


@dataclass(eq=False, repr=False)
class ContainmentDetector(AbstractDetector):
    """
    Detects containments between bodies being established and being lost.

    The detector reports a :class:`ContainmentEvent` when a body ends up inside
    something and a :class:`LossOfContainmentEvent` when it leaves something it was
    inside.
    """

    containment_threshold: float = 0.9
    """
    The threshold for the containment ratio between two bodies to be considered containment.
    """

    def get_containment_pairs(
        self, context: MotionStatechartContext, tracked_objects: List[Body]
    ) -> Dict[Body, Set[Body]]:
        """
        Computes containment relationships.

        :param tracked_objects: Bodies that should be checked.
        :return: Mapping of body → containing bodies.
        """
        containment_pairs: Dict[Body, Set[Body]] = {}
        candidates = self.bodies_outside_end_effectors(context.world)

        for tracked_object in tracked_objects:
            containers = {
                body
                for body in candidates
                if tracked_object is not body
                and InsideOf(tracked_object, body).compute_containment_ratio()
                > self.containment_threshold
            }
            if containers:
                containment_pairs[tracked_object] = containers

        return containment_pairs

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        objects_to_check: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects newly established and newly lost containments and updates the stored
        containment state.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param objects_to_check: List of Body objects to check for containment changes.
        :return: The containments established, then the containments lost.
        """
        containments_now = self.get_containment_pairs(context, objects_to_check)
        latest_containments = segmind_context.latest_containments
        new_containments = self.remember_new_relations(
            latest_containments, containments_now
        )
        lost_containments = self.forget_lost_relations(
            latest_containments, containments_now, objects_to_check
        )
        return [
            ContainmentEvent(tracked_object=body, with_object=container)
            for body, containers in new_containments.items()
            for container in containers
        ] + [
            LossOfContainmentEvent(tracked_object=body, with_object=container)
            for body, containers in lost_containments.items()
            for container in containers
        ]


@dataclass(eq=False, repr=False)
class InsertionDetector(EventCombiningDetector):
    """
    Detects insertion events based on object interaction context.

    The InsertionDetector class is used to analyze the interaction between tracked
    objects and identify insertion events. It tracks specific events such as
    contacts and containment, and generates an InsertionEvent when specific
    conditions are met. The class leverages a context that holds relevant
    event logs and tracked objects.
    """

    @classmethod
    def get_required_detector_types(cls) -> Tuple[Type[AbstractDetector], ...]:
        return (ContactDetector, ContainmentDetector)

    shift_threshold: timedelta = timedelta(seconds=15.0)
    """
    The threshold for the time difference between two events to be considered an insertion.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Updates context and processes tracked objects to generate a list of events.

        This method analyzes contact and containment events within the tracked objects,
        compares their timestamps with a threshold, and generates insertion events if
        specific conditions are met. It modifies the context state to track insertion
        pairs that have already been processed and ensures exclusivity during event
        generation.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: List of Body objects to analyze for insertion events.
        :return List of InsertionEvent objects representing detected insertions.
        """
        events = []
        contact_events = [
            event
            for event in segmind_context.logger.get_events()
            if isinstance(event, ContactEvent)
        ]
        contact_events_with_holes = [
            contact_event
            for contact_event in contact_events
            if contact_event.with_object in segmind_context.holes
        ]
        containment_events = [
            event
            for event in segmind_context.logger.get_events()
            if isinstance(event, ContainmentEvent)
        ]

        contacts_by_object = defaultdict(list)
        for contact_event in contact_events_with_holes:
            contacts_by_object[contact_event.tracked_object].append(contact_event)

        for containment_event in containment_events:
            for contact_event in contacts_by_object.get(
                containment_event.tracked_object, []
            ):
                if (
                    abs(contact_event.timestamp - containment_event.timestamp)
                    >= self.shift_threshold
                ):
                    continue

                key = (contact_event.tracked_object.id, contact_event.with_object.id)
                if key in segmind_context.insertion_pairs:
                    continue

                segmind_context.insertion_pairs.add(key)

                events.append(
                    InsertionEvent(
                        tracked_object=contact_event.tracked_object,
                        with_object=contact_event.with_object,
                        inserted_into_objects=[containment_event.with_object],
                    )
                )
                break

        return events
