from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Set

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
from segmind.detectors.base import AbstractDetector, SegmindContext


@dataclass(eq=False, repr=False)
class SupportDetector(AbstractDetector):
    """
    Class for detecting and updating newly established support relationships.

    This class provides functionality to detect and update support relationships
    between physical bodies. It evaluates the given objects and generates events
    for newly established support connections. This can be useful in simulations
    or physics-based environments to monitor and handle dynamic interactions
    between objects.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        objects_to_check: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects newly established support relationships.

        A held object is carried rather than resting, so what it brushes on the way does
        not become something it rests on. What already holds it up is left alone, so
        taking hold of something that still stands on a surface does not take that
        surface away from it.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param objects_to_check: Bodies that should be evaluated for new supports.
        :return: List of SupportEvent objects representing newly detected supports.
        """

        events = []
        latest_support = segmind_context.latest_support
        carried_by_nothing = [
            body
            for body in objects_to_check
            if not segmind_context.latest_grasps.get(body)
        ]
        new_support_pairs = self.get_relation(
            context,
            carried_by_nothing,
            is_supported_by,
            candidates=self.bodies_outside_end_effectors(context.world),
        )
        for body, support in new_support_pairs.items():
            new_supports = (
                support
                if body not in latest_support
                else support - latest_support[body]
            )
            if new_supports:
                latest_support.setdefault(body, set()).update(new_supports)
                events.extend(
                    [
                        SupportEvent(tracked_object=body, with_object=s)
                        for s in new_supports
                    ]
                )

        return events


@dataclass(eq=False, repr=False)
class LossOfSupportDetector(AbstractDetector):
    """
    Detects and manages the loss of support relationships among objects.

    This class is a specialized support detector that identifies when previously
    registered support relationships are no longer present. It processes a given
    set of objects to detect and update the context with events signifying the loss
    of such support relationships. This functionality is particularly useful in
    simulation or analysis scenarios where maintaining updated context for object
    interactions is essential.
    """

    counterpart = SupportDetector

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        objects_to_check: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects when previously existing support relationships are lost.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param objects_to_check: Bodies that should be evaluated for lost supports.
        :return: List of LossOfSupportEvent objects representing removed supports.
        """

        new_support_pairs = self.get_relation(
            context,
            objects_to_check,
            is_supported_by,
            candidates=self.bodies_outside_end_effectors(context.world),
        )
        lost_supports = self.forget_lost_relations(
            segmind_context.latest_support, new_support_pairs, objects_to_check
        )
        return [
            LossOfSupportEvent(tracked_object=body, with_object=supporter)
            for body, supporters in lost_supports.items()
            for supporter in supporters
        ]


@dataclass(eq=False, repr=False)
class BaseContainmentDetector(AbstractDetector):
    """
    Abstract base class for contaiment-based detectors.

    Provides shared functionality for detecting containment between
    bodies and generating events when containment relationships change.
    """

    containment_threshold: float = 0.9
    """
    The threshold for the containment ratio between two bodies to be considered containment.
    """

    def get_containment_pairs(
        self, context: MotionStatechartContext, tracked_objects: List[Body]
    ) -> Dict[Body, Set[Body]]:
        """
        Computes support relationships.

        :param tracked_objects: Bodies that should be checked.
        :return: Mapping of body → supporting bodies.
        """
        containment_pairs: Dict[Body, Set[Body]] = {}
        candidates = self.bodies_outside_end_effectors(context.world)

        for obj in tracked_objects:
            containers = {
                body
                for body in candidates
                if obj is not body
                and InsideOf(obj, body).compute_containment_ratio()
                > self.containment_threshold
            }
            if containers:
                containment_pairs[obj] = containers

        return containment_pairs


@dataclass(eq=False, repr=False)
class ContainmentDetector(BaseContainmentDetector):
    """
    Handles detection of containment events between objects.

    This class performs the task of identifying and updating containment relations between
    given objects. It determines when a new containment relationship is established and
    generates corresponding containment events. The purpose of this class is to provide
    event-driven responses based on the spatial interactions of objects.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        objects_to_check: List[Body],
    ) -> List[DetectionEvent]:
        """
        Updates the tracking context with new containment relationships and generates
        containment events for identified changes. The function processes a list of
        objects, compares the current containment status against the latest tracked
        data, and generates events for any newly identified containment relationships.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param objects_to_check: List of Body objects to check for containment changes.
        :return: List of ContainmentEvent objects representing newly established containments.
        """
        new_containment_pairs = self.get_containment_pairs(context, objects_to_check)
        latest_containment = segmind_context.latest_containments
        events = []

        for obj, containment_list in new_containment_pairs.items():
            new_containments = containment_list - latest_containment.get(obj, set())

            if not new_containments:
                continue

            latest_containment.setdefault(obj, set()).update(new_containments)
            events.extend(
                ContainmentEvent(tracked_object=obj, with_object=c)
                for c in new_containments
            )

        return events


@dataclass(eq=False, repr=False)
class LossOfContainmentDetector(BaseContainmentDetector):
    """
    Detects and processes loss of containment events.

    The LossOfContainmentDetector class is responsible for identifying instances where an
    object loses containment with another object. It updates the current containment context
    and generates a list of events representing these loss of containment occurrences. This
    class extends BaseContainmentDetector and utilizes its utilities for containment
    verification and context management.

    """

    counterpart = ContainmentDetector

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        objects_to_check: List[Body],
    ) -> List[DetectionEvent]:
        """
        Updates the context with the latest containment pairs and generates events for
        any lost containments.

        This method checks the current state of containment pairs against the previously
        stored state in the context. If any containments have been lost, it removes
        them from the context and generates corresponding events.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param objects_to_check: List of Body objects to check for containment loss.
        :return: List of LossOfContainmentEvent objects representing the loss of containment.
        """
        new_containment_pairs = self.get_containment_pairs(context, objects_to_check)
        lost_containments = self.forget_lost_relations(
            segmind_context.latest_containments, new_containment_pairs, objects_to_check
        )
        return [
            LossOfContainmentEvent(tracked_object=obj, with_object=container)
            for obj, containers in lost_containments.items()
            for container in containers
        ]


@dataclass(eq=False, repr=False)
class InsertionDetector(AbstractDetector):
    """
    Detects insertion events based on object interaction context.

    The InsertionDetector class is used to analyze the interaction between tracked
    objects and identify insertion events. It tracks specific events such as
    contacts and containment, and generates an InsertionEvent when specific
    conditions are met. The class leverages a context that holds relevant
    event logs and tracked objects.
    """

    requires = (ContactDetector, ContainmentDetector)

    shift_threshold: timedelta = timedelta(seconds=15.0)
    """
    The threshold for the time difference between two events to be considered an insertion.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objs: List[Body],
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
        :param tracked_objs: List of Body objects to analyze for insertion events.
        :return List of InsertionEvent objects representing detected insertions.
        """
        events = []
        contact_events = [
            i
            for i in segmind_context.logger.get_events()
            if isinstance(i, ContactEvent)
        ]
        contact_events_with_holes = [
            i for i in contact_events if i.with_object in segmind_context.holes
        ]
        containment_event = [
            i
            for i in segmind_context.logger.get_events()
            if isinstance(i, ContainmentEvent)
        ]

        by_object = defaultdict(list)
        for i in contact_events_with_holes:
            by_object[i.tracked_object].append(i)

        for j in containment_event:
            for i in by_object.get(j.tracked_object, []):
                if abs(i.timestamp - j.timestamp) >= self.shift_threshold:
                    continue

                key = (i.tracked_object.id, i.with_object.id)
                if key in segmind_context.insertion_pairs:
                    continue

                segmind_context.insertion_pairs.add(key)

                events.append(
                    InsertionEvent(
                        tracked_object=i.tracked_object,
                        with_object=i.with_object,
                        inserted_into_objects=[j.with_object],
                    )
                )
                break

        return events
