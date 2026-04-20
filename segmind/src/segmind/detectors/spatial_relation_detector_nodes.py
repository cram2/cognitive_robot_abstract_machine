from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set


from segmind.datastructures.events import (
    DetectionEvent,
    SupportEvent,
    LossOfSupportEvent,
    ContainmentEvent,
    LossOfContainmentEvent, ContactEvent, InsertionEvent,
)

from semantic_digital_twin.reasoning.predicates import is_supported_by, InsideOf
from semantic_digital_twin.world_description.world_entity import Body

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
        self, context:SegmindContext, objects_to_check: List[Body]
    ) -> List[DetectionEvent]:
        """
        Detects newly established support relationships.

        :param objects_to_check: Bodies that should be evaluated for new supports.
        :return: List of SupportEvent objects representing newly detected supports.
        """

        events = []
        latest_support = context.latest_support
        new_support_pairs = self.get_relation(context, objects_to_check, is_supported_by)
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

    def update_context_and_events(
        self,context:SegmindContext, objects_to_check: List[Body]
    ) -> List[DetectionEvent]:
        """
        Detects when previously existing support relationships are lost.

        :param objects_to_check: Bodies that should be evaluated for lost supports.
        :return: List of LossOfSupportEvent objects representing removed supports.
        """

        events = []
        latest_support = context.latest_support
        new_support_pairs = self.get_relation(context, objects_to_check, is_supported_by)
        for body, support in list(latest_support.items()):
            loss_supports = (
                support
                if body not in new_support_pairs
                else support - new_support_pairs[body]
            )
            if loss_supports:
                latest_support.pop(body)
                events.extend(
                    [
                        LossOfSupportEvent(tracked_object=body, with_object=s)
                        for s in loss_supports
                    ]
                )

        return events


# Attention: Class will be refactored!!
@dataclass(eq=False, repr=False)
class BaseContainmentDetector(AbstractDetector):
    """
    Abstract base class for contaiment-based detectors.

    Provides shared functionality for detecting containment between
    bodies and generating events when containment relationships change.
    """

    containment_threshold: float = 0.9

    def get_containment_pairs(
        self,context:SegmindContext, tracked_objects: List[Body]
    ) -> Dict[Body, Set[Body]]:
        """
        Computes support relationships.

        :param tracked_objects: Bodies that should be checked.
        :return: Mapping of body → supporting bodies.
        """
        containment_pairs: Dict[Body, Set[Body]] = {}
        bodies_with_collision = context.world.bodies_with_collision
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if obj is body:
                    continue
                if InsideOf(obj, body).compute_containment_ratio() > self.containment_threshold:
                    containment_pairs.setdefault(obj, set()).add(body)
        return containment_pairs


    def update_context_and_events(
        self, context: SegmindContext, objects_to_check: List[Body]
    ) -> List[DetectionEvent]:
        """
        Updates the cached support relationships and generates events.

        :param objects_to_check: Bodies that should be evaluated for support changes.
        :return: List of generated support-related events.
        """

        pass


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
        self, context:SegmindContext, objects_to_check: List[Body]
    ) -> List[DetectionEvent]:
        """
        Updates the tracking context with new containment relationships and generates
        containment events for identified changes. The function processes a list of
        objects, compares the current containment status against the latest tracked
        data, and generates events for any newly identified containment relationships.

        :param objects_to_check: List of Body objects to check for containment changes.
        :return: List of ContainmentEvent objects representing newly established containments.
        """
        new_containment_pairs = self.get_containment_pairs(context, objects_to_check)
        latest_containment = context.latest_containments
        events = []
        for obj, containment_list in new_containment_pairs.items():
            new_containments = (
                containment_list
                if obj not in latest_containment
                else containment_list - latest_containment[obj]
            )
            if new_containments:
                latest_containment.setdefault(obj, set()).update(new_containments)
                events.extend(
                    [
                        ContainmentEvent(tracked_object=obj, with_object=c)
                        for c in new_containments
                    ]
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
    def update_context_and_events(
        self, context, objects_to_check: List[Body]
    ) -> List[DetectionEvent]:
        """
        Updates the context with the latest containment pairs and generates events for
        any lost containments.

        This method checks the current state of containment pairs against the previously
        stored state in the context. If any containments have been lost, it removes
        them from the context and generates corresponding events.

        :param objects_to_check: List of Body objects to check for containment loss.
        :return: List of LossOfContainmentEvent objects representing the loss of containment.
        """
        new_containment_pairs = self.get_containment_pairs(context, objects_to_check)
        latest_containment = context.latest_containments
        events = []
        for obj, containment_list in list(latest_containment.items()):
            lost_containments = (
                containment_list
                if obj not in new_containment_pairs
                else containment_list - new_containment_pairs[obj]
            )
            if lost_containments:
                latest_containment.pop(obj)
                events.extend(
                    [
                        LossOfContainmentEvent(tracked_object=obj, with_object=c)
                        for c in lost_containments
                    ]
                )

        return events



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

    shift_threshold: float = 15.0
    """
    The threshold for the time difference between two events to be considered an insertion.
    """

    def update_context_and_events(self, context, tracked_objs: List[Body]) -> List[DetectionEvent]:
        """
        Updates context and processes tracked objects to generate a list of events.

        This method analyzes contact and containment events within the tracked objects,
        compares their timestamps with a threshold, and generates insertion events if
        specific conditions are met. It modifies the context state to track insertion
        pairs that have already been processed and ensures exclusivity during event
        generation.

        :param tracked_objs: List of Body objects to analyze for insertion events.
        :return List of InsertionEvent objects representing detected insertions.
        """
        events = []
        contact_events = [i for i in context.logger.get_events() if isinstance(i, ContactEvent)]
        contact_events_with_holes = [i for i in contact_events if i.with_object in context.holes]
        containment_event = [i for i in context.logger.get_events() if isinstance(i, ContainmentEvent)]

        by_object = defaultdict(list)
        for i in contact_events_with_holes:
            by_object[i.tracked_object].append(i)

        for j in containment_event:
            for i in by_object.get(j.tracked_object, []):
                if abs(i.timestamp - j.timestamp) >= self.shift_threshold:
                    continue

                key = (i.tracked_object.id, i.with_object.id)
                if key in context.insertion_pairs:
                    continue

                context.insertion_pairs.add(key)

                events.append(
                    InsertionEvent(
                        i.tracked_object,
                        [j.with_object],
                        i.with_object,
                        timestamp=i.timestamp,
                    )
                )
                break

        return events
