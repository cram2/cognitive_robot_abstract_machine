from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import List
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    SupportEvent,
    DetectionEvent,
    PlacingEvent, TranslationEvent, LossOfSupportEvent, PickUpEvent, StopTranslationEvent,
)
from semantic_digital_twin.world_description.world_entity import Body
from segmind.detectors.base import AbstractDetector, SegmindContext


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


@dataclass
class PlacingDetector(AbstractInteractionDetector):
    """
    Represents a class detection mechanism for identifying and managing new
    placing events from observed system interactions.

    This class is typically used to analyze specific event types, such as stop
    motion and support events, and identify correlations that form the basis
    of new placing events. By ensuring that placing events are uniquely paired,
    the class helps maintain consistency and prevent duplication of events.
    """

    def update_context_and_events(self, context:MotionStatechartContext, segmind_context:SegmindContext, obj: List[Body]) -> List[DetectionEvent]:
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
        stop_translation_events = [
            event
            for event in segmind_context.logger.get_events()
            if isinstance(event, StopTranslationEvent)
        ]
        support_events = [
            event for event in segmind_context.logger.get_events() if isinstance(event, SupportEvent)
        ]

        events = []

        by_object = defaultdict(list)
        for event in stop_translation_events:
            by_object[event.tracked_object].append(event)

        for support_event in support_events:
            for event in by_object.get(support_event.tracked_object, []):
                if abs(support_event.timestamp - event.timestamp) >= self.shift_threshold:
                    continue

                key = (support_event.tracked_object.id, support_event.with_object.id)
                if key in segmind_context.placing_pairs:
                    continue

                segmind_context.placing_pairs.add(key)
                events.append(
                    PlacingEvent(
                        tracked_object=event.tracked_object,
                        with_object=support_event.with_object,
                    )
                )
                break

        return events


@dataclass
class PickUpDetector(AbstractInteractionDetector):
    """
    Detects and processes interactions suggesting an object has been picked up.

    The PickUpDetector class determines if a "pickup" event has occurred by analyzing
    contextual events such as TranslationEvent and LossOfSupportEvent. It ensures
    that such events are detected and processed by checking their timestamps and
    associating them with corresponding objects. The resulting detected events are
    then returned. This class interfaces with a logger to gather the needed event
    data and uses a context to manage event pairs and thresholds.
    """

    def update_context_and_events(self, context:MotionStatechartContext, segmind_context: SegmindContext,obj: List[Body]) -> List[DetectionEvent]:
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
        translation_events = [
            event
            for event in segmind_context.logger.get_events()
            if isinstance(event, TranslationEvent)
        ]
        loss_of_support_events = [
            event for event in segmind_context.logger.get_events() if isinstance(event, LossOfSupportEvent)
        ]
        events = []
        by_object = defaultdict(list)
        for event in translation_events:
            by_object[event.tracked_object].append(event)

        for loss_of_support_event in loss_of_support_events:
            for event in by_object.get(loss_of_support_event.tracked_object, []):
                if abs(loss_of_support_event.timestamp - event.timestamp) >= self.shift_threshold:
                    continue

                key = (loss_of_support_event.tracked_object.id, loss_of_support_event.with_object.id)
                if key in segmind_context.placing_pairs:
                    continue

                segmind_context.placing_pairs.add(key)

                events.append(
                    PickUpEvent(
                        tracked_object=event.tracked_object,
                    )
                )
                break

        return events
