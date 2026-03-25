from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, List

from giskardpy.motion_statechart.data_types import ObservationStateValues
from segmind.datastructures.events import (
    StopMotionEvent,
    SupportEvent,
    ContainmentEvent,
    Event,
    PlacingEvent, TranslationEvent, LossOfSupportEvent, PickUpEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    SegmindContext,
    DetectorStateChartNode,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from segmind.detectors.base import AbstractDetector


@dataclass
class AbstractInteractionDetector(AbstractDetector):
    """
    Abstract base class for interaction-based detectors.

    Provides shared functionality for monitoring interactions of
    bodies and generating events when detected.
    """

    shift_threshold: float = 15
    """
    The threshold for the time difference between two events to be considered an interaction.
    """

    def update_context_and_events(self, tracked_objects: List[Body]) -> List[Event]:
        pass

@dataclass
class PlacingDetector(AbstractInteractionDetector):


    def update_context_and_events(self, obj: List[Body]) -> List[Event]:
        """
        Updates the system context with new placing event instances based on past
        actions logged in the system. It analyzes and filters specific event types
        to detect new events that can be generated. This function ensures distinct
        events are created by maintaining exclusivity through a pairing mechanism.

        Args:
            obj (List[Body]): Input list of body objects to be processed. Currently
                              not utilized in the function logic.

        Returns:
            List[Event]: A list of PlacingEvent instances generated after analyzing
                         stop translation and support events.
        """
        stop_translation_event = [
            i
            for i in self.context.logger.get_events()
            if isinstance(i, StopMotionEvent)
        ]
        support_event = [
            i for i in self.context.logger.get_events() if isinstance(i, SupportEvent)
        ]

        events = []

        for i in support_event:
            for j in stop_translation_event:

                if i.tracked_object == j.tracked_object:
                    if abs(i.timestamp - j.timestamp) < self.shift_threshold:

                        key = (
                            i.tracked_object.id,
                            i.with_object.id,
                            #int(i.timestamp * 1000),
                        )

                        if key in self.context.placing_pairs:
                            continue
                        self.context.placing_pairs.add(key)
                        e = PlacingEvent(
                            tracked_object=j.tracked_object,
                            with_object=i.with_object,
                            timestamp=i.timestamp,
                        )
                        events.append(e)
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

    def update_context_and_events(self, obj: List[Body]) -> List[Event]:
        """
        Updates the context and generates a list of events based on translation events and loss
        of support events. The method identifies pairs of related events that are close in
        timestamp, determines if they are exclusive, and creates a new event when conditions
        are met.

        Parameters:
            obj (List[Body]): A list of body objects to update the context and analyze events.

        Returns:
            List[Event]: A list of generated events based on the processed relationships
            between translation and loss of support events.
        """
        translation_event = [
            i
            for i in self.context.logger.get_events()
            if isinstance(i, TranslationEvent)
        ]
        loss_of_support_event = [
            i for i in self.context.logger.get_events() if isinstance(i, LossOfSupportEvent)
        ]
        events = []
        for i in loss_of_support_event:
            for j in translation_event:
                if i.tracked_object == j.tracked_object:
                    if abs(i.timestamp - j.timestamp) < self.shift_threshold:
                        key = (
                            i.tracked_object.id,
                            i.with_object.id,
                        )
                        if key in self.context.placing_pairs:
                            continue
                        self.context.placing_pairs.add(key)
                        e = PickUpEvent(
                            tracked_object=j.tracked_object,
                            with_object=i.with_object,
                            timestamp=i.timestamp,
                        )
                        events.append(e)
                        break
        return events
