from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Callable, Any
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    SupportEvent,
    DetectionEvent,
    PlacingEvent, TranslationEvent, LossOfSupportEvent, PickUpEvent, StopTranslationEvent, ContactEvent,
    ContainmentEvent, InsertionEvent,
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
        tracked object whose timestamp is within :attr:`shift_threshold`. If such a pair
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
            e for e in segmind_context.logger.get_events()
            if isinstance(e, primary_event_type)
        ]
        secondary_events = [
            e for e in segmind_context.logger.get_events()
            if isinstance(e, secondary_event_type)
        ]

        events = []
        by_object = defaultdict(list)
        for e in primary_events:
            by_object[e.tracked_object].append(e)

        for secondary in secondary_events:
            for primary in by_object.get(secondary.tracked_object, []):
                if abs(secondary.timestamp - primary.timestamp) >= self.shift_threshold:
                    continue

                key = (secondary.tracked_object.id, secondary.with_object.id)
                if key in segmind_context.placing_pairs:
                    continue

                segmind_context.placing_pairs.add(key)
                events.append(make_event(primary, secondary))
                break

        return events


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
        return self._find_interaction_events(
            segmind_context,
            primary_event_type=StopTranslationEvent,
            secondary_event_type=SupportEvent,
            make_event=lambda primary, secondary: PlacingEvent(
                tracked_object=primary.tracked_object,
                with_object=secondary.with_object,
            ),
        )


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
        return self._find_interaction_events(
            segmind_context,
            primary_event_type=TranslationEvent,
            secondary_event_type=LossOfSupportEvent,
            make_event=lambda primary, secondary: PickUpEvent(
                tracked_object=primary.tracked_object,
            ),
        )

    @dataclass(eq=False, repr=False)
    class InsertionDetector(AbstractInteractionDetector):
        """
        Detects insertion events based on object interaction context.

        The InsertionDetector class is used to analyze the interaction between tracked
        objects and identify insertion events. It tracks specific events such as
        contacts and containment, and generates an InsertionEvent when specific
        conditions are met. The class leverages a context that holds relevant
        event logs and tracked objects.
        """

        def update_context_and_events(self, context: MotionStatechartContext, segmind_context: SegmindContext,
                                      tracked_objs: List[Body]) -> List[DetectionEvent]:
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
            contact_events = [i for i in segmind_context.logger.get_events() if isinstance(i, ContactEvent)]
            contact_events_with_holes = [i for i in contact_events if i.with_object in segmind_context.holes]
            containment_event = [i for i in segmind_context.logger.get_events() if isinstance(i, ContainmentEvent)]

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
