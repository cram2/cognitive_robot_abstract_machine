from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, List

from giskardpy.motion_statechart.data_types import ObservationStateValues
from segmind.datastructures.events import (
    StopMotionEvent,
    SupportEvent,
    ContainmentEvent,
    Event,
    PlacingEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    SegmindContext,
    DetectorStateChartNode,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class AbstractInteractionDetector(DetectorStateChartNode, ABC):
    """
    Abstract base class for interaction-based detectors.

    Provides shared functionality for monitoring interactions of
    bodies and generating events when detected.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    context: SegmindContext = field(kw_only=True)
    """
    :param context: Segmind context containing world information,
    contact history and logging utilities.
    """

    shift_threshold: float = 15

    placing_pairs: set = field(default_factory=set)

    def on_tick(self, context: SegmindContext) -> Optional[ObservationStateValues]:
        objects_to_check = (
            [self.tracked_object]
            if self.tracked_object
            else [
                body
                for body in self.context.world.bodies
                if type(body.parent_connection) is Connection6DoF
            ]
        )
        events = self.check_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)

        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    @abstractmethod
    def check_and_trigger_events(self, obj: List[Body]) -> List[Event]:
        pass


@dataclass
class PlacingDetector(AbstractInteractionDetector):
    # Here we need to look at different logged events and by time, check if its a placing event
    # PlacingEvent -> StopMotionEvent + Support or Containment Event

    def check_and_trigger_events(self, obj: Body) -> List[Event]:

        stop_translation_event = [
            i
            for i in self.context.logger.get_events()
            if isinstance(i, StopMotionEvent)
        ]
        support_event = [
            i for i in self.context.logger.get_events() if isinstance(i, SupportEvent)
        ]

        events = []
        # Now we need to check if the stop translation event and the support event are close enough timestamp wise
        for i in stop_translation_event:
            for j in support_event:

                if i.tracked_object == j.tracked_object:

                    if abs(i.timestamp - j.timestamp) < self.shift_threshold:

                        # needs to be reworked, it doesnt work
                        key = (id(i), id(j))

                        # ✅ exclusivity check
                        if key in self.placing_pairs:
                            continue

                        self.placing_pairs.add(key)

                        e = PlacingEvent(
                            tracked_object=i.tracked_object,
                            with_object=j.with_object,
                            timestamp=i.timestamp,
                        )

                        events.append(e)

        return events
