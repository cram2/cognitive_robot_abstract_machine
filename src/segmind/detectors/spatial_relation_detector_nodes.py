from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from segmind.datastructures.events import Event, SupportEvent, LossOfSupportEvent
from segmind.detectors.atomic_event_detectors_nodes import SegmindContext
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from abc import ABC, abstractmethod


@dataclass(eq=False, repr=False)
class BaseSupportDetector(MotionStatechartNode, ABC):
    """
    Abstract base class for support-based detectors.

    Provides shared functionality for detecting support between
    bodies and generating events when support relationships change.

    :param tracked_object: Optional body that should be monitored.
        If None, all trackable objects in the world are checked.
    :param context: Segmind context containing world information,
         history and logging utilities.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    context: SegmindContext = field(kw_only=True)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Executes one detector update cycle.

        :param context: Motion statechart context.
        :return: TRUE if events were generated, FALSE otherwise.
        """
        trackable_objects = [
            body
            for body in self.context.world.bodies
            if type(body.parent_connection) is Connection6DoF
        ]
        objects_to_check = (
            [self.tracked_object] if self.tracked_object else trackable_objects
        )
        events = self.update_latest_support_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)

        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    def get_support_pairs(self, tracked_objects: List[Body]) -> Dict[Body, Set[Body]]:
        """
        Computes support relationships.

        :param tracked_objects: Bodies that should be checked.
        :return: Mapping of body → supporting bodies.
        """
        support_pairs: Dict[Body, Set[Body]] = {}
        bodies_with_collision = self.context.world.bodies_with_collision
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if obj is body:
                    continue
                if is_supported_by(obj, body):
                    support_pairs.setdefault(obj, set()).add(body)
        return support_pairs

    @abstractmethod
    def update_latest_support_and_trigger_events(
        self, objects_to_check: List[Body]
    ) -> List[Event]:
        """
        Updates the cached support relationships and generates events.

        :param objects_to_check: Bodies that should be evaluated for support changes.
        :return: List of generated support-related events.
        """

        pass


@dataclass(eq=False, repr=False)
class SupportDetector(BaseSupportDetector):

    def update_latest_support_and_trigger_events(
        self, objects_to_check: List[Body]
    ) -> List[Event]:
        """
        Detects newly established support relationships.

        :param objects_to_check: Bodies that should be evaluated for new supports.
        :return: List of SupportEvent objects representing newly detected supports.
        """

        events = []
        latest_support = self.context.latest_support
        new_support_pairs = self.get_support_pairs(objects_to_check)
        for body, support in new_support_pairs.items():
            if body not in latest_support:
                latest_support[body] = support
                for s in support:
                    events.append(SupportEvent(tracked_object=body, with_object=s))
            else:
                new_supports = support - latest_support[body]
                if new_supports:
                    latest_support[body] |= new_supports
                    for s in new_supports:
                        events.append(SupportEvent(tracked_object=body, with_object=s))

        return events


@dataclass(eq=False, repr=False)
class LossOfSupportDetector(BaseSupportDetector):

    def update_latest_support_and_trigger_events(
        self, objects_to_check: List[Body]
    ) -> List[Event]:
        """
        Detects when previously existing support relationships are lost.

        :param objects_to_check: Bodies that should be evaluated for lost supports.
        :return: List of LossOfSupportEvent objects representing removed supports.
        """

        events = []
        latest_support = self.context.latest_support
        new_support_pairs = self.get_support_pairs(objects_to_check)
        for body, support in list(latest_support.items()):
            if body not in new_support_pairs:
                latest_support.pop(body)
                for s in support:
                    events.append(
                        LossOfSupportEvent(tracked_object=body, with_object=s)
                    )
            else:
                new_supports = new_support_pairs[body]
                lost_supports = support - new_supports
                if lost_supports:
                    latest_support[body] = new_supports
                    for s in lost_supports:
                        events.append(
                            LossOfSupportEvent(tracked_object=body, with_object=s)
                        )

        return events
