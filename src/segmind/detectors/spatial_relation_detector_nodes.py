from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from segmind.datastructures.events import (
    Event,
    SupportEvent,
    LossOfSupportEvent,
    ContainmentEvent,
    LossOfContainmentEvent, ContactEvent, InsertionEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    SegmindContext,
    DetectorStateChartNode,
)
from semantic_digital_twin.reasoning.predicates import is_supported_by, InsideOf
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from abc import ABC, abstractmethod


@dataclass(eq=False, repr=False)
class BaseSupportDetector(DetectorStateChartNode, ABC):
    """
    Abstract base class for support-based detectors.

    Provides shared functionality for detecting support between
    bodies and generating events when support relationships change.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    context: SegmindContext = field(kw_only=True)
    """
    :param context: Segmind context containing world information,
    history and logging utilities.
    """

    def on_tick(self, context: SegmindContext) -> Optional[ObservationStateValues]:
        """
        Executes one detector update cycle.

        :param context: Motion statechart context.
        :return: TRUE if events were generated, FALSE otherwise.
        """
        objects_to_check = (
            [self.tracked_object]
            if self.tracked_object
            else [
                body
                for body in self.context.world.bodies
                if type(body.parent_connection) is Connection6DoF
            ]
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
                if is_supported_by(obj, body, max_intersection_height=0.01):
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
class BaseContainmentDetector(DetectorStateChartNode):
    """
    Abstract base class for contaiment-based detectors.

    Provides shared functionality for detecting containment between
    bodies and generating events when containment relationships change.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    context: SegmindContext = field(kw_only=True)
    """
    :param context: Segmind context containing world information,
    history and logging utilities.
    """

    def on_tick(self, context: SegmindContext) -> Optional[ObservationStateValues]:
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
        events = self.update_latest_containment_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)

        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    def get_containment_pairs(
        self, tracked_objects: List[Body]
    ) -> Dict[Body, Set[Body]]:
        """
        Computes support relationships.

        :param tracked_objects: Bodies that should be checked.
        :return: Mapping of body → supporting bodies.
        """
        containment_pairs: Dict[Body, Set[Body]] = {}
        bodies_with_collision = self.context.world.bodies_with_collision
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if obj is body:
                    continue
                if InsideOf(obj, body).compute_containment_ratio() > 0.9:
                    containment_pairs.setdefault(obj, set()).add(body)
        return containment_pairs

    @abstractmethod
    def update_latest_containment_and_trigger_events(
        self, objects_to_check: List[Body]
    ) -> List[Event]:
        """
        Updates the cached support relationships and generates events.

        :param objects_to_check: Bodies that should be evaluated for support changes.
        :return: List of generated support-related events.
        """

        pass


@dataclass(eq=False, repr=False)
class ContainmentDetector(BaseContainmentDetector):

    def update_latest_containment_and_trigger_events(
        self, objects_to_check: List[Body]
    ) -> List[Event]:
        new_containment_pairs = self.get_containment_pairs(objects_to_check)
        latest_containment = self.context.latest_containments
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

    def update_latest_containment_and_trigger_events(
        self, objects_to_check: List[Body]
    ) -> List[Event]:
        new_containment_pairs = self.get_containment_pairs(objects_to_check)
        latest_containment = self.context.latest_containments
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
class InsertionDetector(MotionStatechartNode):
    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    context: SegmindContext = field(kw_only=True)
    shift_threshold: float = 15.0

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
        events = self.update_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE


    def update_and_trigger_events(self, tracked_objs: List[Body]) -> List[Event]:
        events = []

        contact_events = [i for i in self.context.logger.get_events() if isinstance(i, ContactEvent)]
        contact_events_with_holes = [i for i in contact_events if i.with_object in self.context.holes]
        containment_event = [i for i in self.context.logger.get_events() if isinstance(i, ContainmentEvent)]



        for i in contact_events_with_holes:
            for j in containment_event:
                if i.tracked_object == j.tracked_object:
                    if abs(i.timestamp - j.timestamp) < self.shift_threshold:

                        # needs to be reworked, it doesnt work
                        key = (id(i), id(j))

                        # ✅ exclusivity check
                        if key in self.context.insertion_pairs:
                            continue

                        self.context.insertion_pairs.add(key)

                        e = InsertionEvent(i.tracked_object, [j.with_object], i.with_object, timestamp=i.timestamp)

                        events.append(e)
                        break
        return events
