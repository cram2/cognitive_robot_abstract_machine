from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Any

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.datastructures.events import MotionEvent, Event
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


# ToDo: See if we can create our own MotionStatechartNode or change its name (talk to simon)
@dataclass(repr=False, eq=False)
class DetectorStateChartNode(MotionStatechartNode):
    pass


@dataclass
class DetectorStateChart(MotionStatechart):
    """
    Statechart responsible for running the different motion detectors.

    Currently acts as a container for the detectors and inherits the
    functionality from MotionStatechart.
    """

    pass


IndexedBodyPairs = Dict[Body, Set[Body]]
"""
Type hint for dictionaries mapping bodies to sets of bodies
"""

#ToDo: Make SegmindContext aware of the detectors,
# so the parameters will be added when the detector is added to the context.
@dataclass
class SegmindContext(MotionStatechartContext):
    """
    Context object shared across the motion statechart detectors.

    Stores the latest detected contact and support relationships
    between bodies in the simulation as well as the event logger.
    """

    object_moving_status: Dict[Body, bool] = field(default_factory=dict)
    """
    Dictionary mapping each body to a boolean indicating if it is currently moving.
    """

    latest_contact_bodies: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies it is currently in contact with.
    """

    latest_support: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently support it.
    """

    latest_containments: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently contain it.
    """

    latest_poses: Dict[Body, List[Pose]] = field(default_factory=dict)
    """
    Dictionary mapping each body to a list of its recent poses (pose history).
    """

    latest_motion_events: Dict[Body, MotionEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active motion event, if any.
    """

    logger: Any = None
    """
    The event logger used to record detected events.
    """

    placing_pairs: set = field(default_factory=set)
    """
    Set of placing pairs, to avoid duplicate events
    """

    holes: List[Body] = field(default_factory=list)
    """
    List of bodies that can be considered holes
    """

    insertion_pairs: set = field(default_factory=set)
    """
    List of insertion pairs, to avoid duplicate events
    """


@dataclass
class AbstractDetector(ABC, DetectorStateChartNode):
    """
    Abstract base class for all detectors.
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

    def on_tick(self, context: SegmindContext) -> Optional[ObservationStateValues]:
        """
        Executes one update cycle of the detector.

        Determines the objects that should be checked for contacts,
        computes new contact relationships and triggers events if
        contact changes are detected.

        :return: ObservationStateValues.TRUE if events were triggered,
        otherwise ObservationStateValues.FALSE.
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
        events = self.update_context_and_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE


    def get_relation(self, tracked_objects: List[Body], predicate) -> Dict[Body, Set[Body]]:
        """
        Get the relation between tracked objects.

        :param tracked_objects: List of bodies to check for contact changes.
        :param predicate: Function that returns true if the objects are related.
        :return: Dictionary mapping bodies to sets of related bodies.
        """

        related_bodies: Dict[Body, Set[Body]] = {}
        bodies_with_collision = self.context.world.bodies_with_collision
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if body is obj:
                    continue
                if predicate(obj, body):
                    related_bodies.setdefault(obj, set()).add(body)
        return related_bodies



    @abstractmethod
    def update_context_and_events(self, tracked_objects: List[Body]) -> List[Event]:
        """
        Updates the stored contact relationships and generates events
        when changes are detected.

        Implementations define how contact changes are interpreted
        (e.g. new contact or loss of contact).

        :param tracked_objects: List of bodies to check for contact changes.
        :return: List of events generated during the update.
        """
        pass
