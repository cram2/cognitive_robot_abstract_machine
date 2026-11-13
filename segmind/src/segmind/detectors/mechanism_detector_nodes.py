"""
Detecting a moving part of the scene, such as a drawer or a door, being opened, being
closed, and leaving either.

What is read here is how far a part stands along the joint it moves on, rather than a
relation between two bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from segmind.datastructures.events import (
    ClosedEvent,
    DetectionEvent,
    OpenedEvent,
    PartlyOpenEvent,
)
from segmind.detectors.base import AbstractDetector, SegmindContext
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


class MechanismState(StrEnum):
    """
    How far a moving part stands along its joint.
    """

    OPEN = "open"
    """
    At least as far out as counts as open.
    """

    CLOSED = "closed"
    """
    At most as far out as counts as closed.
    """

    PARTLY_OPEN = "partly open"
    """
    In between.
    """


@dataclass(eq=False, repr=False)
class OpeningDetector(AbstractDetector):
    """
    Reports a moving part coming to stand open, closed, or partly open from either.

    The part is a body moving on a joint of its own, a drawer on its rails or a door on
    its hinge, and how far it stands is the share of its joint's travel it stands out.
    The state a part is first seen in is where the run starts, and is not reported.
    """

    open_ratio: float = field(default=0.9, kw_only=True)
    """
    The share of its travel from which on a part counts as open.
    """

    closed_ratio: float = field(default=0.05, kw_only=True)
    """
    The share of its travel up to which a part counts as closed.
    """

    states: dict[Body, MechanismState] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The state each part was last seen in.
    """

    def update_context_and_events(
        self,
        segmind_context: SegmindContext,
        tracked_objects: list[Body],
    ) -> list[DetectionEvent]:
        """
        Detects parts whose state changed since they were last seen.

        :param segmind_context: The shared SegmindContext; nothing in it is needed.
        :param tracked_objects: The bodies to check; those not moving on a joint of
            their own are left out.
        :return: One event per part whose state changed.
        """
        events: list[DetectionEvent] = []
        for body in tracked_objects:
            joint = body.parent_connection
            if not isinstance(joint, ActiveConnection1DOF):
                continue
            state = self.state_of(joint)
            seen_before = body in self.states
            changed = self.states.get(body) is not state
            self.states[body] = state
            if not (seen_before and changed):
                continue
            match state:
                case MechanismState.OPEN:
                    events.append(OpenedEvent(tracked_object=body))
                case MechanismState.CLOSED:
                    events.append(ClosedEvent(tracked_object=body))
                case MechanismState.PARTLY_OPEN:
                    events.append(PartlyOpenEvent(tracked_object=body))
        return events

    def state_of(self, joint: ActiveConnection1DOF) -> MechanismState:
        """
        :param joint: The joint a part moves on.
        :return: The state the part stands in.
        """
        limits = joint.dof.limits
        ratio = (joint.position - limits.lower.position) / (
            limits.upper.position - limits.lower.position
        )
        if ratio >= self.open_ratio:
            return MechanismState.OPEN
        if ratio <= self.closed_ratio:
            return MechanismState.CLOSED
        return MechanismState.PARTLY_OPEN
