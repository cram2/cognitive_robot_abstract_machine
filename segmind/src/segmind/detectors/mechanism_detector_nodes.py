"""
Detecting a moving part of the scene, such as a drawer or a door, being opened and being
closed.

What is read here is how far a part stands along the joint it moves on, rather than a
relation between two bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from segmind.datastructures.events import (
    ClosedEvent,
    DetectionEvent,
    OpenedEvent,
    OpeningEvent,
)
from segmind.detectors.base import AbstractDetector, SegmindContext
from semantic_digital_twin.semantic_annotations.mixins import HasMovableJoint
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class OpeningDetector(AbstractDetector):
    """
    Reports a moving part, such as a door or a drawer, coming to stand open or closed.

    How far a part stands open is the
    :attr:`~semantic_digital_twin.semantic_annotations.mixins.HasMovableJoint.opening_ratio`
    of the annotation the part is the root of. The state a part is first seen in is
    where the run starts, and is not reported.
    """

    open_ratio: float = field(default=0.9, kw_only=True)
    """
    The opening ratio from which on a part counts as open.
    """

    closed_ratio: float = field(default=0.05, kw_only=True)
    """
    The opening ratio up to which a part counts as closed.
    """

    last_states: dict[Body, type[OpeningEvent] | None] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Whether each part was last seen open, closed, or, as ``None``, in between.
    """

    def update_context_and_events(
        self,
        segmind_context: SegmindContext,
        tracked_objects: list[Body],
    ) -> list[DetectionEvent]:
        """
        Detects parts that came to stand open or closed since they were last seen.

        :param segmind_context: The shared SegmindContext; nothing in it is needed.
        :param tracked_objects: The bodies to check; those that are not the root of a
            part moving on its joint are left out.
        :return: One event per part that came to stand open or closed.
        """
        events: list[DetectionEvent] = []
        tracked = set(tracked_objects)
        for moving_part in self.context.world.get_semantic_annotations_by_type(
            HasMovableJoint
        ):
            body = moving_part.root
            if body not in tracked or moving_part.movable_joint is None:
                continue
            opening_ratio = moving_part.opening_ratio
            state = self.state_at(opening_ratio=opening_ratio)
            seen_before = body in self.last_states
            changed = self.last_states.get(body) is not state
            self.last_states[body] = state
            if seen_before and changed and state is not None:
                events.append(state(tracked_object=body, opening_ratio=opening_ratio))
        return events

    def state_at(self, *, opening_ratio: float) -> type[OpeningEvent] | None:
        """
        :param opening_ratio: How far a part stands open.
        :return: The event a part standing at ``opening_ratio`` is reported with, or
            ``None`` when it stands between closed and open.
        """
        if opening_ratio >= self.open_ratio:
            return OpenedEvent
        if opening_ratio <= self.closed_ratio:
            return ClosedEvent
        return None
