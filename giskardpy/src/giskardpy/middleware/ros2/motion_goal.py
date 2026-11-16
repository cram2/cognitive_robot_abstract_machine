from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing_extensions import Self, Any


from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from semantic_digital_twin.adapters.ros.messages import MetaData, StreamPosition

from cramph.statechart import Statechart

# %% the payload of a motion goal


class MotionGoalPayloadKey(StrEnum):
    """
    The keys of the json a client sends to ask for a motion.
    """

    MOTION_STATECHART = "motion_statechart"
    """
    The motion statechart to execute, still serialized.
    """

    CLIENT = "client"
    """
    The client that waits for the goal.
    """

    REQUIRED_POSITION = "required_position"
    """
    The change of the client's world the motion statechart was built on.
    """

    PUBLISHES_DEBUG_EXPRESSIONS = "publishes_debug_expressions"
    """
    Whether Giskard publishes the debug markers of the goal.
    """


@dataclass(kw_only=True)
class MotionGoal(SubclassJSONSerializer):
    """
    What a client asks Giskard to execute, together with the change of the world the
    request was built on.

    The motion statechart stays serialized until it is parsed, because parsing resolves
    the entities it refers to against a world that still has to catch up with
    ``required_position``.
    """

    motion_statechart_json_data: dict[str, Any]
    """
    The motion statechart to execute, as json.
    """

    client: MetaData
    """
    Who is waiting for this goal.

    Giskard watches that client while the motion runs and stops the robot once it is
    gone, so the goal has to say who it belongs to. The action itself cannot tell: rclpy
    hands the server the sequence number of the request, not the participant that sent
    it.
    """

    required_position: StreamPosition | None = None
    """
    The position in the client's stream that the world has to contain before the motion
    statechart may be parsed.

    ``None`` when the client never changed the world, in which case there is nothing to
    wait for.
    """

    publishes_debug_expressions: bool = False
    """
    Whether Giskard publishes the spatial debug expressions of the running nodes as RViz
    markers while it executes this goal.
    """

    @classmethod
    def for_motion_statechart(
        cls,
        *,
        motion_statechart: Statechart,
        client: MetaData,
        required_position: StreamPosition | None = None,
        publishes_debug_expressions: bool = False,
    ) -> MotionGoal:
        """
        Build the goal that asks for the given motion statechart.
        """
        return cls(
            motion_statechart_json_data=motion_statechart.to_json(),
            client=client,
            required_position=required_position,
            publishes_debug_expressions=publishes_debug_expressions,
        )

    def to_json(self, **kwargs) -> dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            MotionGoalPayloadKey.MOTION_STATECHART: self.motion_statechart_json_data,
            MotionGoalPayloadKey.CLIENT: to_json(self.client),
            MotionGoalPayloadKey.REQUIRED_POSITION: (
                None
                if self.required_position is None
                else to_json(self.required_position, **kwargs)
            ),
            MotionGoalPayloadKey.PUBLISHES_DEBUG_EXPRESSIONS: self.publishes_debug_expressions,
        }

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs) -> Self:
        """
        Rebuild a goal from what a client sent.

        A client that sends no :attr:`publishes_debug_expressions` gets its default.
        """
        required_position = data.get(MotionGoalPayloadKey.REQUIRED_POSITION)
        return cls(
            motion_statechart_json_data=data[MotionGoalPayloadKey.MOTION_STATECHART],
            client=from_json(data[MotionGoalPayloadKey.CLIENT]),
            required_position=(
                None if required_position is None else from_json(required_position)
            ),
            publishes_debug_expressions=data.get(
                MotionGoalPayloadKey.PUBLISHES_DEBUG_EXPRESSIONS,
                cls.publishes_debug_expressions,
            ),
        )

    def parse_motion_statechart(self, **kwargs: Any) -> Statechart:
        """
        Resolve the motion statechart against the world described by the given kwargs.
        """
        return Statechart.from_json(data=self.motion_statechart_json_data, **kwargs)
