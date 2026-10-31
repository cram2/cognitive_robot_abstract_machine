from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Self, Any


from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from semantic_digital_twin.adapters.ros.messages import MetaData, StreamPosition

from cramph.statechart import Statechart

# %% the payload of a motion goal


@dataclass
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

    client: MetaData = field(kw_only=True)
    """
    Who is waiting for this goal.

    Giskard watches that client while the motion runs and stops the robot once it is
    gone, so the goal has to say who it belongs to. The action itself cannot tell: rclpy
    hands the server the sequence number of the request, not the participant that sent
    it.
    """

    required_position: StreamPosition | None = field(default=None, kw_only=True)
    """
    The position in the client's stream that the world has to contain before the motion
    statechart may be parsed.

    ``None`` when the client never changed the world, in which case there is nothing to
    wait for.
    """

    @classmethod
    def for_motion_statechart(
        cls,
        motion_statechart: Statechart,
        client: MetaData,
        required_position: StreamPosition | None = None,
    ) -> MotionGoal:
        """
        Build the goal that asks for the given motion statechart.
        """
        return cls(
            motion_statechart_json_data=motion_statechart.to_json(),
            client=client,
            required_position=required_position,
        )

    def to_json(self, **kwargs) -> dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            "motion_statechart": self.motion_statechart_json_data,
            "client": to_json(self.client),
            "required_position": (
                None
                if self.required_position is None
                else to_json(self.required_position, **kwargs)
            ),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        """
        Rebuild a goal from what a client sent.
        """
        required_position = data.get("required_position")
        return cls(
            motion_statechart_json_data=data["motion_statechart"],
            client=from_json(data["client"]),
            required_position=(
                None if required_position is None else from_json(required_position)
            ),
        )

    def parse_motion_statechart(self, **kwargs: Any) -> Statechart:
        """
        Resolve the motion statechart against the world described by the given kwargs.
        """
        return Statechart.from_json(self.motion_statechart_json_data, **kwargs)
