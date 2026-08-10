from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from typing_extensions import Self

from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from semantic_digital_twin.adapters.ros.messages import StreamPosition

from giskardpy.motion_statechart.motion_statechart import MotionStatechart

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

    motion_statechart_json_data: Dict[str, Any]
    """
    The motion statechart to execute, as json.
    """

    required_position: Optional[StreamPosition] = field(default=None, kw_only=True)
    """
    The position in the client's stream that the world has to contain before the motion
    statechart may be parsed.

    ``None`` when the client never changed the world, in which case there is nothing to
    wait for.
    """

    @classmethod
    def for_motion_statechart(
        cls,
        motion_statechart: MotionStatechart,
        required_position: Optional[StreamPosition] = None,
    ) -> MotionGoal:
        """
        Build the goal that asks for the given motion statechart.
        """
        return cls(
            motion_statechart_json_data=motion_statechart.to_json(),
            required_position=required_position,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "motion_statechart": self.motion_statechart_json_data,
            "required_position": (
                None
                if self.required_position is None
                else to_json(self.required_position)
            ),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Rebuild a goal from what a client sent.
        """
        required_position = data.get("required_position")
        return cls(
            motion_statechart_json_data=data["motion_statechart"],
            required_position=(
                None if required_position is None else from_json(required_position)
            ),
        )

    def parse_motion_statechart(self, **kwargs) -> MotionStatechart:
        """
        Resolve the motion statechart against the world described by the given kwargs.
        """
        return MotionStatechart.from_json(self.motion_statechart_json_data, **kwargs)
