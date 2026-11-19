"""
Exceptions raised by the data structures of the semantic digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.joint_state import JointState
    from semantic_digital_twin.world_description.connections import (
        ActiveConnection1DOF,
    )

# %% joint states


@dataclass(kw_only=True)
class ConnectionNotInJointStateError(UsageError):
    """
    Raised when a joint state is asked for the target value of a connection it sets no
    target value for.
    """

    joint_state: JointState
    """
    The joint state that was asked.
    """

    connection: ActiveConnection1DOF
    """
    The connection the joint state sets no target value for.
    """

    def error_message(self) -> str:
        return (
            f"Joint state {self.joint_state.name} sets no target value for connection "
            f"{self.connection.name}."
        )

    def suggest_correction(self) -> str:
        return (
            "Ask for one of the connections the joint state sets: "
            f"{[connection.name for connection in self.joint_state.connections]}."
        )
