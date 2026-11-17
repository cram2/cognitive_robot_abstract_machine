from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.exceptions import ParsingError


@dataclass
class UnpairedProcthorWalls(ParsingError):
    """
    Raised when the walls of a house that hold no door do not come in pairs.

    Each wall is described once from either side, so its walls are always paired.
    """

    wall_count: int = field(kw_only=True)
    """
    The number of walls without a door.
    """

    def error_message(self) -> str:
        return f"Expected an even number of walls without a door, but found {self.wall_count}."

    def suggest_correction(self) -> str:
        return "Describe every wall from both of its sides."


@dataclass
class ProcthorDoorWallNotFound(ParsingError):
    """
    Raised when a door refers to a wall that the house does not describe.
    """

    door_id: str = field(kw_only=True)
    """
    The identifier of the door.
    """

    wall_id: str | None = field(kw_only=True)
    """
    The identifier of the wall the door refers to, or None if it refers to none.
    """

    def error_message(self) -> str:
        return f"Door {self.door_id} refers to wall {self.wall_id}, which the house does not describe."

    def suggest_correction(self) -> str:
        return "Give the door the identifiers of the two walls it sits in as 'wall0' and 'wall1'."
