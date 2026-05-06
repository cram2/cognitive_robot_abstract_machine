from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from enum import StrEnum

from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Room,
    Wall,
    Door,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class NaturalLanguageWithTypeDescription(NaturalLanguageDescription):
    """
    A natural language description of a Sage10k object including the type information of the object.
    """

    type_description: Optional[str] = field(default=None)
    """
    The cleaned description of the type of the object.
    """


class Sage10kActionableScenes(StrEnum):
    """
    A collection of Sage10k scenes that can be used for demonstration purposes.
    """

    GYM = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_171403_layout_26384448.zip"
    TV_STUDIO = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_180931_layout_d83fc25f.zip"
    CRAFTSMAN_LOBBY = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_205353_layout_9584241f.zip"
    TROPICAL_WAREHOUSE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251214_182016_layout_a72cf11f.zip"
    VAPORWAVE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_090236_layout_7e07a47a.zip"
    ECLECTIC_RESIDENCE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_090413_layout_d59e4e4b.zip"
    SOUTHWESTERN_STORE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_123747_layout_2d89d0a5.zip"
    BRUTALIST_STORE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_153933_layout_50ffb500.zip"
    AMERICAN_BUFFET_RESTAURANT = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_172548_layout_edf26267.zip"


@dataclass(eq=False)
class RoomWithWallsAndDoors(Room):

    room_type: Optional[str] = field(kw_only=True, default=None)
    """
    Description of the type of the room in natural language.
    """

    walls: List[Wall] = field(kw_only=True, default_factory=list)
    """
    The walls enclosing this room.
    """

    doors: List[Door] = field(kw_only=True, default_factory=list)
    """
    The doors of the room.
    """


@dataclass(eq=False)
class DoorWithType(Door):
    type_description: Optional[str] = field(kw_only=True, default=None)
