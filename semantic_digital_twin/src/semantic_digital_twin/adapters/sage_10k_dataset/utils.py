from __future__ import annotations

import logging
import os
from importlib.resources import files
from pathlib import Path
from enum import StrEnum

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


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


def create_hsrb_in_world(world: World):

    urdf_dir = os.path.join(
        Path(files("semantic_digital_twin")).parent.parent.parent,
        "pycram",
        "resources",
        "robots",
    )
    hsr = os.path.join(urdf_dir, "hsrb.urdf")

    hsrb_parser = URDFParser.from_file(file_path=hsr)
    world_with_hsrb = hsrb_parser.parse()
    with world_with_hsrb.modify_world():
        hsrb_root = world_with_hsrb.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_hsrb.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=hsrb_root, world=world_with_hsrb
        )
        world_with_hsrb.add_connection(c_root_bf)

    world.merge_world(world_with_hsrb)

    hsrb = HSRB.from_world(world)
    return hsrb
