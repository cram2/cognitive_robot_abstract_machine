from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from semantic_digital_twin.world import World


class ArtVipCategory(StrEnum):
    """
    Top-level asset category of the ArtVIP dataset :cite:t:`jin2026artvip`, and the
    folder name each category's objects are stored under.
    """

    IKEA_FURNITURE = "Ikea_furniture"
    SMALL_FURNITURE = "small_furniture"
    LARGE_FURNITURE = "large_furniture"
    HOUSEHOLD_ITEMS = "household_items"
    SMALL_APPLIANCES = "small_appliances"
    MAJOR_APPLIANCES = "major_appliances"
    INDUSTRIAL_MACHINERY = "industrial_machinery"
    MEDICAL_EQUIPMENT = "Medical_equipment"
    LAB_ITEMS = "lab_items"


@dataclass
class ArtVipObject:
    """
    An ArtVIP object parsed into a :class:`~semantic_digital_twin.world.World` by
    :class:`~semantic_digital_twin.adapters.artvip_dataset.loader.ArtVipDatasetLoader`.
    """

    world: World
    """
    One Body per rigid link the object's USD physics joints connect.

    A link with a :class:`~pxr.UsdPhysics.RevoluteJoint`/:class:`~pxr.UsdPhysics.PrismaticJoint` to its
    parent is connected by a matching
    :class:`~semantic_digital_twin.world_description.connections.RevoluteConnection`/
    :class:`~semantic_digital_twin.world_description.connections.PrismaticConnection`,
    with the joint's own authored limits and axis; every other link (including a link
    joined by a :class:`~pxr.UsdPhysics.FixedJoint`) is connected by a
    :class:`~semantic_digital_twin.world_description.connections.FixedConnection`.
    """

    category: ArtVipCategory
    """
    The object's category.
    """

    name: str
    """
    The object's folder name, e.g.
    ``"EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm"``.
    """
