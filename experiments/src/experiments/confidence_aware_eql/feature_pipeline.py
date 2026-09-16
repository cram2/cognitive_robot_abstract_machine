"""Turn semantic objects into a feature dataframe for confidence-aware evaluation.

The out-of-distribution check needs the features of an object as a row of a
dataframe. This module bridges the semantic objects of a world to that dataframe: the
object's class together with its collision geometry's volume and aspect ratio are
kept as the features the confidence model is learned on.
"""

from __future__ import annotations

import enum

import pandas as pd
from semantic_digital_twin.world_description.geometry import Mesh
from typing_extensions import Any, List


class ObjectClass(enum.StrEnum):
    """The semantic-annotation classes the confidence model distinguishes.

    A :class:`~enum.StrEnum`, so a member equals and hashes like its string value.
    Enum members carry the class name as their value, so an instance's class is
    looked up with ``ObjectClass(type(instance).__name__)``. Covers every class
    :class:`~semantic_digital_twin.adapters.robocasa_dataset.semantics.RoboCasaObjectResolver`
    maps a robocasa object category to.
    """

    APPLE = "Apple"
    BANANA = "Banana"
    ORANGE = "Orange"
    TOMATO = "Tomato"
    LETTUCE = "Lettuce"
    CARROT = "Carrot"
    POTATO = "Potato"
    BOTTLE = "Bottle"
    CUP = "Cup"
    MUG = "Mug"
    BOWL = "Bowl"
    PLATE = "Plate"
    PAN = "Pan"
    POT = "Pot"
    KETTLE = "Kettle"
    BREAD = "Bread"


class Feature(enum.StrEnum):
    """The columns of the feature dataframe :func:`extract_feature_dataframe` produces.

    A :class:`~enum.StrEnum`, so a member equals and hashes like its string value: existing
    code that indexes a column by its plain string name keeps working unchanged.
    """

    CLASS = "class"
    """The object's semantic-annotation class."""

    VOLUME = "volume"
    """The object's root body collision geometry's total watertight mesh volume, in cubic meters."""

    ASPECT_RATIO = "aspect_ratio"
    """The object's height over its widest horizontal extent."""


def _collision_volume(instance: Any) -> float:
    """Sum the watertight collision mesh volume of an object's root body.

    A collision mesh that is not watertight has no well-defined enclosed volume and is
    excluded from the sum rather than raising, since a real object is commonly made of
    several convex collision pieces and only some of them need to be watertight for
    the total to still be meaningful.

    :param instance: The semantic object whose root body's collision volume is summed.
    :return: The total volume, in cubic meters; ``0.0`` if no collision shape is watertight.
    """
    return sum(
        shape.volume
        for shape in instance.root.collision
        if isinstance(shape, Mesh) and shape.mesh.is_watertight
    )


def _aspect_ratio(instance: Any) -> float:
    """The height of an object's root body collision geometry over its widest horizontal extent.

    :param instance: The semantic object whose shape is measured.
    :return: The extent along the vertical axis divided by the greater of the two
        horizontal extents.
    """
    minimum, maximum = instance.root.collision.combined_mesh.bounds
    horizontal_extent, vertical_extent = maximum[:2] - minimum[:2], maximum[2] - minimum[2]
    return float(vertical_extent / max(horizontal_extent))


def extract_feature_dataframe(objects: List[Any]) -> pd.DataFrame:
    """Extract the class, volume, and aspect ratio of each object as a feature dataframe.

    The feature values are read from each object's own collision geometry directly,
    which a data access object does not carry.

    :param objects: The semantic objects whose features are extracted.
    :return: One row per object with a class, volume, and aspect ratio column.
    """
    return pd.DataFrame(
        {
            Feature.CLASS: [ObjectClass(type(instance).__name__) for instance in objects],
            Feature.VOLUME: [_collision_volume(instance) for instance in objects],
            Feature.ASPECT_RATIO: [_aspect_ratio(instance) for instance in objects],
        }
    )
