"""Turn semantic objects into a feature dataframe for confidence-aware evaluation.

The out-of-distribution check needs the features of an object as a row of a
dataframe. This module bridges the semantic objects of a world to that dataframe:
:class:`Feature` names one column each and reads its own value off an object, through
the :class:`ObjectFeature` that measures it.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from semantic_digital_twin.world_description.geometry import Mesh
from typing_extensions import Any, Dict, List, Type

# %% object classes


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


# %% features


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

    def extract(self, instance: Any) -> Any:
        """Read this feature's value off one semantic object.

        :param instance: The semantic object to measure.
        :return: The value this feature's column holds for that object.
        """
        features: Dict[Feature, Type[ObjectFeature]] = {
            Feature.CLASS: SemanticClass,
            Feature.VOLUME: CollisionVolume,
            Feature.ASPECT_RATIO: AspectRatio,
        }
        return features[self](instance).value()


@dataclass
class ObjectFeature(ABC):
    """Measures the value one :class:`Feature` holds for a semantic object."""

    instance: Any
    """The semantic object being measured."""

    @abstractmethod
    def value(self) -> Any:
        """
        :return: The measured value.
        """


@dataclass
class SemanticClass(ObjectFeature):
    """The class an object's semantic annotation is an instance of."""

    def value(self) -> ObjectClass:
        """
        :return: The object's class as an :class:`ObjectClass` member.
        """
        return ObjectClass(type(self.instance).__name__)


@dataclass
class CollisionVolume(ObjectFeature):
    """The volume an object's root body collision geometry encloses.

    A collision mesh that is not watertight has no well-defined enclosed volume and is
    left out of the sum rather than raising, since a real object is commonly made of
    several convex collision pieces and only some of them need to be watertight for the
    total to still be meaningful.
    """

    def value(self) -> float:
        """
        :return: The total volume, in cubic meters; ``0.0`` if no collision shape is
            watertight.
        """
        return sum(
            shape.volume
            for shape in self.instance.root.collision
            if isinstance(shape, Mesh) and shape.mesh.is_watertight
        )


@dataclass
class AspectRatio(ObjectFeature):
    """How tall an object stands relative to how wide it spreads."""

    def value(self) -> float:
        """
        :return: The collision geometry's extent along the vertical axis divided by the
            greater of its two horizontal extents.
        """
        minimum, maximum = self.instance.root.collision.combined_mesh.bounds
        horizontal_extent = maximum[:2] - minimum[:2]
        vertical_extent = maximum[2] - minimum[2]
        return float(vertical_extent / max(horizontal_extent))


# %% extraction


def extract_feature_dataframe(objects: List[Any]) -> pd.DataFrame:
    """Extract every :class:`Feature` from each object.

    The feature values are read from each object directly, including from collision
    geometry a data access object does not carry.

    :param objects: The semantic objects whose features are extracted.
    :return: One row per object, with a column per feature.
    """
    return pd.DataFrame(
        {
            feature: [feature.extract(instance) for instance in objects]
            for feature in Feature
        }
    )
