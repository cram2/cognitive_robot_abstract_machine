"""Turn semantic objects into a feature dataframe for confidence-aware evaluation.

The out-of-distribution check needs the features of an object as a row of a
dataframe. This module bridges the semantic objects of a world to that dataframe:
:class:`Feature` names one column each, and the geometric ones are measured by
:class:`ObjectShapeAggregations`, which aggregates over the shapes an object's
collision geometry is built from.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import pandas as pd
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)
from semantic_digital_twin.world_description.geometry import Mesh, Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Any, List

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


# %% modelled object

SHAPES = "shapes"
"""Name of the field :class:`ObjectShapeAggregations` aggregates over."""


@dataclass
class ObjectShape:
    """The collision geometry of one object, as the shapes it is built from."""

    shapes: List[Shape] = field(default_factory=list)
    """The shapes making up the object's collision geometry."""

    @classmethod
    def from_annotation(cls, annotation: SemanticAnnotation) -> ObjectShape:
        """Describe the object a semantic annotation is attached to.

        :param annotation: The annotation whose root body's collision geometry is read.
        :return: The shapes of that collision geometry.
        """
        return cls(list(annotation.root.collision))


@dataclass
class ObjectShapeAggregations(AggregationStatistic[ObjectShape]):
    """Statistics describing the collision geometry of an :class:`ObjectShape`."""

    @aggregation_statistic(SHAPES)
    def volume(self) -> float:
        """The volume the object's collision geometry encloses.

        A mesh that is not watertight encloses no well-defined volume and is left out
        of the sum rather than raising, since a real object is commonly built from
        several convex pieces and only some of them need to be watertight for the
        total to stay meaningful.

        :return: The total volume in cubic meters, ``0.0`` if no shape is watertight.
        """
        return sum(
            shape.volume
            for shape in self.instance.shapes
            if isinstance(shape, Mesh) and shape.mesh.is_watertight
        )

    @aggregation_statistic(SHAPES)
    def aspect_ratio(self) -> float:
        """How tall the object stands relative to how wide it spreads.

        :return: The collision geometry's vertical extent divided by the greater of
            its two horizontal extents.
        """
        minimum, maximum = ShapeCollection(self.instance.shapes).combined_mesh.bounds
        horizontal_extent = maximum[:2] - minimum[:2]
        vertical_extent = maximum[2] - minimum[2]
        return float(vertical_extent / max(horizontal_extent))


# %% features


class Feature(enum.StrEnum):
    """The columns of the feature dataframe :func:`extract_feature_dataframe` produces.

    A :class:`~enum.StrEnum`, so a member equals and hashes like its string value: existing
    code that indexes a column by its plain string name keeps working unchanged.
    """

    CLASS = "class"
    """The object's semantic-annotation class."""

    VOLUME = "volume"
    """The object's collision geometry's total watertight mesh volume, in cubic meters."""

    ASPECT_RATIO = "aspect_ratio"
    """The object's height over its widest horizontal extent."""


# %% extraction


def extract_feature_dataframe(objects: List[Any]) -> pd.DataFrame:
    """Extract every :class:`Feature` from each object.

    The geometric features are read from each object's own collision geometry, which
    a data access object does not carry.

    :param objects: The semantic objects whose features are extracted.
    :return: One row per object, with a column per feature.
    """
    return pd.DataFrame(
        [
            {
                Feature.CLASS: ObjectClass(type(instance).__name__),
                **ObjectShapeAggregations(
                    instance=ObjectShape.from_annotation(instance), field_name=SHAPES
                ).apply_mapping(),
            }
            for instance in objects
        ]
    )
