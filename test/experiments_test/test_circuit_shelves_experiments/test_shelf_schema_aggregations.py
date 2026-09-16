from __future__ import annotations

import pytest

from experiments.scene_generation_experiments.utils import ObjectType
from experiments.scene_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.scene_generation_experiments.shelf_schema_aggregations import (
    RelationalCircuitExperimentShelfAggregations,
    RelationalCircuitExperimentShelfLayerAggregations,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale


def _object(source_id: str) -> RelationalCircuitExperimentObject2D:
    return RelationalCircuitExperimentObject2D(
        object_type=ObjectType.BOOK,
        scale=Scale(x=0.05, y=0.1, z=0.2),
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id=source_id,
    )


def _layer(object_count: int) -> RelationalCircuitExperimentShelfLayer:
    return RelationalCircuitExperimentShelfLayer(
        objects=[_object(f"book_{index}") for index in range(object_count)],
        theme_dominant_type=ObjectType.BOOK,
        relative_height=0.2,
    )


@pytest.mark.parametrize("object_count", [0, 1, 3, 7])
def test_a_layer_counts_the_objects_it_holds(object_count: int) -> None:
    """
    Aggregation statistics are the only thing a fitted circuit passes from a shelf to
    its layers, or from a layer to its objects.

    One that reports a constant leaves the two levels independent while still appearing
    in the model, so its value has to be pinned and not merely its presence.
    """
    layer = _layer(object_count)

    assert (
        RelationalCircuitExperimentShelfLayerAggregations(layer).total_count()
        == object_count
    )


@pytest.mark.parametrize("layer_count", [1, 2, 5])
def test_a_shelf_counts_the_layers_it_holds(layer_count: int) -> None:
    """
    How many levels a kind of shelf has is the one structural quantity the data
    supplies, and it is what decides how many slabs a sampled shelf gets.
    """
    shelf = RelationalCircuitExperimentShelf(
        scale=Scale(x=0.3, y=0.6, z=2.0),
        layers=[_layer(1) for _ in range(layer_count)],
        theme_dominant_type=ObjectType.BOOK,
    )

    assert (
        RelationalCircuitExperimentShelfAggregations(shelf).layer_count() == layer_count
    )
