"""
Flattening recorded attempts into one fixed-width table, named the way EQL names the
same attributes, so a query built for the relational pipeline reads the flat table's
columns unchanged.

Values are kept as they are -- an enum member stays a member -- which is also how a
fitted tree's leaves and a query's conditions read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import pandas as pd
from krrood.utils import get_class_and_attribute_name
from typing_extensions import Any, Callable, Dict, Iterable, List, Tuple

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutteredObject,
    ClutterPickScene,
    ClutterPickSceneAggregations,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    FlatTableSchemaMismatchError,
)


@dataclass(frozen=True)
class SceneSchema:
    """
    The attributes of an attempt as EQL names them: the scene's own scalars, its
    aggregation statistics, and each neighbour's attributes under the neighbour's index.
    """

    neighbours_field: str = "neighbours"
    """
    The exchangeable-part field of
    :class:`~experiments.causal_reasoning.tracy_clutter_picking.domain.ClutterPickScene` that holds
    the neighbours.
    """

    @property
    def aggregation_statistics(self) -> Tuple[Callable[..., Any], ...]:
        """
        The aggregation statistics of
        :class:`~experiments.causal_reasoning.tracy_clutter_picking.domain.ClutterPickSceneAggregations`
        over the neighbours; the flat table carries them as columns, so it holds the
        same summaries the relational model derives from its parts.
        """
        return tuple(
            ClutterPickSceneAggregations.aggregation_registry[self.neighbours_field]
        )

    @property
    def scene_scalar_fields(self) -> Tuple[str, ...]:
        """
        The scene's own scalar attributes.
        """
        return tuple(
            scene_field.name
            for scene_field in fields(ClutterPickScene)
            if scene_field.name != self.neighbours_field
        )

    @property
    def neighbour_fields(self) -> Tuple[str, ...]:
        """
        A neighbour's attributes.
        """
        return tuple(
            neighbour_field.name for neighbour_field in fields(ClutteredObject)
        )

    def scene_column(self, field_name: str) -> str:
        """
        :param field_name: A scene scalar field.
        :return: The column, and EQL variable, name of that field.
        """
        return get_class_and_attribute_name(ClutterPickScene.__name__, field_name)

    def neighbour_column(self, index: int, field_name: str) -> str:
        """
        :param index: The neighbour's position in the scene's neighbour list.
        :param field_name: A neighbour field.
        :return: The column, and EQL variable, name of that neighbour's field.
        """
        return self.scene_column(f"{self.neighbours_field}[{index}].{field_name}")

    def aggregation_column(self, statistic_name: str) -> str:
        """
        :param statistic_name: The name of one of the :attr:`aggregation_statistics`.
        :return: The column, and EQL variable, name of that statistic, which grounding
            names by its class and its call.
        """
        return get_class_and_attribute_name(
            ClutterPickSceneAggregations.__name__, f"{statistic_name}()"
        )

    @property
    def aggregation_columns(self) -> Tuple[str, ...]:
        """
        The column names of every aggregation statistic.
        """
        return tuple(
            self.aggregation_column(statistic.__name__)
            for statistic in self.aggregation_statistics
        )


@dataclass
class FlatTable:
    """
    Recorded attempts as one row each, every scene with the same neighbour count.
    """

    neighbour_count: int
    """
    How many neighbours every row unrolls; scenes with another count cannot be rows.
    """

    schema: SceneSchema = field(default_factory=SceneSchema)
    """
    How the columns are named.
    """

    @property
    def columns(self) -> List[str]:
        """
        The table's columns, in order.
        """
        return (
            [self.schema.scene_column(name) for name in self.schema.scene_scalar_fields]
            + list(self.schema.aggregation_columns)
            + [
                self.schema.neighbour_column(index, name)
                for index in range(self.neighbour_count)
                for name in self.schema.neighbour_fields
            ]
        )

    def row(self, scene: ClutterPickScene) -> Dict[str, Any]:
        """
        :param scene: The attempt to flatten.
        :return: The attempt's values, keyed by column.
        :raises FlatTableSchemaMismatchError: If ``scene`` has another neighbour count.
        """
        if len(scene.neighbours) != self.neighbour_count:
            raise FlatTableSchemaMismatchError(
                [
                    self.schema.neighbour_column(index, name)
                    for index in range(
                        min(len(scene.neighbours), self.neighbour_count),
                        max(len(scene.neighbours), self.neighbour_count),
                    )
                    for name in self.schema.neighbour_fields
                ]
            )
        aggregations = ClutterPickSceneAggregations(instance=scene)
        scene_values = vars(scene)
        row = {
            self.schema.scene_column(name): scene_values[name]
            for name in self.schema.scene_scalar_fields
        }
        row.update(
            {
                self.schema.aggregation_column(statistic.__name__): statistic(
                    aggregations
                )
                for statistic in self.schema.aggregation_statistics
            }
        )
        for index, neighbour in enumerate(scene.neighbours):
            neighbour_values = vars(neighbour)
            row.update(
                {
                    self.schema.neighbour_column(index, name): neighbour_values[name]
                    for name in self.schema.neighbour_fields
                }
            )
        return row

    def dataframe(self, scenes: Iterable[ClutterPickScene]) -> pd.DataFrame:
        """
        :param scenes: The attempts to flatten.
        :return: One row per attempt, columns in :attr:`columns` order.
        """
        return pd.DataFrame([self.row(scene) for scene in scenes], columns=self.columns)
