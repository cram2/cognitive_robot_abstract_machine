"""
Flattening examples into one table, named the way EQL names the same attributes, so a
query built for the relational pipeline reads the flat table's columns unchanged.

An example has some number of exchangeable parts and no canonical order over them, so a
flat table has to choose what to do with the parts. The three layouts are the choices a
flat learner has: keep only the example's own scalars, add the aggregation counts the
relational model derives from the parts, or unroll the parts into one block of columns
per position and pad the positions an example does not fill. Values are kept as they
are, an enum member stays a member, which is also how a fitted tree's leaves and a
query's conditions read them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd
from krrood.utils import get_class_and_attribute_name
from typing_extensions import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from experiments.causal_reasoning.comparison.domain import (
    ExampleView,
    PartPadding,
    RelationalDomain,
)
from experiments.causal_reasoning.comparison.exceptions import (
    FlatTableSchemaMismatchError,
)


class TableLayout(StrEnum):
    """
    What a flat table holds of an example besides its own scalars.
    """

    SCALARS = "scalars-only"
    """
    The example's own scalars and nothing of its parts.
    """

    PROPOSITIONAL = "propositional"
    """
    The scalars and the aggregation counts over the parts, the classic
    propositionalisation of a relational example.
    """

    UNROLLED = "unrolled"
    """
    The scalars, the counts, and every part's attributes under the part's position,
    padded where an example has fewer parts than the widest one.
    """

    @property
    def has_counts(self) -> bool:
        """
        Whether the layout carries the aggregation counts.
        """
        return self is not TableLayout.SCALARS

    @property
    def has_parts(self) -> bool:
        """
        Whether the layout carries the parts' own attributes.
        """
        return self is TableLayout.UNROLLED


@dataclass(frozen=True)
class PartAttribute:
    """
    One attribute of one of an example's exchangeable parts, as a query names it.
    """

    part_field: str
    """
    The exchangeable-part field of the example, ``objects`` or ``viewpoints``.
    """

    index: int
    """
    The part's position in that field's list.
    """

    attribute: str
    """
    The part's attribute.
    """


@dataclass(frozen=True)
class Schema:
    """
    The attributes of an example as EQL names them: the example's own scalars, its
    aggregation counts, and each part's attributes under the part's index.
    """

    domain: RelationalDomain
    """
    The example and its parts.
    """

    @property
    def part_fields(self) -> Tuple[str, ...]:
        """
        The exchangeable-part fields of the example.
        """
        return self.domain.part_fields

    def part_class(self, part_field: str) -> Type:
        """
        :param part_field: An exchangeable-part field.
        :return: The class of that field's parts.
        """
        return self.domain.part_class(part_field)

    def part_attribute_types(self, part_field: str) -> Dict[str, Type]:
        """
        :param part_field: An exchangeable-part field.
        :return: The attributes of that field's parts and their types.
        """
        return self.domain.part_attribute_types(part_field)

    @property
    def aggregation_statistics(self) -> Tuple[Callable[..., Any], ...]:
        """
        The aggregation statistics over every part field; a flat table carries them as
        columns, so it holds the same summaries the relational model derives from its
        parts.
        """
        return tuple(
            statistic
            for part_field in self.part_fields
            for statistic in self.domain.aggregation_class.aggregation_registry[
                part_field
            ]
        )

    @property
    def scalar_fields(self) -> Tuple[str, ...]:
        """
        The example's own scalar attributes.
        """
        return self.domain.scalar_fields

    def scalar_column(self, field_name: str) -> str:
        """
        :param field_name: An example scalar field.
        :return: The column, and EQL variable, name of that field.
        """
        return get_class_and_attribute_name(
            self.domain.example_class.__name__, field_name
        )

    @property
    def scalar_columns(self) -> Tuple[str, ...]:
        """
        The column names of every scalar field.
        """
        return tuple(self.scalar_column(name) for name in self.scalar_fields)

    def part_column(self, part: PartAttribute) -> str:
        """
        :param part: One part's attribute.
        :return: The column, and EQL variable, name of that attribute.
        """
        return self.scalar_column(f"{part.part_field}[{part.index}].{part.attribute}")

    def part_attribute(self, variable_name: str) -> Optional[PartAttribute]:
        """
        :param variable_name: A variable name, as EQL names it.
        :return: The part attribute it names, or ``None`` if it names an example-level
            variable.
        """
        pattern = re.compile(
            rf"^{re.escape(self.domain.example_class.__name__)}\."
            rf"({'|'.join(map(re.escape, self.part_fields))})\[(\d+)\]\.(\w+)$"
        )
        match = pattern.match(variable_name)
        if match is None:
            return None
        return PartAttribute(match.group(1), int(match.group(2)), match.group(3))

    def aggregation_column(self, statistic_name: str) -> str:
        """
        :param statistic_name: The name of one of the :attr:`aggregation_statistics`.
        :return: The column, and EQL variable, name of that statistic, which grounding
            names by its class and its call.
        """
        return get_class_and_attribute_name(
            self.domain.aggregation_class.__name__, f"{statistic_name}()"
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
    Examples as one row each, holding what the layout says of them.
    """

    schema: Schema
    """
    How the columns are named.
    """

    layout: TableLayout = TableLayout.PROPOSITIONAL
    """
    What the rows hold besides the example's own scalars.
    """

    part_widths: Dict[str, int] = field(default_factory=dict)
    """
    Per unrolled part field, how many positions a row has; an example with more parts of
    that kind cannot be a row, one with fewer is padded.

    A part field left out is not unrolled at all, which is how a table holds an
    example's viewpoints by position and leaves its objects to a model that treats them
    as exchangeable.
    """

    @property
    def padding(self) -> PartPadding:
        """
        What a position without a part holds.
        """
        return self.schema.domain.padding

    @classmethod
    def unrolled_for(
        cls,
        schema: Schema,
        examples: Sequence[Any],
        part_fields: Optional[Sequence[str]] = None,
    ) -> FlatTable:
        """
        :param schema: How the columns are named.
        :param examples: The examples the table has to hold.
        :param part_fields: The part fields to unroll; every one if not given.
        :return: An unrolled table wide enough for the largest of them.
        """
        return cls(
            schema=schema,
            layout=TableLayout.UNROLLED,
            part_widths={
                part_field: max(len(vars(example)[part_field]) for example in examples)
                for part_field in (part_fields or schema.part_fields)
            },
        )

    @property
    def unrolled_fields(self) -> List[str]:
        """
        The part fields the table holds by position, in the schema's order.
        """
        if not self.layout.has_parts:
            return []
        return [
            part_field
            for part_field in self.schema.part_fields
            if part_field in self.part_widths
        ]

    @property
    def part_columns(self) -> List[str]:
        """
        The unrolled parts' columns, in position order.
        """
        return [
            self.schema.part_column(PartAttribute(part_field, index, attribute))
            for part_field in self.unrolled_fields
            for index in range(self.part_widths[part_field])
            for attribute in self.schema.part_attribute_types(part_field)
        ]

    @property
    def columns(self) -> List[str]:
        """
        The table's columns, in order.
        """
        columns = list(self.schema.scalar_columns)
        if self.layout.has_counts:
            columns += list(self.schema.aggregation_columns)
        if self.layout.has_parts:
            columns += self.part_columns
        return columns

    def columns_of(self, view: ExampleView) -> Optional[List[str]]:
        """
        :param view: How much of an example to look at.
        :return: The columns holding that much, or ``None`` if the layout holds less.
        """
        if view is ExampleView.SCALARS:
            return list(self.schema.scalar_columns)
        if view is ExampleView.SCALARS_AND_COUNTS and self.layout.has_counts:
            return list(self.schema.scalar_columns) + list(
                self.schema.aggregation_columns
            )
        if view is ExampleView.WHOLE and self.layout.has_parts:
            return self.columns
        return None

    def fits(self, example: Any) -> bool:
        """
        :param example: An example.
        :return: Whether the table has a position for every one of its parts.
        """
        return not self._overflowing_columns(example)

    def _overflowing_columns(self, example: Any) -> List[str]:
        """
        :param example: An example.
        :return: The columns its parts past the table's width would need.
        """
        return [
            self.schema.part_column(PartAttribute(part_field, index, attribute))
            for part_field in self.unrolled_fields
            for index in range(
                self.part_widths[part_field], len(vars(example)[part_field])
            )
            for attribute in self.schema.part_attribute_types(part_field)
        ]

    def row(self, example: Any) -> Dict[str, Any]:
        """
        :param example: The example to flatten.
        :return: The example's values, keyed by column.
        :raises FlatTableSchemaMismatchError: If the example has more parts than the table
            has positions.
        """
        overflowing = self._overflowing_columns(example)
        if overflowing:
            raise FlatTableSchemaMismatchError(overflowing)
        example_values = vars(example)
        row = {
            self.schema.scalar_column(name): example_values[name]
            for name in self.schema.scalar_fields
        }
        if self.layout.has_counts:
            aggregations = self.schema.domain.aggregation_class(instance=example)
            row.update(
                {
                    self.schema.aggregation_column(statistic.__name__): statistic(
                        aggregations
                    )
                    for statistic in self.schema.aggregation_statistics
                }
            )
        for part_field in self.unrolled_fields:
            row.update(self._part_values(part_field, example_values[part_field]))
        return row

    def _part_values(self, part_field: str, parts: Sequence[Any]) -> Dict[str, Any]:
        """
        :param part_field: The exchangeable-part field the parts belong to.
        :param parts: The example's parts of that kind, in the example's order.
        :return: Every position's attribute values, padded past the last part.
        """
        attribute_types = self.schema.part_attribute_types(part_field)
        values = {}
        for index in range(self.part_widths[part_field]):
            part_values = vars(parts[index]) if index < len(parts) else None
            for attribute, attribute_type in attribute_types.items():
                column = self.schema.part_column(
                    PartAttribute(part_field, index, attribute)
                )
                values[column] = (
                    self.padding.value_for(attribute_type)
                    if part_values is None
                    else part_values[attribute]
                )
        return values

    def dataframe(self, examples: Iterable[Any]) -> pd.DataFrame:
        """
        :param examples: The examples to flatten.
        :return: One row per example, columns in :attr:`columns` order.
        """
        return pd.DataFrame(
            [self.row(example) for example in examples], columns=self.columns
        )
