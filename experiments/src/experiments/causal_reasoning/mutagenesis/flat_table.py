"""
Flattening molecules into one table, named the way EQL names the same attributes, so a
query built for the relational pipeline reads the flat table's columns unchanged.

A molecule has anywhere from 14 to 40 atoms and no canonical atom order, so a flat table
has to choose what to do with the parts. The three layouts are the choices a flat
learner has: keep only the molecule's own scalars, add the aggregation counts the
relational model derives from the parts, or unroll the parts into one block of columns
per position and pad the positions a molecule does not fill. Values are kept as they
are, an enum member stays a member, which is also how a fitted tree's leaves and a
query's conditions read them.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field, fields
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
    get_args,
    get_type_hints,
)

from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisMolecule,
    MutagenesisMoleculeAggregations,
)
from experiments.causal_reasoning.mutagenesis.exceptions import (
    FlatTableSchemaMismatchError,
)


class TableLayout(StrEnum):
    """
    What a flat table holds of a molecule besides its own scalars.
    """

    SCALARS = "scalars-only"
    """
    The molecule's own scalars and nothing of its parts.
    """

    PROPOSITIONAL = "propositional"
    """
    The scalars and the aggregation counts over the parts, the classic
    propositionalisation of a relational example.
    """

    UNROLLED = "unrolled"
    """
    The scalars, the counts, and every part's attributes under the part's position,
    padded where a molecule has fewer parts than the widest one.
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


class MoleculeView(StrEnum):
    """
    How much of a molecule a likelihood is taken over.
    """

    SCALARS = "scalars"
    """
    The molecule's own scalars.
    """

    SCALARS_AND_COUNTS = "scalars and counts"
    """
    The scalars and the aggregation counts over the parts.
    """

    WHOLE_MOLECULE = "whole molecule"
    """
    The scalars, the counts, and every atom and bond.
    """


class AbsentPart(StrEnum):
    """
    The symbol an unrolled column holds where a molecule has no part at that position.
    """

    ABSENT = "absent"


@dataclass(frozen=True)
class PartPadding:
    """
    What an unrolled column holds at a position the molecule has no part for, per kind
    of attribute.
    """

    symbol: AbsentPart = AbsentPart.ABSENT
    """
    For an enum attribute: a symbol of its own next to the enum's members.
    """

    integer: int = 0
    """
    For an integer attribute: a value no real part shows (every atom has at least one
    bond and an atom-type code of at least one).
    """

    real: float = 10.0
    """
    For a continuous attribute: a value outside every real one (partial charges lie
    within one unit of zero).
    """

    def value_for(self, attribute_type: Type) -> Any:
        """
        :param attribute_type: The attribute's type.
        :return: The padding value of that kind.
        """
        if issubclass(attribute_type, enum.Enum):
            return self.symbol
        if issubclass(attribute_type, float):
            return self.real
        return self.integer


@dataclass(frozen=True)
class PartAttribute:
    """
    One attribute of one of a molecule's exchangeable parts, as a query names it.
    """

    part_field: str
    """
    The exchangeable-part field of the molecule, ``atoms`` or ``bonds``.
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
class MoleculeSchema:
    """
    The attributes of a molecule as EQL names them: the molecule's own scalars, its
    aggregation counts, and each atom's or bond's attributes under the part's index.
    """

    @property
    def part_fields(self) -> Tuple[str, ...]:
        """
        The exchangeable-part fields of
        :class:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisMolecule`,
        which are the fields
        :class:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisMoleculeAggregations`
        aggregates over.
        """
        return tuple(MutagenesisMoleculeAggregations.aggregation_registry)

    def part_class(self, part_field: str) -> Type:
        """
        :param part_field: An exchangeable-part field.
        :return: The class of that field's parts.
        """
        [part_class] = get_args(get_type_hints(MutagenesisMolecule)[part_field])
        return part_class

    def part_attribute_types(self, part_field: str) -> Dict[str, Type]:
        """
        :param part_field: An exchangeable-part field.
        :return: The attributes of that field's parts and their types.
        """
        return get_type_hints(self.part_class(part_field))

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
            for statistic in MutagenesisMoleculeAggregations.aggregation_registry[
                part_field
            ]
        )

    @property
    def scalar_fields(self) -> Tuple[str, ...]:
        """
        The molecule's own scalar attributes.
        """
        return tuple(
            molecule_field.name
            for molecule_field in fields(MutagenesisMolecule)
            if molecule_field.name not in self.part_fields
        )

    def scalar_column(self, field_name: str) -> str:
        """
        :param field_name: A molecule scalar field.
        :return: The column, and EQL variable, name of that field.
        """
        return get_class_and_attribute_name(MutagenesisMolecule.__name__, field_name)

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
        :return: The part attribute it names, or ``None`` if it names a molecule-level
            variable.
        """
        pattern = re.compile(
            rf"^{re.escape(MutagenesisMolecule.__name__)}\."
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
            MutagenesisMoleculeAggregations.__name__, f"{statistic_name}()"
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
    Molecules as one row each, holding what the layout says of them.
    """

    layout: TableLayout = TableLayout.PROPOSITIONAL
    """
    What the rows hold besides the molecule's own scalars.
    """

    part_widths: Dict[str, int] = field(default_factory=dict)
    """
    Per part field, how many positions an unrolled row has; a molecule with more parts
    than that cannot be a row, one with fewer is padded.
    """

    padding: PartPadding = PartPadding()
    """
    What a position without a part holds.
    """

    schema: MoleculeSchema = field(default_factory=MoleculeSchema)
    """
    How the columns are named.
    """

    @classmethod
    def unrolled_for(
        cls,
        molecules: Sequence[MutagenesisMolecule],
        schema: MoleculeSchema = MoleculeSchema(),
    ) -> FlatTable:
        """
        :param molecules: The molecules the table has to hold.
        :param schema: How the columns are named.
        :return: An unrolled table wide enough for the largest of them.
        """
        return cls(
            layout=TableLayout.UNROLLED,
            part_widths={
                part_field: max(
                    len(vars(molecule)[part_field]) for molecule in molecules
                )
                for part_field in schema.part_fields
            },
            schema=schema,
        )

    @property
    def part_columns(self) -> List[str]:
        """
        The unrolled parts' columns, in position order.
        """
        return [
            self.schema.part_column(PartAttribute(part_field, index, attribute))
            for part_field in self.schema.part_fields
            for index in range(self.part_widths.get(part_field, 0))
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

    def columns_of(self, view: MoleculeView) -> Optional[List[str]]:
        """
        :param view: How much of a molecule to look at.
        :return: The columns holding that much, or ``None`` if the layout holds less.
        """
        if view is MoleculeView.SCALARS:
            return list(self.schema.scalar_columns)
        if view is MoleculeView.SCALARS_AND_COUNTS and self.layout.has_counts:
            return list(self.schema.scalar_columns) + list(
                self.schema.aggregation_columns
            )
        if view is MoleculeView.WHOLE_MOLECULE and self.layout.has_parts:
            return self.columns
        return None

    def fits(self, molecule: MutagenesisMolecule) -> bool:
        """
        :param molecule: A molecule.
        :return: Whether the table has a position for every one of its parts.
        """
        return not self._overflowing_columns(molecule)

    def _overflowing_columns(self, molecule: MutagenesisMolecule) -> List[str]:
        """
        :param molecule: A molecule.
        :return: The columns its parts past the table's width would need.
        """
        if not self.layout.has_parts:
            return []
        return [
            self.schema.part_column(PartAttribute(part_field, index, attribute))
            for part_field in self.schema.part_fields
            for index in range(
                self.part_widths[part_field], len(vars(molecule)[part_field])
            )
            for attribute in self.schema.part_attribute_types(part_field)
        ]

    def row(self, molecule: MutagenesisMolecule) -> Dict[str, Any]:
        """
        :param molecule: The molecule to flatten.
        :return: The molecule's values, keyed by column.
        :raises FlatTableSchemaMismatchError: If the molecule has more parts than the
            table has positions.
        """
        overflowing = self._overflowing_columns(molecule)
        if overflowing:
            raise FlatTableSchemaMismatchError(overflowing)
        molecule_values = vars(molecule)
        row = {
            self.schema.scalar_column(name): molecule_values[name]
            for name in self.schema.scalar_fields
        }
        if self.layout.has_counts:
            aggregations = MutagenesisMoleculeAggregations(instance=molecule)
            row.update(
                {
                    self.schema.aggregation_column(statistic.__name__): statistic(
                        aggregations
                    )
                    for statistic in self.schema.aggregation_statistics
                }
            )
        if self.layout.has_parts:
            for part_field in self.schema.part_fields:
                row.update(self._part_values(part_field, molecule_values[part_field]))
        return row

    def _part_values(self, part_field: str, parts: Sequence[Any]) -> Dict[str, Any]:
        """
        :param part_field: The exchangeable-part field the parts belong to.
        :param parts: The molecule's parts of that kind, in the molecule's order.
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

    def dataframe(self, molecules: Iterable[MutagenesisMolecule]) -> pd.DataFrame:
        """
        :param molecules: The molecules to flatten.
        :return: One row per molecule, columns in :attr:`columns` order.
        """
        return pd.DataFrame(
            [self.row(molecule) for molecule in molecules], columns=self.columns
        )
