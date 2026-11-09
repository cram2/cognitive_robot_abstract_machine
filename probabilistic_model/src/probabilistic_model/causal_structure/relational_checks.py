"""
Checking, on the training data, the assumption a relational circuit makes about its
causal structure: an instance's attributes reach its parts' attributes only through the
aggregation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from probabilistic_model.causal_structure.discretization import Discretizer
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    PartTrainingTable,
    RelationalTrainingTables,
)

# %% one dependence the assumption does not allow


@dataclass(frozen=True)
class ExampleToPartDependence:
    """
    An instance attribute that the data says depends on a part attribute even given the
    aggregation statistics. A part is one of the objects an instance holds in a
    one-to-many attribute, such as one of the objects of a room.
    """

    example_attribute: str
    """
    The name of the instance attribute.
    """

    part_attribute: str
    """
    The name of the part attribute, as its part's table names it.
    """

    part_collection: str
    """
    The name of the instance's one-to-many attribute that holds the parts, such as
    ``objects``.
    """

    probability: float
    """
    The probability, were the attributes independent given the statistics, of a
    dependence at least as strong as the one observed, estimated by permutation.
    """


# %% the tables the check reads


@dataclass
class DiscretizedPartTables:
    """
    The instance table and one part's table of a relational circuit, cut into levels.
    """

    example_levels: pd.DataFrame
    """
    One row of levels per instance, over its attributes and the part's aggregation
    statistics.
    """

    part_levels: pd.DataFrame
    """
    One row of levels per part, over the part's own attributes.
    """

    example_positions: np.ndarray
    """
    Per part row, the position of its instance among the instances.
    """

    aggregation_names: list[str]
    """
    The names of the part's aggregation statistics.
    """

    @property
    def varying_example_attributes(self) -> list[str]:
        """
        :return: The instance attributes, other than the aggregation statistics, that
            take more than one level.
        """
        return [
            column
            for column in self.example_levels.columns
            if column not in self.aggregation_names
            and self.example_levels[column].nunique() > 1
        ]

    @property
    def varying_part_attributes(self) -> list[str]:
        """
        :return: The part attributes that take more than one level.
        """
        return [
            column
            for column in self.part_levels.columns
            if self.part_levels[column].nunique() > 1
        ]

    def group_of_each_instance(self, example_attributes: list[str]) -> np.ndarray:
        """
        :param example_attributes: Names of instance columns.
        :return: One integer per instance, the same for instances whose levels agree on
            those columns.
        """
        if not example_attributes:
            return np.zeros(len(self.example_levels), dtype=int)
        return pd.factorize(
            pd.MultiIndex.from_frame(self.example_levels[example_attributes])
        )[0]

    def group_of_each_part(
        self, instance_groups: np.ndarray, part_attributes: list[str]
    ) -> np.ndarray:
        """
        :param instance_groups: One group per instance.
        :param part_attributes: Names of part columns.
        :return: One integer per part, the same for parts whose instances share a group
            and whose levels agree on those columns.
        """
        return pd.factorize(
            pd.MultiIndex.from_arrays(
                [instance_groups[self.example_positions]]
                + [self.part_levels[column].to_numpy() for column in part_attributes]
            )
        )[0]


@dataclass
class ConditioningAttributes:
    """
    The attributes a test conditions on besides the aggregation statistics.
    """

    example_attributes: list[str] = field(default_factory=list)
    """
    The names of the instance attributes.
    """

    part_attributes: list[str] = field(default_factory=list)
    """
    The names of the part attributes.
    """


# %% the check


@dataclass
class ExampleToPartCheck:
    """
    Tests every instance attribute against every part attribute given the aggregation
    statistics, with a likelihood-ratio statistic (G²) whose null distribution comes from
    permuting the instance attribute among the instances with the same aggregation
    statistics.
    """

    significance_level: float = 0.05
    """
    The level below which a :attr:`ExampleToPartDependence.probability` rejects
    independence.
    """

    permutation_count: int = 199
    """
    How many permutations the null distribution is drawn from.
    """

    discretizer: Discretizer = field(default_factory=Discretizer)
    """
    How the columns are cut into levels.
    """

    random_seed: int = 0
    """
    Seed of the permutations.
    """

    def dependences(
        self, tables: RelationalTrainingTables
    ) -> list[ExampleToPartDependence]:
        """
        :param tables: The tables a relational circuit is fitted on.
        :return: The dependences between an instance attribute and a part attribute,
            given the aggregation statistics, in every part's table.
        """
        return [
            dependence
            for part_collection, table in tables.part_tables.items()
            for dependence in self.dependences_of_part(
                tables.example_rows, part_collection, table
            )
        ]

    def dependences_of_part(
        self, example_rows: pd.DataFrame, part_collection: str, table: PartTrainingTable
    ) -> list[ExampleToPartDependence]:
        """
        :param example_rows: One row per instance.
        :param part_collection: The name of the instance's attribute that holds the parts.
        :param table: The part's training table.
        :return: The dependences between an instance attribute and one of the part's
            attributes, given the part's aggregation statistics.
        """
        tables = self.discretized(example_rows, table)
        random_state = np.random.default_rng(self.random_seed)
        return [
            ExampleToPartDependence(
                example_attribute, part_attribute, part_collection, probability
            )
            for example_attribute in tables.varying_example_attributes
            for part_attribute in tables.varying_part_attributes
            if (
                probability := self.permutation_probability(
                    tables,
                    example_attribute,
                    part_attribute,
                    ConditioningAttributes(),
                    random_state,
                )
            )
            < self.significance_level
        ]

    def dependences_given_another_attribute(
        self,
        example_rows: pd.DataFrame,
        part_collection: str,
        table: PartTrainingTable,
        adjacent_part_attributes: dict[str, set[str]],
    ) -> list[ExampleToPartDependence]:
        """
        :param example_rows: One row per instance.
        :param part_collection: The name of the instance's attribute that holds the parts.
        :param table: The part's training table.
        :param adjacent_part_attributes: Per part attribute, the part attributes the
            part's own graph joins it to.
        :return: The dependences of :meth:`dependences_of_part` that remain given any one
            other attribute the instance attribute could reach the part attribute
            through, each with the largest probability it was tested with.
        """
        dependences = self.dependences_of_part(example_rows, part_collection, table)
        tables = self.discretized(example_rows, table)
        random_state = np.random.default_rng(self.random_seed)
        remaining = []
        for dependence in dependences:
            probability = max(
                [dependence.probability]
                + [
                    self.permutation_probability(
                        tables,
                        dependence.example_attribute,
                        dependence.part_attribute,
                        conditioning,
                        random_state,
                    )
                    for conditioning in self.other_paths(
                        dependence,
                        dependences,
                        adjacent_part_attributes.get(dependence.part_attribute, set()),
                    )
                ]
            )
            if probability < self.significance_level:
                remaining.append(replace(dependence, probability=probability))
        return remaining

    @staticmethod
    def other_paths(
        dependence: ExampleToPartDependence,
        dependences: list[ExampleToPartDependence],
        adjacent_part_attributes: set[str],
    ) -> list[ConditioningAttributes]:
        """
        :param dependence: A dependence between an instance and a part attribute.
        :param dependences: Every dependence found in the part's table.
        :param adjacent_part_attributes: The part attributes the part's own graph joins
            the dependence's part attribute to.
        :return: One conditioning per attribute the dependence could run through: another
            instance attribute that also depends on the part attribute, or an adjacent
            part attribute the instance attribute also depends on.
        """
        return [
            ConditioningAttributes(example_attributes=[other.example_attribute])
            for other in dependences
            if other.part_attribute == dependence.part_attribute
            and other.example_attribute != dependence.example_attribute
        ] + [
            ConditioningAttributes(part_attributes=[other.part_attribute])
            for other in dependences
            if other.example_attribute == dependence.example_attribute
            and other.part_attribute in adjacent_part_attributes
        ]

    def discretized(
        self, example_rows: pd.DataFrame, table: PartTrainingTable
    ) -> DiscretizedPartTables:
        """
        :param example_rows: One row per instance.
        :param table: One part's training table.
        :return: Both tables cut into levels by :attr:`discretizer`.
        """
        statistics = table.aggregation_names
        example_attributes = [
            column for column in example_rows.columns if column not in statistics
        ]
        part_attributes = [
            column for column in table.rows.columns if column not in statistics
        ]
        return DiscretizedPartTables(
            example_levels=self.discretizer.fit_transform(
                example_rows[example_attributes + statistics]
            ),
            part_levels=self.discretizer.fit_transform(table.rows[part_attributes]),
            example_positions=np.asarray(table.example_positions),
            aggregation_names=statistics,
        )

    def permutation_probability(
        self,
        tables: DiscretizedPartTables,
        example_attribute: str,
        part_attribute: str,
        conditioning: ConditioningAttributes,
        random_state: np.random.Generator,
    ) -> float:
        """
        :param tables: The discretized tables.
        :param example_attribute: The name of the instance attribute.
        :param part_attribute: The name of the part attribute.
        :param conditioning: The attributes conditioned on besides the aggregation
            statistics.
        :param random_state: Source of the permutations.
        :return: The share of permutations of the instance attribute, counting the
            observed levels as one, whose G² statistic is at least the observed one.
        """
        instance_groups = tables.group_of_each_instance(
            tables.aggregation_names + conditioning.example_attributes
        )
        part_groups = tables.group_of_each_part(
            instance_groups, conditioning.part_attributes
        )
        values = tables.example_levels[example_attribute].to_numpy()
        part_values = tables.part_levels[part_attribute].to_numpy()
        positions = tables.example_positions
        observed = self.likelihood_ratio_statistic(
            values[positions], part_values, part_groups
        )
        exceeding = sum(
            self.likelihood_ratio_statistic(
                self.permuted_within_groups(values, instance_groups, random_state)[
                    positions
                ],
                part_values,
                part_groups,
            )
            >= observed
            for _ in range(self.permutation_count)
        )
        return (1 + exceeding) / (1 + self.permutation_count)

    @staticmethod
    def permuted_within_groups(
        values: np.ndarray, groups: np.ndarray, random_state: np.random.Generator
    ) -> np.ndarray:
        """
        :param values: One value per instance.
        :param groups: Per instance, the group of instances it is permuted among.
        :param random_state: Source of randomness.
        :return: The values shuffled among the instances of each group.
        """
        permuted = values.copy()
        for group in np.unique(groups):
            members = np.flatnonzero(groups == group)
            permuted[members] = values[random_state.permutation(members)]
        return permuted

    @staticmethod
    def likelihood_ratio_statistic(
        first: np.ndarray, second: np.ndarray, groups: np.ndarray
    ) -> float:
        """
        :param first: Integer levels.
        :param second: Integer levels.
        :param groups: Integer groups the rows fall into.
        :return: The likelihood-ratio statistic (G²) of the two within each group, summed
            over the groups.
        """
        shape = (groups.max() + 1, first.max() + 1, second.max() + 1)
        observed = np.bincount(
            np.ravel_multi_index((groups, first, second), shape),
            minlength=int(np.prod(shape)),
        ).reshape(shape)
        expected = (
            observed.sum(axis=2, keepdims=True)
            * observed.sum(axis=1, keepdims=True)
            / np.maximum(observed.sum(axis=(1, 2), keepdims=True), 1)
        )
        nonzero = observed > 0
        return 2 * float(
            (observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])).sum()
        )
