"""
Checking, on the training data, the assumption a relational circuit makes about its
causal structure: an instance's attributes reach its parts' attributes only through the
aggregation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        statistics = table.aggregation_names
        example_attributes = [
            column for column in example_rows.columns if column not in statistics
        ]
        part_attributes = [
            column for column in table.rows.columns if column not in statistics
        ]
        example_codes = self.discretizer.fit_transform(
            example_rows[example_attributes + statistics]
        )
        part_codes = self.discretizer.fit_transform(table.rows[part_attributes])
        positions = np.asarray(table.example_positions)
        groups = (
            self.group_of_each_row(example_codes[statistics])
            if statistics
            else np.zeros(len(example_rows), dtype=int)
        )
        random_state = np.random.default_rng(self.random_seed)
        dependences = []
        for example_attribute in example_attributes:
            values = example_codes[example_attribute].to_numpy()
            if len(np.unique(values)) < 2:
                continue
            permutations = [
                self.permuted_within_groups(values, groups, random_state)[positions]
                for _ in range(self.permutation_count)
            ]
            for part_attribute in part_attributes:
                part_values = part_codes[part_attribute].to_numpy()
                if len(np.unique(part_values)) < 2:
                    continue
                probability = self.permutation_probability(
                    values[positions], permutations, part_values, groups[positions]
                )
                if probability < self.significance_level:
                    dependences.append(
                        ExampleToPartDependence(
                            example_attribute,
                            part_attribute,
                            part_collection,
                            probability,
                        )
                    )
        return dependences

    def permutation_probability(
        self,
        example_values: np.ndarray,
        permutations: list[np.ndarray],
        part_values: np.ndarray,
        groups: np.ndarray,
    ) -> float:
        """
        :param example_values: Per part row, its instance's level of the instance
            attribute.
        :param permutations: The same levels after each permutation of the instances.
        :param part_values: Per part row, its level of the part attribute.
        :param groups: Per part row, the group of instances with the same aggregation
            statistics its instance belongs to.
        :return: The share of permutations, counting the observed levels as one, whose
            G² statistic is at least the observed one.
        """
        observed = self.likelihood_ratio_statistic(example_values, part_values, groups)
        exceeding = sum(
            self.likelihood_ratio_statistic(permuted, part_values, groups) >= observed
            for permuted in permutations
        )
        return (1 + exceeding) / (1 + len(permutations))

    @staticmethod
    def group_of_each_row(codes: pd.DataFrame) -> np.ndarray:
        """
        :param codes: Integer columns.
        :return: One integer per row, the same for rows with the same values.
        """
        return pd.factorize(pd.MultiIndex.from_frame(codes))[0]

    @staticmethod
    def permuted_within_groups(
        values: np.ndarray, groups: np.ndarray, random_state: np.random.Generator
    ) -> np.ndarray:
        """
        :param values: One value per instance.
        :param groups: Per instance, the group of instances with the same aggregation
            statistics it belongs to.
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
        statistic = 0.0
        for group in np.unique(groups):
            members = groups == group
            observed = pd.crosstab(first[members], second[members]).to_numpy(float)
            expected = (
                observed.sum(axis=1, keepdims=True)
                * observed.sum(axis=0, keepdims=True)
                / observed.sum()
            )
            nonzero = observed > 0
            statistic += 2 * float(
                (
                    observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
                ).sum()
            )
        return statistic
