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
    aggregation statistics.
    """

    example_attribute: str
    """
    The name of the instance attribute.
    """

    part_attribute: str
    """
    The name of the part attribute, as its part's table names it.
    """

    part: str
    """
    The exchangeable part the attribute belongs to.
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
    statistics, with a G² statistic whose null distribution comes from permuting the
    instance attribute across whole instances that share the statistics' values.
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
            for part, table in tables.part_tables.items()
            for dependence in self.dependences_of_part(tables.example_rows, part, table)
        ]

    def dependences_of_part(
        self, example_rows: pd.DataFrame, part: str, table: PartTrainingTable
    ) -> list[ExampleToPartDependence]:
        """
        :param example_rows: One row per instance.
        :param part: The name of the exchangeable part.
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
        strata = (
            self.strata_of(example_codes[statistics])
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
                self.permuted_within(values, strata, random_state)[positions]
                for _ in range(self.permutation_count)
            ]
            for part_attribute in part_attributes:
                part_values = part_codes[part_attribute].to_numpy()
                if len(np.unique(part_values)) < 2:
                    continue
                probability = self.permutation_probability(
                    values[positions], permutations, part_values, strata[positions]
                )
                if probability < self.significance_level:
                    dependences.append(
                        ExampleToPartDependence(
                            example_attribute, part_attribute, part, probability
                        )
                    )
        return dependences

    def permutation_probability(
        self,
        example_values: np.ndarray,
        permutations: list[np.ndarray],
        part_values: np.ndarray,
        strata: np.ndarray,
    ) -> float:
        """
        :param example_values: Per part row, its instance's level of the instance
            attribute.
        :param permutations: The same levels after each permutation of the instances.
        :param part_values: Per part row, its level of the part attribute.
        :param strata: Per part row, its instance's stratum.
        :return: The share of permutations, counting the observed levels as one, whose
            G² statistic is at least the observed one.
        """
        observed = self.g_squared(example_values, part_values, strata)
        exceeding = sum(
            self.g_squared(permuted, part_values, strata) >= observed
            for permuted in permutations
        )
        return (1 + exceeding) / (1 + len(permutations))

    @staticmethod
    def strata_of(codes: pd.DataFrame) -> np.ndarray:
        """
        :param codes: Integer columns.
        :return: One integer per row, the same for rows with the same values.
        """
        return pd.factorize(pd.MultiIndex.from_frame(codes))[0]

    @staticmethod
    def permuted_within(
        values: np.ndarray, strata: np.ndarray, random_state: np.random.Generator
    ) -> np.ndarray:
        """
        :param values: One value per instance.
        :param strata: One stratum per instance.
        :param random_state: Source of randomness.
        :return: The values shuffled among the instances of each stratum.
        """
        permuted = values.copy()
        for stratum in np.unique(strata):
            members = np.flatnonzero(strata == stratum)
            permuted[members] = values[random_state.permutation(members)]
        return permuted

    @staticmethod
    def g_squared(first: np.ndarray, second: np.ndarray, strata: np.ndarray) -> float:
        """
        :param first: Integer levels.
        :param second: Integer levels.
        :param strata: Integer strata.
        :return: The G² statistic of the two given the strata, summed over the strata.
        """
        statistic = 0.0
        for stratum in np.unique(strata):
            members = strata == stratum
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
