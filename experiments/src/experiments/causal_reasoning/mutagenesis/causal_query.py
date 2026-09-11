"""
Registering the Mutagenesis dataset's branching-atom count as a cause of
mutagenicity and running backdoor adjustment on the grounded circuit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from krrood.entity_query_language.factories import a, cause, confounder, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.product_algebra import SimpleEvent
from typing_extensions import List

from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisBond,
    MutagenesisMolecule,
    MutagenesisMoleculeAggregations,
)
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)


@dataclass(frozen=True)
class BranchingAtomCountCausalEffect:
    """
    The naive and backdoor-adjusted probability of mutagenicity at one branching-atom-
    count value.
    """

    branching_atom_count: int
    """
    The branching-atom-count value this row reports on.
    """

    region_probability: float
    """
    ``P(branching_atom_count)`` under the fitted circuit -- the share of the training
    population this value accounts for.
    """

    naive_probability_mutagenic: float
    """
    ``P(mutagenic = True | branching_atom_count)``, read directly off the grounded
    circuit with no adjustment.
    """

    adjusted_probability_mutagenic: float
    """
    ``P(mutagenic = True | do(branching_atom_count))``, backdoor-adjusted for the
    ``ind1`` structural indicator.
    """


@dataclass(frozen=True)
class ConfusionMatrix:
    """
    Counts of a binary classifier's predictions against actual outcomes.
    """

    true_positive: int
    """
    Predicted positive, actually positive.
    """

    false_positive: int
    """
    Predicted positive, actually negative.
    """

    true_negative: int
    """
    Predicted negative, actually negative.
    """

    false_negative: int
    """
    Predicted negative, actually positive.
    """

    @property
    def total(self) -> int:
        """
        Total number of classified instances.
        """
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def accuracy(self) -> float:
        """
        Fraction of instances the classifier labeled correctly.
        """
        return (self.true_positive + self.true_negative) / self.total


@dataclass
class MutagenesisCausalQueryResult:
    """
    Everything a run of :meth:`BranchingAtomCountCausalQuery.run` produced.
    """

    training_molecule_count: int
    """
    Number of molecules the circuit was fitted on.
    """

    atom_count: int
    """
    Number of atoms the grounding query specified.
    """

    bond_count: int
    """
    Number of bonds the grounding query specified.
    """

    support_determinism_verified: bool
    """
    Whether verify_support_determinism passed on the grounded circuit.
    """

    effects: List[BranchingAtomCountCausalEffect] = field(default_factory=list)
    """
    One row per branching-atom-count value the grounded circuit's support covers.
    """


@dataclass
class BranchingAtomCountCausalQuery:
    """
    Fits a relational circuit on Mutagenesis molecules, marks branching-atom count as
    a candidate cause of mutagenicity in the query itself, and compares naive
    conditioning against backdoor adjustment for the structural indicator ``ind1``.
    """

    def run(
        self,
        training_molecules: List[MutagenesisMolecule],
        atom_count: int = 2,
        bond_count: int = 1,
        random_seed: int = 0,
        monte_carlo_sample_count: int = 2000,
    ) -> MutagenesisCausalQueryResult:
        """
        Fit a relational circuit, mark branching-atom count as the query's cause and
        mutagenicity as its effect, and compare naive conditioning against backdoor
        adjustment for the structural indicator ``ind1``.

        Fits the class circuit stratified by branching-atom count, via
        :meth:`~probabilistic_model.probabilistic_circuit.relational.causal.RelationalCausalCircuit.fit`:
        a plain, unconstrained fit gives no guarantee that training rows sharing a
        branching-atom-count value end up under one circuit branch, and
        `CausalCircuit.verify_support_determinism` rejects the registration when they
        do not (two branches would then each claim the same value). Stratifying
        partitions the training dataframe by that exact value before fitting, so every
        value's rows share one branch by construction, on the full dataset without
        subsampling it down to one row per value.

        The query itself, not a separate registration call, declares what to search
        for: ``branching_atom_count=cause`` and ``.causes_effect(mutagenic=True)``
        mark the candidate cause and the effect right where the rest of the query is
        built, and :class:`~krrood.parametrization.model_registries.RelationalCircuitRegistry`
        reads those markers to ground, register and verify the resulting
        ``CausalCircuit`` in one step.

        :param training_molecules: Molecules to fit the circuit on.
        :param atom_count: Number of atoms the grounding query specifies; every atom's
            element, atom type, charge and bond count are left unspecified, so
            grounding must retain every atom-level aggregation statistic rather than
            integrating it out.
        :param bond_count: Number of bonds the grounding query specifies; every bond's
            type is left unspecified.
        :param random_seed: Seed applied to the global NumPy random state before
            grounding, which draws the Monte-Carlo samples that retain branching-atom
            count. Fixing it keeps the result reproducible across runs.
        :param monte_carlo_sample_count: Number of Monte-Carlo samples grounding draws
            when retaining branching-atom count. 2000 comfortably covers every value
            observed in the 188-molecule training population (see
            ``causal_query_results.md``).
        :return: The fitted result, including one causal-effect row per branching-
            atom-count value the grounded circuit's support covers.
        """
        model = RelationalProbabilisticCircuit(MutagenesisMolecule)
        model.monte_carlo_sample_count = monte_carlo_sample_count
        branching_atom_count_variable_expression = variable(
            MutagenesisMoleculeAggregations
        ).branching_atom_count()
        RelationalCausalCircuit().fit(
            model,
            [to_dao(molecule) for molecule in training_molecules],
            stratify_by=branching_atom_count_variable_expression._name_,
        )

        query = self._build_query(atom_count, bond_count)
        registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)

        np.random.seed(random_seed)
        causal_circuit = registry.get_model(UnderspecifiedParameters(query))

        [branching_atom_count_variable] = causal_circuit.causal_variables
        [mutagenic_variable] = causal_circuit.effect_variables
        indicator_1_variable = RelationalCausalCircuit.resolve_variable(
            causal_circuit.probabilistic_circuit, "indicator_1"
        )

        naive_circuit = causal_circuit.backdoor_adjustment(
            branching_atom_count_variable, mutagenic_variable
        )
        adjusted_circuit = causal_circuit.backdoor_adjustment(
            branching_atom_count_variable,
            mutagenic_variable,
            adjustment_variables=[indicator_1_variable],
        )

        branching_atom_count_regions = (
            causal_circuit._extract_disjoint_regions_for_variable(
                branching_atom_count_variable
            )
        )
        regions_by_value = {
            int(
                region.event.simple_sets[0][branching_atom_count_variable]
                .simple_sets[0]
                .lower
            ): region
            for region in branching_atom_count_regions
        }

        effects = [
            BranchingAtomCountCausalEffect(
                branching_atom_count=value,
                region_probability=region.probability,
                naive_probability_mutagenic=self._probability_mutagenic_at(
                    naive_circuit,
                    branching_atom_count_variable,
                    mutagenic_variable,
                    value,
                ),
                adjusted_probability_mutagenic=self._probability_mutagenic_at(
                    adjusted_circuit,
                    branching_atom_count_variable,
                    mutagenic_variable,
                    value,
                ),
            )
            for value, region in sorted(regions_by_value.items())
        ]

        return MutagenesisCausalQueryResult(
            training_molecule_count=len(training_molecules),
            atom_count=atom_count,
            bond_count=bond_count,
            support_determinism_verified=True,
            effects=effects,
        )

    @staticmethod
    def classify_by_branching_atom_count_and_indicator_1(
        molecules: List[MutagenesisMolecule],
    ) -> ConfusionMatrix:
        """
        Classify each molecule as mutagenic exactly when most molecules sharing its
        (branching-atom count, ``ind1``) pair are mutagenic, and tally the result
        against the molecule's actual label.

        This is not what the causal query is for -- backdoor adjustment answers a
        different question than classification does -- but it gives a reference point
        for how much signal these two fields carry about mutagenicity together. The
        majority vote is read directly off ``molecules`` themselves, not off a fitted
        circuit, and evaluated on the same molecules it was computed from.

        :param molecules: Molecules to classify, and to compute the per-pair majority
            vote from.
        :return: The resulting confusion matrix.
        """
        labels_by_key = defaultdict(list)
        for molecule in molecules:
            labels_by_key[
                BranchingAtomCountCausalQuery._classification_key(molecule)
            ].append(molecule.mutagenic)
        predicted_mutagenic_by_key = {
            key: labels.count(True) > labels.count(False)
            for key, labels in labels_by_key.items()
        }

        true_positive = false_positive = true_negative = false_negative = 0
        for molecule in molecules:
            key = BranchingAtomCountCausalQuery._classification_key(molecule)
            predicted_mutagenic = predicted_mutagenic_by_key[key]
            if predicted_mutagenic and molecule.mutagenic:
                true_positive += 1
            elif predicted_mutagenic and not molecule.mutagenic:
                false_positive += 1
            elif not predicted_mutagenic and molecule.mutagenic:
                false_negative += 1
            else:
                true_negative += 1
        return ConfusionMatrix(
            true_positive, false_positive, true_negative, false_negative
        )

    @staticmethod
    def _classification_key(molecule: MutagenesisMolecule) -> tuple[int, bool]:
        """
        :param molecule: The molecule to key.
        :return: The (branching-atom count, ``ind1``) pair
            :meth:`classify_by_branching_atom_count_and_indicator_1` groups by.
        """
        return (
            MutagenesisMoleculeAggregations(instance=molecule).branching_atom_count(),
            molecule.indicator_1,
        )

    @staticmethod
    def _build_query(atom_count: int, bond_count: int):
        """
        Build a molecule query marking branching-atom count as the cause, ``ind1`` as
        a confounder to adjust for, and mutagenicity as the effect. Every atom's
        element, atom type, charge and bond count are left unspecified, and every
        bond's type is left unspecified, so grounding must retain branching-atom
        count as an undetermined latent rather than integrating it out.

        :param atom_count: Number of atoms the query specifies.
        :param bond_count: Number of bonds the query specifies.
        :return: The resolved query, with its cause and effect marked.
        """
        query = a(MutagenesisMolecule)(
            indicator_1=confounder,
            logp=...,
            lumo=...,
            mutagenic=...,
            branching_atom_count=cause,
            atoms=[
                a(MutagenesisAtom)(
                    element=..., atom_type=..., charge=..., bond_count=...
                )
                for _ in range(atom_count)
            ],
            bonds=[a(MutagenesisBond)(bond_type=...) for _ in range(bond_count)],
        )
        query.causes_effect(query.variable.mutagenic == True)
        return query

    @staticmethod
    def _probability_mutagenic_at(
        interventional_circuit,
        branching_atom_count_variable,
        mutagenic_variable,
        branching_atom_count_value: int,
    ) -> float:
        """
        Read ``P(mutagenic = True)`` off ``interventional_circuit`` at one
        branching-atom-count value.

        :param interventional_circuit: A joint circuit over (branching-atom count,
            mutagenic), as returned by
            :meth:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit.backdoor_adjustment`.
        :param branching_atom_count_variable: The branching-atom-count Variable.
        :param mutagenic_variable: The mutagenicity Variable.
        :param branching_atom_count_value: The branching-atom-count value to truncate
            to.
        :return: The truncated circuit's probability that ``mutagenic`` is ``True``.
        """
        event = SimpleEvent.from_data(
            {branching_atom_count_variable: float(branching_atom_count_value)}
        ).as_composite_set()
        truncated_circuit, _ = interventional_circuit.truncated(
            event.fill_missing_variables_pure(interventional_circuit.variables)
        )
        true_event = (
            SimpleEvent.from_data({mutagenic_variable: True})
            .as_composite_set()
            .fill_missing_variables_pure(truncated_circuit.variables)
        )
        return float(truncated_circuit.probability(true_event))
