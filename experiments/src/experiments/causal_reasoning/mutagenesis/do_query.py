"""
Asking the Mutagenesis circuit what a molecule's properties cause.

Each question marks one cause and one effect in the query itself, grounds the relational
circuit against it, and reads the effect off every region of the cause twice: once by
conditioning alone and once with backdoor adjustment, so the two can be set side by
side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum

import numpy as np
from krrood.entity_query_language.factories import a, cause, confounder, variable
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from typing_extensions import Any, Dict, List, Tuple

from experiments.causal_reasoning.do_query import CauseRegion, DoQueryAnswer
from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisBond,
    MutagenesisElement,
    MutagenesisMolecule,
    MutagenesisMoleculeAggregations,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

# %% what a molecule offers as a cause


class MoleculeAttribute(StrEnum):
    """
    A molecule attribute a question names as its cause or adjusts for.
    """

    INDICATOR_1 = "indicator_1"
    """
    The dataset's ``ind1`` structural indicator.
    """

    @property
    def noun(self) -> str:
        """
        What the attribute is, in words.
        """
        return {MoleculeAttribute.INDICATOR_1: "the ind1 indicator"}[self]

    @property
    def circuit_variable_name(self) -> str:
        """
        The name a fitted circuit gives the attribute.
        """
        molecule = variable(MutagenesisMolecule)
        return {MoleculeAttribute.INDICATOR_1: molecule.indicator_1}[self]._name_


class MoleculeCount(StrEnum):
    """
    An aggregation statistic counting over a molecule's atoms or bonds.
    """

    ATOMS = "atom_count"
    """
    How many atoms the molecule holds, whatever their element.
    """

    CHLORINE_ATOMS = "chlorine_count"
    """
    How many of its atoms are chlorine.
    """

    BRANCHING_ATOMS = "branching_atom_count"
    """
    How many of its atoms carry three or four bonds.
    """

    DOUBLE_BONDS = "double_bond_count"
    """
    How many of its bonds are double.
    """

    AROMATIC_BONDS = "aromatic_bond_count"
    """
    How many of its bonds are aromatic.
    """

    @property
    def noun(self) -> str:
        """
        What the statistic counts, in words.
        """
        return {
            MoleculeCount.ATOMS: "atoms",
            MoleculeCount.CHLORINE_ATOMS: "chlorine atoms",
            MoleculeCount.BRANCHING_ATOMS: "branching atoms",
            MoleculeCount.DOUBLE_BONDS: "double bonds",
            MoleculeCount.AROMATIC_BONDS: "aromatic bonds",
        }[self]

    @property
    def circuit_variable_name(self) -> str:
        """
        The name a fitted circuit gives the statistic.
        """
        aggregations = variable(MutagenesisMoleculeAggregations)
        return {
            MoleculeCount.ATOMS: aggregations.atom_count(),
            MoleculeCount.CHLORINE_ATOMS: aggregations.chlorine_count(),
            MoleculeCount.BRANCHING_ATOMS: aggregations.branching_atom_count(),
            MoleculeCount.DOUBLE_BONDS: aggregations.double_bond_count(),
            MoleculeCount.AROMATIC_BONDS: aggregations.aromatic_bond_count(),
        }[self]._name_


# %% the questions


@dataclass(frozen=True, kw_only=True)
class MutagenesisQuestion(ABC):
    """
    One question of the form: does this property of a molecule cause that, once the
    confounders are adjusted for?
    """

    atom_count: int = 2
    """
    How many atoms the grounding query lists.
    """

    bond_count: int = 1
    """
    How many bonds the grounding query lists.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        What the question is called where answers are collected.
        """

    @property
    @abstractmethod
    def asked(self) -> str:
        """
        The question, in words.
        """

    @property
    @abstractmethod
    def cause_variable_name(self) -> str:
        """
        The name a fitted circuit gives the cause, to stratify the fit by.
        """

    @property
    @abstractmethod
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        """
        The names a fitted circuit gives the variables to adjust for.
        """

    @property
    @abstractmethod
    def effect_value(self) -> Any:
        """
        The value of the effect whose probability the question asks for.
        """

    @abstractmethod
    def build(self) -> Any:
        """
        The grounding query, with its cause and its effect marked.
        """

    def _molecule(self, **specified: Any) -> Any:
        """
        A molecule query leaving every atom's and every bond's own attributes open, so
        that grounding has to retain the statistics over them rather than integrate them
        out.

        :param specified: Molecule attributes to mark or fix; the rest are left open.
        :return: The molecule query.
        """
        parts: Dict[str, Any] = {
            "atoms": [
                a(MutagenesisAtom)(
                    element=..., atom_type=..., charge=..., bond_count=...
                )
                for _ in range(self.atom_count)
            ],
            "bonds": [
                a(MutagenesisBond)(bond_type=...) for _ in range(self.bond_count)
            ],
        }
        attributes: Dict[str, Any] = {
            field.name: ...
            for field in fields(MutagenesisMolecule)
            if field.name not in parts
        }
        attributes.update(specified)
        return a(MutagenesisMolecule)(**parts, **attributes)


@dataclass(frozen=True, kw_only=True)
class CountCausesMutagenicity(MutagenesisQuestion):
    """
    Does how many of something a molecule holds cause it to be mutagenic?
    """

    count: MoleculeCount
    """
    What to count.
    """

    adjusted_for: Tuple[MoleculeAttribute, ...] = (MoleculeAttribute.INDICATOR_1,)
    """
    The molecule attributes to adjust for.
    """

    @property
    def name(self) -> str:
        adjusting = (
            "_and_".join(attribute.value for attribute in self.adjusted_for)
            if self.adjusted_for
            else "nothing"
        )
        return f"{self.count.value}_causes_mutagenicity_adjusting_{adjusting}"

    @property
    def asked(self) -> str:
        adjusting = (
            " and ".join(attribute.noun for attribute in self.adjusted_for)
            if self.adjusted_for
            else "nothing"
        )
        return (
            f"Does how many {self.count.noun} a molecule holds cause it to be "
            f"mutagenic, adjusting for {adjusting}?"
        )

    @property
    def cause_variable_name(self) -> str:
        return self.count.circuit_variable_name

    @property
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        return tuple(attribute.circuit_variable_name for attribute in self.adjusted_for)

    @property
    def effect_value(self) -> bool:
        return True

    def build(self) -> Any:
        query = self._molecule(
            **{self.count.value: cause},
            **{attribute.value: confounder for attribute in self.adjusted_for},
        )
        query.causes_effect(query.mutagenic == True)
        return query


@dataclass(frozen=True, kw_only=True)
class IndicatorCausesMutagenicity(MutagenesisQuestion):
    """
    Does the ``ind1`` structural indicator cause mutagenicity, or does it only mark the
    large molecules that are mutagenic for their size?
    """

    adjusted_for: Tuple[MoleculeCount, ...] = (MoleculeCount.ATOMS,)
    """
    The counts to adjust for.
    """

    @property
    def name(self) -> str:
        adjusting = "_and_".join(count.value for count in self.adjusted_for)
        return f"indicator_1_causes_mutagenicity_adjusting_{adjusting}"

    @property
    def asked(self) -> str:
        adjusting = " and ".join(count.noun for count in self.adjusted_for)
        return (
            "Does the ind1 indicator cause a molecule to be mutagenic, adjusting for "
            f"how many {adjusting} it holds?"
        )

    @property
    def cause_variable_name(self) -> str:
        return MoleculeAttribute.INDICATOR_1.circuit_variable_name

    @property
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        return tuple(count.circuit_variable_name for count in self.adjusted_for)

    @property
    def effect_value(self) -> bool:
        return True

    def build(self) -> Any:
        query = self._molecule(
            **{MoleculeAttribute.INDICATOR_1.value: cause},
            **{count.value: confounder for count in self.adjusted_for},
        )
        query.causes_effect(query.mutagenic == True)
        return query


@dataclass(frozen=True, kw_only=True)
class IndicatorCausesElement(MutagenesisQuestion):
    """
    Does the ``ind1`` indicator cause one of a molecule's atoms to be of a given
    element?

    The cause is the molecule's own attribute and the effect is one atom's, so answering
    it needs a model that keeps the atoms rather than only summarising them.
    """

    element: MutagenesisElement
    """
    The element the atom is asked to be.
    """

    atom_index: int = 0
    """
    Which of the listed atoms the effect is read on.
    """

    @property
    def name(self) -> str:
        return f"indicator_1_causes_{self._element_noun}_atom"

    @property
    def asked(self) -> str:
        return (
            "Does the ind1 indicator cause one of a molecule's atoms to be "
            f"{self._element_noun}?"
        )

    @property
    def _element_noun(self) -> str:
        """
        The element's name rather than its one-letter symbol, for reading.
        """
        return self.element.name.lower()

    @property
    def cause_variable_name(self) -> str:
        return MoleculeAttribute.INDICATOR_1.circuit_variable_name

    @property
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def effect_value(self) -> MutagenesisElement:
        return self.element

    def build(self) -> Any:
        query = self._molecule(**{MoleculeAttribute.INDICATOR_1.value: cause})
        query.causes_effect(query.atoms[self.atom_index].element == self.element)
        return query


@dataclass(frozen=True, kw_only=True)
class CountCausesTerminalAtom(MutagenesisQuestion):
    """
    Does how many of something a molecule holds cause one of its atoms to be terminal,
    carrying a single bond?

    The cause counts over the atoms and the effect is one atom's own attribute.
    """

    count: MoleculeCount = MoleculeCount.BRANCHING_ATOMS
    """
    What to count.
    """

    atom_index: int = 0
    """
    Which of the listed atoms the effect is read on.
    """

    bonds_of_a_terminal_atom: int = 1
    """
    How many bonds an atom carries to count as terminal.
    """

    @property
    def name(self) -> str:
        return f"{self.count.value}_causes_terminal_atom"

    @property
    def asked(self) -> str:
        return (
            f"Does how many {self.count.noun} a molecule holds cause one of its atoms "
            "to be terminal?"
        )

    @property
    def cause_variable_name(self) -> str:
        return self.count.circuit_variable_name

    @property
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def effect_value(self) -> int:
        return self.bonds_of_a_terminal_atom

    def build(self) -> Any:
        query = self._molecule(**{self.count.value: cause})
        query.causes_effect(
            query.atoms[self.atom_index].bond_count == self.bonds_of_a_terminal_atom
        )
        return query


def question_catalogue() -> List[MutagenesisQuestion]:
    """
    :return: The questions asked of the dataset: each count as a cause of mutagenicity
        adjusting for the ind1 indicator, the branching atoms again adjusting for
        nothing so the difference adjusting makes is visible, the indicator as a cause
        of mutagenicity adjusting for the molecule's size, and two whose effect is an
        atom's own attribute rather than the molecule's.
    """
    return [
        *(
            CountCausesMutagenicity(count=count)
            for count in (
                MoleculeCount.BRANCHING_ATOMS,
                MoleculeCount.CHLORINE_ATOMS,
                MoleculeCount.AROMATIC_BONDS,
            )
        ),
        CountCausesMutagenicity(count=MoleculeCount.BRANCHING_ATOMS, adjusted_for=()),
        IndicatorCausesMutagenicity(),
        IndicatorCausesElement(element=MutagenesisElement.CARBON),
        CountCausesTerminalAtom(),
    ]


# %% asking one question of a fitted circuit


@dataclass
class MutagenesisDoQuery:
    """
    Fits a relational circuit on molecules, grounds it against one question, and reads
    the question's effect off every region of its cause.
    """

    question: MutagenesisQuestion
    """
    The question to ask.
    """

    monte_carlo_sample_count: int = 2000
    """
    How many samples grounding draws when it retains a statistic over the parts.
    """

    random_seed: int = 0
    """
    Seed applied to the global NumPy random state before grounding, which draws those
    samples.
    """

    def run(self, training_molecules: List[MutagenesisMolecule]) -> DoQueryAnswer:
        """
        Fit a circuit on the molecules and answer the question on it.

        :param training_molecules: Molecules to fit on.
        :return: The answer, one entry per region of the cause.
        """
        causal_circuit = self._grounded_against(training_molecules)
        [cause_variable] = causal_circuit.causal_variables
        [effect_variable] = causal_circuit.effect_variables
        conditioned_circuit = causal_circuit.backdoor_adjustment(
            cause_variable, effect_variable
        )
        adjusted_circuit = causal_circuit.backdoor_adjustment(
            cause_variable,
            effect_variable,
            adjustment_variables=self._adjustment_variables(causal_circuit),
        )

        regions = [
            CauseRegion(
                description=str(region.event.simple_sets[0][cause_variable]),
                probability=region.probability,
                conditioned_probability=self._effect_probability(
                    conditioned_circuit, region.event, effect_variable
                ),
                adjusted_probability=self._effect_probability(
                    adjusted_circuit, region.event, effect_variable
                ),
            )
            for region in causal_circuit.disjoint_support_regions_of(
                cause_variable
            )
        ]
        return DoQueryAnswer(
            asked=self.question.asked,
            training_example_count=len(training_molecules),
            regions=tuple(sorted(regions, key=lambda region: region.description)),
        )

    def _grounded_against(
        self, training_molecules: List[MutagenesisMolecule]
    ) -> RelationalCausalCircuit:
        """
        Fit a circuit whose class-level branches each hold one value of the cause, and
        ground it against the question.

        Stratifying by the cause is what lets the registration verify support
        determinism: an unconstrained fit gives no guarantee that training rows sharing
        a value of the cause end up under one branch, and two branches claiming one
        value are rejected.

        :param training_molecules: Molecules to fit on.
        :return: The grounded circuit, with its cause and effect registered.
        """
        model = RelationalProbabilisticCircuit(
            MutagenesisMolecule,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
            learning_method=StratifiedLearning(
                variables=[self.question.cause_variable_name],
                method=JointProbabilityTree(),
            ),
        )
        model.fit(training_molecules)
        registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)
        np.random.seed(self.random_seed)
        return registry.get_model(UnderspecifiedParameters(self.question.build()))

    def _adjustment_variables(
        self, causal_circuit: RelationalCausalCircuit
    ) -> List[Variable]:
        """
        :param causal_circuit: The grounded circuit to resolve against.
        :return: One variable per attribute the question adjusts for.
        """
        return [
            RelationalCausalCircuit.resolve_variable(
                causal_circuit.probabilistic_circuit, name
            )
            for name in self.question.adjustment_variable_names
        ]

    def _effect_probability(
        self, interventional_circuit: Any, region: Event, effect_variable: Variable
    ) -> float:
        """
        Read how likely the question's effect is on one region of the cause.

        :param interventional_circuit: A joint circuit over the cause and the effect.
        :param region: The region of the cause to truncate to.
        :param effect_variable: The effect's variable.
        :return: The truncated circuit's probability of the effect's value.
        """
        truncated_circuit, _ = interventional_circuit.truncated(
            region.fill_missing_variables_pure(interventional_circuit.variables)
        )
        effect = (
            SimpleEvent.from_data({effect_variable: self.question.effect_value})
            .as_composite_set()
            .fill_missing_variables_pure(truncated_circuit.variables)
        )
        return float(truncated_circuit.probability(effect))
