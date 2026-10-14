"""
Asking the Mutagenesis circuit what a molecule's properties cause.

Each question marks one cause and one effect in the query itself, grounds the relational
circuit against it, and reads the effect off every region of the cause twice: once by
conditioning alone and once with backdoor adjustment, so the two can be set side by
side.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum

from krrood.entity_query_language.factories import a, cause, confounder, variable
from typing_extensions import Any, Dict, List, Tuple, Type

from experiments.causal_reasoning.do_query import (
    CausalQuestion,
    CauseStratification,
    DoQuery,
)
from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisBond,
    MutagenesisElement,
    MutagenesisMolecule,
    MutagenesisMoleculeAggregations,
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
class MutagenesisQuestion(CausalQuestion):
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
    def stratification(self) -> CauseStratification:
        """
        :return: The grouping for the cause, which a molecule always carries itself.
        """
        return CauseStratification(column=self.cause_variable_name)

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
class MutagenesisDoQuery(DoQuery):
    """
    Asks one mutagenesis question of a circuit fitted on molecules.
    """

    question: MutagenesisQuestion
    """
    The question to ask.
    """

    @property
    def fitted_type(self) -> Type:
        """
        :return: The molecule the circuit is fitted on.
        """
        return MutagenesisMolecule
