"""
The causal questions both pipelines are asked, each as one ``cause``/``causes_effect``
EQL query that reads the same for either pipeline.

Every query lists at least one atom and one bond with every attribute left open: that is
what makes grounding retain the molecule's aggregation counts as variables instead of
computing them from an empty part list, and the flat table ignores parts a query says
nothing about.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from krrood.entity_query_language.factories import cause, confounder
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List, Tuple

from experiments.causal_reasoning.comparison.domain import RelationalDomain
from experiments.causal_reasoning.comparison.queries import (
    AdjustedCountCase,
    CausalQueryCase,
    Confounder,
    part_query,
)
from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisElement,
    PartField,
    molecule_domain,
)

# %% building blocks


@dataclass(frozen=True, kw_only=True)
class MoleculeQueryCase(CausalQueryCase, ABC):
    """
    One question about a molecule.
    """

    @property
    def domain(self) -> RelationalDomain:
        return molecule_domain()

    def _open_atoms(self) -> List[Match]:
        """
        :return: One fully open query per listed atom.
        """
        return self.open_parts()[PartField.ATOMS]

    def _open_bonds(self) -> List[Match]:
        """
        :return: One fully open query per listed bond.
        """
        return self.open_parts()[PartField.BONDS]

    def _molecule_query(
        self, atoms: List[Match], bonds: List[Match], **specified: Any
    ) -> Match:
        """
        :param atoms: One query per listed atom.
        :param bonds: One query per listed bond.
        :param specified: Molecule attribute markers or values to set instead of
            leaving open.
        :return: The query for the molecule.
        """
        return self.example_query(
            {PartField.ATOMS: atoms, PartField.BONDS: bonds}, **specified
        )


INDICATOR = Confounder(name="indicator_1", noun="the ind1 indicator")
"""
Adjusting for the ``ind1`` structural indicator, which marks the fused-ring molecules
that are both large and mostly mutagenic.
"""

ATOM_COUNT = Confounder(name="atom_count", noun="the number of atoms")
"""
Adjusting for the size of the molecule.
"""


@dataclass(frozen=True, kw_only=True)
class CountCausesMutagenicity(MoleculeQueryCase, AdjustedCountCase):
    """
    Does one of the molecule's aggregation counts cause it to be mutagenic, once the
    given confounders are adjusted for?

    A flat table without a column for one of them refuses the question.
    """

    count_noun: str
    """
    What the count counts, in words, such as ``branching atoms``.
    """

    confounders: Tuple[Confounder, ...] = (INDICATOR,)
    """
    What to adjust for: the ``ind1`` indicator unless asked otherwise.
    """

    @property
    def name(self) -> str:
        adjusting = "_and_".join(confounder.name for confounder in self.confounders)
        return f"{self.statistic_name}_causes_mutagenicity_adjusting_{adjusting}"

    @property
    def question(self) -> str:
        adjusting = " and ".join(confounder.noun for confounder in self.confounders)
        return (
            f"How many {self.count_noun} cause a molecule to be mutagenic, adjusting "
            f"for {adjusting}?"
        )

    def build(self) -> Match:
        query = self._molecule_query(
            self._open_atoms(),
            self._open_bonds(),
            **{self.statistic_name: cause},
            **{adjusted.name: confounder for adjusted in self.confounders},
        )
        query.causes_effect(query.variable.mutagenic == True)
        return query

    def describe_cause(self, region: str) -> str:
        return f"{region} {self.count_noun}"

    @property
    def effect(self) -> str:
        return "the molecule is mutagenic"


@dataclass(frozen=True, kw_only=True)
class IndicatorCausesMutagenicity(MoleculeQueryCase):
    """
    Does the ``ind1`` structural indicator cause mutagenicity, once one other attribute
    of the molecule is adjusted for?
    """

    confounder_name: str
    """
    The molecule attribute or aggregation count to adjust for.
    """

    confounder_noun: str
    """
    The confounder in words, such as ``branching-atom count``.
    """

    @property
    def name(self) -> str:
        return f"indicator_causes_mutagenicity_adjusting_{self.confounder_name}"

    @property
    def question(self) -> str:
        return (
            "Does the ind1 indicator cause a molecule to be mutagenic, adjusting for "
            f"its {self.confounder_noun}?"
        )

    def build(self) -> Match:
        query = self._molecule_query(
            self._open_atoms(),
            self._open_bonds(),
            indicator_1=cause,
            **{self.confounder_name: confounder},
        )
        query.causes_effect(query.variable.mutagenic == True)
        return query

    def describe_cause(self, region: str) -> str:
        return f"ind1 = {region}"

    @property
    def effect(self) -> str:
        return "the molecule is mutagenic"


@dataclass(frozen=True, kw_only=True)
class IndicatorCausesElement(MoleculeQueryCase):
    """
    Does the ``ind1`` structural indicator cause one of the molecule's atoms to be of a
    given element?

    The cause is a molecule attribute, the effect an atom's own.
    """

    element: MutagenesisElement
    """
    The element the effect asks for.
    """

    atom_index: int = 0
    """
    Which listed atom the effect is about.
    """

    @property
    def name(self) -> str:
        return f"indicator_causes_{self.element.name.lower()}_atom_{self.atom_index}"

    @property
    def question(self) -> str:
        return (
            f"Does the ind1 indicator cause atom {self.atom_index} of a molecule to be "
            f"{self.element.name.lower()}?"
        )

    def build(self) -> Match:
        query = self._molecule_query(
            self._open_atoms(), self._open_bonds(), indicator_1=cause
        )
        query.causes_effect(
            query.variable.atoms[self.atom_index].element == self.element
        )
        return query

    def describe_cause(self, region: str) -> str:
        return f"ind1 = {region}"

    @property
    def effect(self) -> str:
        return f"atom {self.atom_index} is {self.element.name.lower()}"


@dataclass(frozen=True, kw_only=True)
class BranchingAtomsCauseTerminalAtom(MoleculeQueryCase):
    """
    Does the number of branching atoms in a molecule cause one of its atoms to be a
    terminal atom, one with a single bond?

    The cause is a count over the atoms, the effect one atom's own attribute.
    """

    atom_index: int = 0
    """
    Which listed atom the effect is about.
    """

    @property
    def name(self) -> str:
        return f"branching_atom_count_causes_terminal_atom_{self.atom_index}"

    @property
    def question(self) -> str:
        return (
            "How many branching atoms cause atom "
            f"{self.atom_index} of a molecule to be terminal, with a single bond?"
        )

    def build(self) -> Match:
        query = self._molecule_query(
            self._open_atoms(), self._open_bonds(), branching_atom_count=cause
        )
        query.causes_effect(query.variable.atoms[self.atom_index].bond_count == 1)
        return query

    def describe_cause(self, region: str) -> str:
        return f"{region} branching atoms"

    @property
    def effect(self) -> str:
        return f"atom {self.atom_index} is terminal"


@dataclass(frozen=True, kw_only=True)
class ElementCausesTerminalAtom(MoleculeQueryCase):
    """
    Does an atom's element cause it to be a terminal atom, one with a single bond?

    Cause and effect both live on one atom.
    """

    atom_index: int = 0
    """
    Which listed atom the question is about.
    """

    @property
    def name(self) -> str:
        return f"element_causes_terminal_atom_{self.atom_index}"

    @property
    def question(self) -> str:
        return (
            f"Does the element of atom {self.atom_index} of a molecule cause it to be "
            "terminal, with a single bond?"
        )

    def build(self) -> Match:
        atoms = self._open_atoms()
        atoms[self.atom_index] = part_query(MutagenesisAtom, element=cause)
        query = self._molecule_query(atoms, self._open_bonds())
        query.causes_effect(query.variable.atoms[self.atom_index].bond_count == 1)
        return query

    def describe_cause(self, region: str) -> str:
        return f"atom {self.atom_index} being of element {region}"

    @property
    def effect(self) -> str:
        return f"atom {self.atom_index} is terminal"


def molecule_level_cases() -> List[CausalQueryCase]:
    """
    :return: The questions whose cause and effect are both molecule attributes or
        counts, in the order they are asked.
    """
    counts = (
        ("branching_atom_count", "branching atoms"),
        ("aromatic_bond_count", "aromatic bonds"),
        ("double_bond_count", "double bonds"),
    )
    adjustments = ((INDICATOR,), (ATOM_COUNT,), (INDICATOR, ATOM_COUNT))
    return [
        CountCausesMutagenicity(
            statistic_name=statistic_name,
            count_noun=count_noun,
            confounders=confounders,
        )
        for statistic_name, count_noun in counts
        for confounders in adjustments
    ] + [
        IndicatorCausesMutagenicity(
            confounder_name="logp", confounder_noun="hydrophobicity (logp)"
        ),
        IndicatorCausesMutagenicity(
            confounder_name="branching_atom_count",
            confounder_noun="branching-atom count",
        ),
    ]


def atom_level_cases() -> List[CausalQueryCase]:
    """
    :return: The questions whose cause or effect lives on one atom, in the order they
        are asked.
    """
    return [
        IndicatorCausesElement(element=MutagenesisElement.CARBON),
        BranchingAtomsCauseTerminalAtom(),
        ElementCausesTerminalAtom(),
    ]


def query_catalogue() -> List[CausalQueryCase]:
    """
    Every question of the experiment: the molecule-level causes of mutagenicity first,
    then the questions whose cause or effect lives on one atom.

    :return: The questions, in the order they are asked.
    """
    return molecule_level_cases() + atom_level_cases()


def monte_carlo_cases() -> List[CausalQueryCase]:
    """
    The questions whose answers are followed as grounding draws more samples: one
    molecule-level count question and one whose effect lives on an atom, both of which
    leave every count open.

    :return: The two questions.
    """
    return [
        CountCausesMutagenicity(
            statistic_name="branching_atom_count",
            count_noun="branching atoms",
            confounders=(ATOM_COUNT,),
        ),
        BranchingAtomsCauseTerminalAtom(),
    ]
