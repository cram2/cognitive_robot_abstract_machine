"""
The causal questions both pipelines are asked, each as one ``cause``/``causes_effect``
EQL query that reads the same for either pipeline.

Every query lists at least one atom and one bond with every attribute left open: that is
what makes grounding retain the molecule's aggregation counts as variables instead of
computing them from an empty part list, and the flat table ignores parts a query says
nothing about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from krrood.entity_query_language.factories import a, cause, confounder
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List

from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisBond,
    MutagenesisElement,
    MutagenesisMolecule,
)
from experiments.causal_reasoning.mutagenesis.flat_table import MoleculeSchema

# %% building blocks


def atom_query(**specified: Any) -> Match:
    """
    A query for one atom with every attribute left open but the given ones.

    :param specified: Attribute markers or values to set instead of leaving open.
    :return: The query.
    """
    return a(MutagenesisAtom)(
        **{
            atom_field.name: specified.get(atom_field.name, ...)
            for atom_field in fields(MutagenesisAtom)
        }
    )


def bond_query(**specified: Any) -> Match:
    """
    A query for one bond with every attribute left open but the given ones.

    :param specified: Attribute markers or values to set instead of leaving open.
    :return: The query.
    """
    return a(MutagenesisBond)(
        **{
            bond_field.name: specified.get(bond_field.name, ...)
            for bond_field in fields(MutagenesisBond)
        }
    )


def molecule_query(
    atoms: List[Match],
    bonds: List[Match],
    schema: MoleculeSchema = MoleculeSchema(),
    **specified: Any,
) -> Match:
    """
    A query for a molecule with the given atoms and bonds and every scalar attribute
    left open but the given ones.

    :param atoms: One query per atom.
    :param bonds: One query per bond.
    :param schema: How the molecule's attributes are named.
    :param specified: Scalar attribute markers or values to set instead of leaving open;
        an aggregation count's name is accepted too.
    :return: The query.
    """
    scalar_fields = schema.scalar_fields
    return a(MutagenesisMolecule)(
        **{name: specified.get(name, ...) for name in scalar_fields},
        **{
            name: value
            for name, value in specified.items()
            if name not in scalar_fields
        },
        atoms=atoms,
        bonds=bonds,
    )


# %% the questions


@dataclass(frozen=True, kw_only=True)
class CausalQueryCase(ABC):
    """
    One question, as a query and as words.
    """

    atom_count: int = 1
    """
    How many atoms the query lists with every attribute open.
    """

    bond_count: int = 1
    """
    How many bonds the query lists with every attribute open.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        A short identifier for tables.
        """

    @property
    @abstractmethod
    def question(self) -> str:
        """
        The question in words.
        """

    @abstractmethod
    def build(self) -> Match:
        """
        :return: The query, freshly built, with its cause, confounders and effect
            marked.
        """

    @abstractmethod
    def describe_cause(self, region: str) -> str:
        """
        :param region: A region of the cause, written out.
        :return: The cause set to that region, in words.
        """

    @property
    @abstractmethod
    def effect(self) -> str:
        """
        The effect in words.
        """

    def _open_atoms(self) -> List[Match]:
        """
        :return: One fully open query per listed atom.
        """
        return [atom_query() for _ in range(self.atom_count)]

    def _open_bonds(self) -> List[Match]:
        """
        :return: One fully open query per listed bond.
        """
        return [bond_query() for _ in range(self.bond_count)]


@dataclass(frozen=True, kw_only=True)
class CountCausesMutagenicity(CausalQueryCase):
    """
    Does one of the molecule's aggregation counts cause it to be mutagenic, once the
    ``ind1`` structural indicator, which marks the fused-ring molecules that are both
    large and mostly mutagenic, is adjusted for?
    """

    statistic_name: str
    """
    The aggregation statistic of
    :class:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisMoleculeAggregations`
    that is the cause.
    """

    count_noun: str
    """
    What the count counts, in words, such as ``branching atoms``.
    """

    @property
    def name(self) -> str:
        return f"{self.statistic_name}_causes_mutagenicity"

    @property
    def question(self) -> str:
        return (
            f"How many {self.count_noun} cause a molecule to be mutagenic, adjusting "
            "for the ind1 indicator?"
        )

    def build(self) -> Match:
        query = molecule_query(
            self._open_atoms(),
            self._open_bonds(),
            indicator_1=confounder,
            **{self.statistic_name: cause},
        )
        query.causes_effect(query.variable.mutagenic == True)
        return query

    def describe_cause(self, region: str) -> str:
        return f"{region} {self.count_noun}"

    @property
    def effect(self) -> str:
        return "the molecule is mutagenic"


@dataclass(frozen=True, kw_only=True)
class IndicatorCausesMutagenicity(CausalQueryCase):
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
        query = molecule_query(
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
class IndicatorCausesElement(CausalQueryCase):
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
        query = molecule_query(
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
class BranchingAtomsCauseTerminalAtom(CausalQueryCase):
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
        query = molecule_query(
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
class ElementCausesTerminalAtom(CausalQueryCase):
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
        atoms[self.atom_index] = atom_query(element=cause)
        query = molecule_query(atoms, self._open_bonds())
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
    return [
        CountCausesMutagenicity(
            statistic_name="branching_atom_count", count_noun="branching atoms"
        ),
        CountCausesMutagenicity(
            statistic_name="aromatic_bond_count", count_noun="aromatic bonds"
        ),
        CountCausesMutagenicity(
            statistic_name="double_bond_count", count_noun="double bonds"
        ),
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
