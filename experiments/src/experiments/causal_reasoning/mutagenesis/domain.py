"""
Domain classes mirroring the CTU relational-learning repository's Mutagenesis
dataset (https://relational.fel.cvut.cz/dataset/Mutagenesis), used to validate the
RSPN grounding and causal-query pipeline against a real, external, standard
relational-learning benchmark rather than only the ``SceneRoom``/``SceneObject`` toy
fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing_extensions import List

from krrood.entity_query_language.factories import entity, count_range, or_, variable
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)


class MutagenesisElement(StrEnum):
    """
    Chemical element of an atom, as recorded in the CTU Mutagenesis dataset.
    """

    CARBON = "c"
    HYDROGEN = "h"
    OXYGEN = "o"
    NITROGEN = "n"
    CHLORINE = "cl"
    BROMINE = "br"
    FLUORINE = "f"
    IODINE = "i"


class MutagenesisBondType(IntEnum):
    """
    ``bonds.bond_type`` value, as recorded in the CTU Mutagenesis dataset.

    1, 2, 3 and 7 follow the encoding documented in the dataset's own ILP literature
    (single, double, triple, aromatic). 4 and 5 also occur in the live table, in a
    handful of bonds, but their chemical meaning is not documented anywhere the schema
    exposes, so they are kept under their raw codes rather than guessed at.
    """

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    UNDOCUMENTED_TYPE_4 = 4
    UNDOCUMENTED_TYPE_5 = 5
    AROMATIC = 7


@dataclass
class MutagenesisAtom:
    """
    One atom of a :class:`MutagenesisMolecule`.
    """

    element: MutagenesisElement
    """
    The atom's chemical element.
    """

    atom_type: int
    """
    The dataset's ILP atom-type code, a finer-grained classification than
    :attr:`element` alone.
    """

    charge: float
    """
    The atom's partial charge.
    """

    bond_count: int
    """
    Number of bonds this atom participates in.

    A bond connects exactly two atoms, so this is read off the CTU dataset's ``bonds``
    table (``atom1_id``, ``atom2_id``) when the molecule is loaded; an atom with three
    or more bonds sits at a ring fusion or branch point in the molecular graph.
    """


@dataclass
class MutagenesisBond:
    """
    One bond of a :class:`MutagenesisMolecule`, modeled as an exchangeable part on its
    own rather than an edge between two :class:`MutagenesisAtom` entries.

    A real bond connects exactly two atoms, and the CTU dataset's own ``bonds`` table
    records which ones (``atom1_id``, ``atom2_id``). This class itself does not carry
    that connectivity, since the RSPN grounding this domain feeds fits and grounds
    ``atoms`` and ``bonds`` as two independent exchangeable parts of the parent
    molecule, and a bond referencing specific atom objects would cross that independence
    boundary. The connectivity is not discarded, though:
    :attr:`MutagenesisAtom.bond_count` is derived from this same ``atom1_id`` /
    ``atom2_id`` data at load time, so each atom still carries how many bonds it
    participates in, without either class needing to reference the other.
    """

    bond_type: MutagenesisBondType
    """
    The bond's chemical type.
    """


@dataclass
class MutagenesisMolecule:
    """
    One molecule of the CTU Mutagenesis dataset, with its atoms and bonds as
    exchangeable parts.
    """

    indicator_1: bool
    """
    The dataset's ``ind1`` structural indicator, a strong known predictor of
    mutagenicity in this benchmark.
    """

    logp: float
    """
    Octanol-water partition coefficient, a measure of the molecule's hydrophobicity.
    """

    lumo: float
    """
    Energy of the molecule's lowest unoccupied molecular orbital.
    """

    mutagenic: bool
    """
    Whether the molecule is mutagenic; the dataset's prediction target.
    """

    atoms: List[MutagenesisAtom]
    """
    The molecule's atoms.
    """

    bonds: List[MutagenesisBond]
    """
    The molecule's bonds.
    """


@dataclass
class MutagenesisMoleculeAggregations(AggregationStatistic[MutagenesisMolecule]):
    """
    Aggregation statistics for :class:`MutagenesisMolecule` over its ``atoms`` and
    ``bonds`` fields.
    """

    @aggregation_statistic("atoms")
    def chlorine_count(self) -> int:
        """
        Count of chlorine atoms.
        """
        element_variable = variable(MutagenesisAtom, self.instance.atoms).element
        [result] = (
            entity(count_range(element_variable))
            .where(element_variable == MutagenesisElement.CHLORINE)
            .tolist()
        )
        return result

    @aggregation_statistic("atoms")
    def branching_atom_count(self) -> int:
        """
        Count of atoms with three or four bonds: ring-fusion and branch points in the
        molecular graph.
        """
        bond_count_variable = variable(MutagenesisAtom, self.instance.atoms).bond_count
        [result] = (
            entity(count_range(bond_count_variable))
            .where(or_(bond_count_variable == 3, bond_count_variable == 4))
            .tolist()
        )
        return result

    @aggregation_statistic("bonds")
    def double_bond_count(self) -> int:
        """
        Count of double bonds.
        """
        bond_type_variable = variable(MutagenesisBond, self.instance.bonds).bond_type
        [result] = (
            entity(count_range(bond_type_variable))
            .where(bond_type_variable == MutagenesisBondType.DOUBLE)
            .tolist()
        )
        return result

    @aggregation_statistic("bonds")
    def aromatic_bond_count(self) -> int:
        """
        Count of aromatic bonds.
        """
        bond_type_variable = variable(MutagenesisBond, self.instance.bonds).bond_type
        [result] = (
            entity(count_range(bond_type_variable))
            .where(bond_type_variable == MutagenesisBondType.AROMATIC)
            .tolist()
        )
        return result
