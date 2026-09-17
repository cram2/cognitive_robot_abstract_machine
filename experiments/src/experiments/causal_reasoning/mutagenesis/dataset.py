"""
Loading the CTU Mutagenesis dataset (https://relational.fel.cvut.cz/dataset/Mutagenesis)
into :class:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisMolecule`
instances, plus a synthetic generator with the same shape for the CI-safe pairing
:mod:`test.causal_reasoning_test.test_mutagenesis_pipeline` needs alongside its
live-dataset tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from typing_extensions import Callable, Dict, List, Self, Tuple, TypeVar

from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisBond,
    MutagenesisBondType,
    MutagenesisElement,
    MutagenesisMolecule,
)
from experiments.causal_reasoning.mutagenesis.exceptions import (
    MutagenesisDatasetUnavailableError,
)

T = TypeVar("T")

# %% the CTU database


@dataclass(frozen=True)
class MutagenesisDatabaseConnection:
    """
    Connection details for the CTU relational-dataset repository's Mutagenesis database.
    """

    host: str = "relational.fel.cvut.cz"
    """
    Hostname of the MariaDB server.
    """

    port: int = 3306
    """
    Port of the MariaDB server.
    """

    user: str = "guest"
    """
    Username for the read-only guest account.
    """

    password: str = "ctu-relational"
    """
    Password for the read-only guest account.
    """

    database: str = "mutagenesis_188"
    """
    Database name; ``mutagenesis_188`` is the 188-molecule "regression friendly" set.
    """

    @property
    def url(self) -> str:
        """
        The SQLAlchemy connection URL for this connection.
        """
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


def is_mutagenesis_dataset_reachable(
    connection: MutagenesisDatabaseConnection = MutagenesisDatabaseConnection(),
) -> bool:
    """
    Check whether the CTU Mutagenesis database can be connected to right now.

    :param connection: Connection details to check.
    :return:``True`` if a connection could be established, ``False`` otherwise.
    """
    engine = create_engine(connection.url, connect_args={"connect_timeout": 5})
    try:
        with engine.connect():
            return True
    except OperationalError:
        return False
    finally:
        engine.dispose()


def fetch_mutagenesis_molecules(
    connection: MutagenesisDatabaseConnection = MutagenesisDatabaseConnection(),
) -> List[MutagenesisMolecule]:
    """
    Download the Mutagenesis dataset and convert it into domain objects.

    Pulls the ``drugs``, ``atoms`` and ``bonds`` tables, groups atoms and bonds by
    their owning molecule, and derives each atom's
    :attr:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisAtom.bond_count`
    from the bond table's ``atom1_id`` / ``atom2_id`` columns.

    :param connection: Connection details for the database.
    :return: One :class:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisMolecule`
        per row of ``drugs``.
    :raises MutagenesisDatasetUnavailableError: If the database cannot be reached.
    """
    engine = create_engine(connection.url, connect_args={"connect_timeout": 5})
    try:
        drugs = pd.read_sql_table("drugs", engine)
        atoms = pd.read_sql_table("atoms", engine)
        bonds = pd.read_sql_table("bonds", engine)
    except OperationalError as error:
        raise MutagenesisDatasetUnavailableError(str(error)) from error
    finally:
        engine.dispose()

    bond_count_by_atom = (
        pd.concat([bonds["atom1_id"], bonds["atom2_id"]]).value_counts().to_dict()
    )
    atoms_by_drug = {
        drug_id: [
            MutagenesisAtom(
                element=MutagenesisElement(row.element),
                atom_type=int(row.atom_type),
                charge=float(row.charge),
                bond_count=bond_count_by_atom.get(row.id, 0),
            )
            for row in group.itertuples()
        ]
        for drug_id, group in atoms.groupby("drug_id")
    }
    bonds_by_drug = {
        drug_id: [
            MutagenesisBond(bond_type=MutagenesisBondType(row.bond_type))
            for row in group.itertuples()
        ]
        for drug_id, group in bonds.groupby("drug_id")
    }

    return [
        MutagenesisMolecule(
            indicator_1=bool(drug.ind1),
            logp=float(drug.logp),
            lumo=float(drug.lumo),
            mutagenic=bool(drug.active),
            atoms=atoms_by_drug[drug.id],
            bonds=bonds_by_drug.get(drug.id, []),
        )
        for drug in drugs.itertuples()
    ]


# %% a synthetic stand-in


def synthetic_mutagenesis_molecules(
    random_state: np.random.Generator,
    molecule_count: int = 20,
    atom_count: int = 2,
    bond_count: int = 3,
) -> List[MutagenesisMolecule]:
    """
    Generate a small, network-free dataset with the same shape as
    :func:`fetch_mutagenesis_molecules`, for tests that must run without live network
    access.

    Each molecule's chlorine count is drawn to be either 0 or ``atom_count`` (all
    chlorine, none hydrogen), and ``mutagenic`` is set to match, baking in a real
    chlorine-count/mutagenicity correlation so a fitted circuit has something genuine to
    pick up, mirroring ``_room_with_chair_count`` in ``test_rspns.py``.

    :param random_state: Source of randomness for the non-causal fields.
    :param molecule_count: How many molecules to generate.
    :param atom_count: How many atoms each molecule has.
    :param bond_count: How many bonds each molecule has.
    :return: The generated molecules.
    """
    bond_types = list(MutagenesisBondType)
    molecules = []
    for index in range(molecule_count):
        all_chlorine = index % 2 == 0
        element = (
            MutagenesisElement.CHLORINE if all_chlorine else MutagenesisElement.HYDROGEN
        )
        atoms = [
            MutagenesisAtom(
                element=element,
                atom_type=int(random_state.integers(1, 10)),
                charge=float(random_state.uniform(-0.5, 0.5)),
                bond_count=int(random_state.integers(1, 5)),
            )
            for _ in range(atom_count)
        ]
        bonds = [
            MutagenesisBond(
                bond_type=bond_types[random_state.integers(len(bond_types))]
            )
            for _ in range(bond_count)
        ]
        molecules.append(
            MutagenesisMolecule(
                indicator_1=bool(random_state.integers(0, 2)),
                logp=float(random_state.uniform(0, 5)),
                lumo=float(random_state.uniform(-3, 0)),
                mutagenic=all_chlorine,
                atoms=atoms,
                bonds=bonds,
            )
        )
    return molecules


# %% a set of molecules


@dataclass(frozen=True)
class MutagenicRate:
    """
    How often a group of molecules is mutagenic.
    """

    molecule_count: int
    """
    How many molecules the group holds.
    """

    mutagenic_count: int
    """
    How many of them are mutagenic.
    """

    @property
    def rate(self) -> float:
        """
        The mutagenic share.
        """
        return self.mutagenic_count / self.molecule_count


@dataclass
class MutagenesisDataset:
    """
    A set of molecules, as fetched and as read by the pipelines.
    """

    molecules: List[MutagenesisMolecule] = field(default_factory=list)
    """
    The molecules.
    """

    @property
    def mutagenic_rate(self) -> float:
        """
        Share of molecules that are mutagenic.
        """
        return sum(molecule.mutagenic for molecule in self.molecules) / len(
            self.molecules
        )

    def mutagenic_rate_by(
        self, key: Callable[[MutagenesisMolecule], T]
    ) -> Dict[T, MutagenicRate]:
        """
        The mutagenic share among the molecules sharing a value.

        :param key: What to group the molecules by.
        :return: Each value's mutagenic rate, by value.
        """
        by_value: Dict[T, List[MutagenesisMolecule]] = {}
        for molecule in self.molecules:
            by_value.setdefault(key(molecule), []).append(molecule)
        return {
            value: MutagenicRate(
                molecule_count=len(molecules),
                mutagenic_count=sum(molecule.mutagenic for molecule in molecules),
            )
            for value, molecules in sorted(by_value.items())
        }

    def with_shuffled_parts(self, random_state: np.random.Generator) -> Self:
        """
        The same molecules with their atoms and bonds in a random order each.

        :param random_state: Source of randomness for the orders.
        :return: The dataset with reordered parts.
        """
        return type(self)(
            [
                replace(
                    molecule,
                    atoms=[
                        molecule.atoms[index]
                        for index in random_state.permutation(len(molecule.atoms))
                    ],
                    bonds=[
                        molecule.bonds[index]
                        for index in random_state.permutation(len(molecule.bonds))
                    ],
                )
                for molecule in self.molecules
            ]
        )

    def split(
        self, train_fraction: float, random_state: np.random.Generator
    ) -> Tuple[Self, Self]:
        """
        Shuffle the molecules and split them in two.

        :param train_fraction: Share of molecules that go into the first part.
        :param random_state: Source of randomness for the shuffle.
        :return: The first and second part.
        """
        order = random_state.permutation(len(self.molecules))
        split_index = int(train_fraction * len(self.molecules))
        first = [self.molecules[index] for index in order[:split_index]]
        second = [self.molecules[index] for index in order[split_index:]]
        return type(self)(first), type(self)(second)
