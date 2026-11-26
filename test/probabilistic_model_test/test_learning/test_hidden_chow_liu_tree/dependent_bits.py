"""
Binary datasets with known dependencies between their variables, for testing structure
learners.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

import numpy as np
from random_events.set import Set
from random_events.variable import Symbolic
from sortedcontainers import SortedSet


# %% dependent bits


@dataclass(kw_only=True)
class DependentBits:
    """
    Rows of binary variables drawn from a mixture of three product distributions, in
    which every odd variable copies its even neighbour with probability 0.9.
    """

    number_of_variables: int = 6
    """
    The number of variables, even.
    """

    number_of_rows: int = 600
    """
    The number of rows.
    """

    seed: int = 0
    """
    The seed of the rows.
    """

    @property
    def variables(self) -> SortedSet:
        """
        The binary symbolic variables, over the domain {"0", "1"}.
        """
        return SortedSet(
            Symbolic(name=f"x{index:02d}", domain=Set.from_iterable(["0", "1"]))
            for index in range(self.number_of_variables)
        )

    @property
    def copied_pairs(self) -> set[frozenset[int]]:
        """
        The indices of every even variable and the odd variable that copies it.
        """
        return {
            frozenset((index, index + 1))
            for index in range(0, self.number_of_variables, 2)
        }

    def rows(self) -> np.ndarray:
        """
        :return: The rows, as the hashes of their values.
        """
        generator = np.random.default_rng(self.seed)
        component = generator.integers(0, 3, self.number_of_rows)
        bits = (
            generator.random((self.number_of_rows, self.number_of_variables))
            < np.array([0.15, 0.5, 0.85])[component][:, None]
        ).astype(int)
        copies = (
            generator.random((self.number_of_rows, self.number_of_variables // 2)) < 0.9
        )
        bits[:, 1::2] = np.where(copies, bits[:, ::2], 1 - bits[:, ::2])
        return self.hashes_of_bits(bits=bits)

    def every_assignment(self) -> np.ndarray:
        """
        :return: Every assignment of the variables, as the hashes of their values.
        """
        bits = np.array(
            np.meshgrid(*[[0, 1]] * self.number_of_variables, indexing="ij")
        ).reshape(self.number_of_variables, -1)
        return self.hashes_of_bits(bits=bits.T)

    @staticmethod
    def hashes_of_bits(*, bits: np.ndarray) -> np.ndarray:
        """
        :param bits: Rows of zeros and ones.
        :return: The rows with every bit replaced by the hash of its symbol.
        """
        return np.array([hash("0"), hash("1")], dtype=float)[bits]
