from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from random_events.variable import Symbolic
from typing_extensions import Any, Tuple

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    SampleColumn,
    StateIndices,
)


@dataclass
class SymbolicEncoding:
    """
    The translation between the two representations of a value of a symbolic variable.

    The events and samples of this package hold the hash of a domain element. A symbolic
    layer holds the position of the element in the domain instead, a small integer that
    it looks up without hashing. The layered circuit encodes its input once per query
    and decodes its samples.
    """

    variable: Symbolic
    """
    The symbolic variable.
    """

    @functools.cached_property
    def elements(self) -> Tuple[Any, ...]:
        """
        :return: The domain elements of the variable, in the order that defines their
            positions.
        """
        return tuple(
            simple_set.element for simple_set in self.variable.domain.simple_sets
        )

    @functools.cached_property
    def hashes(self) -> SampleColumn:
        """
        :return: The hash of every domain element, at its position.
        """
        return np.array([hash(element) for element in self.elements], dtype=float)

    def indices_of_hashes(self, values: SampleColumn) -> StateIndices:
        """
        :param values: Values of the variable as the events of this package hold them.
        :return: The position of every value in the domain, or ``-1`` for a value that
            is not the hash of a domain element.
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        order = np.argsort(self.hashes)
        sorted_hashes = self.hashes[order]
        positions = np.clip(np.searchsorted(sorted_hashes, values), 0, len(order) - 1)
        found = sorted_hashes[positions] == values
        return np.where(found, order[positions], -1)

    def hashes_of_indices(self, indices: StateIndices) -> SampleColumn:
        """
        :param indices: Positions of domain elements.
        :return: The value of every position as the events of this package hold it.
        """
        return self.hashes[np.asarray(indices, dtype=np.int64)]

    def index_of_element(self, element: Any) -> int:
        """
        :param element: A domain element, or its hash.
        :return: The position of the element in the domain, or ``-1``.
        """
        return int(self.indices_of_hashes(np.array([hash(element)], dtype=float))[0])
