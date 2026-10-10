from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_array, csr_array, vstack
from typing_extensions import Generic, List, Self, Tuple, TypeVar

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
    NodeStateValues,
    NodeValues,
    SampleNodeValues,
    StateIndices,
    StateMask,
    StateValues,
    TableEntryValues,
)

StoredLogProbabilities = TypeVar("StoredLogProbabilities")
"""
The representation a probability table stores its log-probabilities in.
"""


class ProbabilityTable(ABC, Generic[StoredLogProbabilities]):
    """
    The probability of every state for every node of a discrete layer, a table of shape
    (#nodes, #states).

    A discrete layer answers every query through the operations of its table. A table is not changed after it was created; every operation that changes the
    probabilities returns a new table of the same type.
    """

    log_probabilities: StoredLogProbabilities
    """
    The log-probabilities of the table. Every type of table declares the representation
    it stores them in.
    """

    # %% construction

    @classmethod
    @abstractmethod
    def from_entries(
        cls,
        rows: NodeIndices,
        columns: StateIndices,
        probabilities: TableEntryValues,
        shape: Tuple[int, int],
    ) -> Self:
        """
        :param rows: The node of every entry.
        :param columns: The state of every entry.
        :param probabilities: The probability of every entry.
        :param shape: The number of nodes and the number of states.
        :return: The table that holds these entries and zero everywhere else.
        """
        raise NotImplementedError

    @abstractmethod
    def entries(
        self,
    ) -> Tuple[NodeIndices, StateIndices, TableEntryValues]:
        """
        :return: The node, the state and the probability of every non-zero entry.
        """
        raise NotImplementedError

    @classmethod
    def of(cls, table: ProbabilityTable) -> Self:
        """
        :param table: A table of any type.
        :return: A table of this type with the same probabilities.
        """
        return cls.from_entries(
            *table.entries(), (table.number_of_nodes, table.number_of_states)
        )

    # %% shape

    @property
    @abstractmethod
    def number_of_nodes(self) -> int:
        """
        :return: The number of nodes, the rows of the table.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def number_of_states(self) -> int:
        """
        :return: The number of states, the columns of the table.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def number_of_stored_entries(self) -> int:
        """
        :return: The number of probabilities this table stores.
        """
        raise NotImplementedError

    # %% queries

    @abstractmethod
    def log_probabilities_of_states(self, indices: StateIndices) -> SampleNodeValues:
        """
        :param indices: One state index per value, ``-1`` for a value that is not a
            state.
        :return: The log-probability of every value under every node, with shape
            (#values, #nodes).
        """
        raise NotImplementedError

    @abstractmethod
    def probability_of_states(self, selected: StateMask) -> NodeValues:
        """
        :param selected: The states to sum over.
        :return: The probability of the selected states under every node.
        """
        raise NotImplementedError

    @abstractmethod
    def dot(self, values: StateValues) -> NodeValues:
        """
        :param values: The values of the states.
        :return: The product of the table with the values, the expectation of the
            values under every node, with shape (#nodes,), or (#nodes, k) for k
            values per state.
        """
        raise NotImplementedError

    @abstractmethod
    def probabilities_of_node(self, node: int) -> Tuple[StateIndices, TableEntryValues]:
        """
        :param node: The index of a node.
        :return: The states with a non-zero probability under the node, ascending, and
            their probabilities.
        """
        raise NotImplementedError

    @abstractmethod
    def dense_probabilities(self) -> NodeStateValues:
        """
        :return: The whole table as a dense array. Meant for small tables and tests.
        """
        raise NotImplementedError

    # %% new tables

    @abstractmethod
    def restricted_to(self, selected: StateMask) -> Self:
        """
        :param selected: The states to keep.
        :return: The table with the probability of every other state set to zero,
            without renormalizing.
        """
        raise NotImplementedError

    @abstractmethod
    def normalized(self) -> Self:
        """
        :return: The table with the probabilities of every node summing to one. A node
            without any probability keeps none.
        """
        raise NotImplementedError

    @abstractmethod
    def select_nodes(self, mask: NodeMask) -> Self:
        """
        :param mask: The nodes to keep.
        :return: The table of the kept nodes.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, tables: List[Self]) -> Self:
        """
        :param tables: Tables with the same states.
        :return: One table with the nodes of all tables, in order.
        """
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> Self:
        """
        :return: A table with the same probabilities that shares no array with this one.
        """
        raise NotImplementedError

    @abstractmethod
    def has_equal_probabilities(self, other: Self) -> bool:
        """
        :param other: A table of the same type and shape.
        :return: Whether both tables hold the same probabilities.
        """
        raise NotImplementedError

    # %% sampling

    def cumulative_distribution_of_entries(
        self,
    ) -> Tuple[NodeIndices, StateIndices, TableEntryValues]:
        """
        :return: The node and the state of every non-zero entry, sorted by node and then
            by state, and the probability of the node for all states up to and including
            the state of the entry, divided by the total probability of the node.
        """
        rows, columns, probabilities = self.entries()
        if len(rows) == 0:
            return rows, columns, np.zeros(0)
        cumulative = np.cumsum(probabilities)
        first_of_row = np.r_[True, rows[1:] != rows[:-1]]
        # the first entry of the row of every entry, carried forward over the row
        start_of_row = np.maximum.accumulate(
            np.where(first_of_row, np.arange(len(rows)), 0)
        )
        cumulative_within_row = cumulative - (cumulative - probabilities)[start_of_row]
        totals = np.bincount(
            rows, weights=probabilities, minlength=self.number_of_nodes
        )
        return rows, columns, cumulative_within_row / totals[rows]

    def sample_state_indices(self, nodes: NodeIndices) -> StateIndices:
        """
        Draw one state per entry of ``nodes`` from the node of the entry, all at once.

        Every row is turned into its cumulative distribution shifted by its node index,
        so that all rows form one ascending sequence; a uniform number plus the node
        index then finds the drawn state of that node with one search.

        :param nodes: The node to draw each state from.
        :return: The drawn state index per entry, ``-1`` for a node without mass.
        """
        nodes = np.asarray(nodes, dtype=np.int64)
        rows, columns, cumulative = self.cumulative_distribution_of_entries()
        if len(rows) == 0:
            return np.full(len(nodes), -1)
        shifted_cumulative = cumulative + rows
        # the last entry of every row reaches its end exactly, whatever the rounding
        last_of_row = np.r_[rows[1:] != rows[:-1], True]
        shifted_cumulative[last_of_row] = rows[last_of_row] + 1.0

        targets = nodes + np.random.uniform(0.0, 1.0, len(nodes))
        entry = np.searchsorted(shifted_cumulative, targets, side="right")
        # rounding must not carry a draw into the next row
        last_entry_of_node = np.searchsorted(rows, nodes, side="right") - 1
        entry = np.minimum(entry, last_entry_of_node)
        has_mass = np.bincount(rows, minlength=self.number_of_nodes)[nodes] > 0
        return np.where(has_mass, columns[np.maximum(entry, 0)], -1)


@dataclass(eq=False)
class DenseProbabilityTable(ProbabilityTable[NodeStateValues]):
    """
    A probability table stored as one dense array of log-probabilities.

    A likelihood is a single gather from the array, so this is the fastest table for
    looking up the probabilities of events. Use it when the variable has few states or
    the nodes have most of them.
    """

    log_probabilities: NodeStateValues
    """
    The log-probability of every state for every node, ``-inf`` for probability zero.
    """

    @classmethod
    def from_entries(
        cls,
        rows: NodeIndices,
        columns: StateIndices,
        probabilities: TableEntryValues,
        shape: Tuple[int, int],
    ) -> Self:
        dense = np.zeros(shape)
        dense[rows, columns] = probabilities
        with np.errstate(divide="ignore"):
            return cls(np.log(dense))

    def entries(self) -> Tuple[NodeIndices, StateIndices, TableEntryValues]:
        rows, columns = np.nonzero(np.isfinite(self.log_probabilities))
        return rows, columns, np.exp(self.log_probabilities[rows, columns])

    @property
    def number_of_nodes(self) -> int:
        return self.log_probabilities.shape[0]

    @property
    def number_of_states(self) -> int:
        return self.log_probabilities.shape[1]

    @property
    def number_of_stored_entries(self) -> int:
        return int(self.log_probabilities.size)

    def log_probabilities_of_states(self, indices: StateIndices) -> SampleNodeValues:
        result = np.full((len(indices), self.number_of_nodes), -np.inf)
        known = indices >= 0
        if known.any():
            result[known] = self.log_probabilities[:, indices[known]].T
        return result

    def probability_of_states(self, selected: StateMask) -> NodeValues:
        return np.exp(self.log_probabilities[:, selected]).sum(axis=1)

    def dot(self, values: StateValues) -> NodeValues:
        return np.exp(self.log_probabilities) @ values

    def probabilities_of_node(self, node: int) -> Tuple[StateIndices, TableEntryValues]:
        [states] = np.nonzero(np.isfinite(self.log_probabilities[node]))
        return states, np.exp(self.log_probabilities[node, states])

    def dense_probabilities(self) -> NodeStateValues:
        return np.exp(self.log_probabilities)

    def restricted_to(self, selected: StateMask) -> Self:
        return self.__class__(np.where(selected, self.log_probabilities, -np.inf))

    def normalized(self) -> Self:
        # divide in linear space, like the scalar distributions do, so that both agree
        # to the last digit
        probabilities = np.exp(self.log_probabilities)
        totals = probabilities.sum(axis=1, keepdims=True)
        normalized = np.divide(
            probabilities,
            totals,
            out=np.zeros_like(probabilities),
            where=totals > 0,
        )
        with np.errstate(divide="ignore"):
            return self.__class__(np.log(normalized))

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(self.log_probabilities[mask])

    @classmethod
    def concatenate(cls, tables: List[Self]) -> Self:
        return cls(np.concatenate([table.log_probabilities for table in tables]))

    def copy(self) -> Self:
        return self.__class__(self.log_probabilities.copy())

    def has_equal_probabilities(self, other: Self) -> bool:
        return np.array_equal(self.log_probabilities, other.log_probabilities)


@dataclass(eq=False)
class SparseProbabilityTable(ProbabilityTable[csr_array]):
    """
    A probability table that stores only the non-zero probabilities, row by row.

    Operations over whole rows cost the number of stored entries instead of the size of
    the table. A likelihood densifies only the columns of the states that the events
    contain. Use it when the variable has many states and every node only a few of them,
    as in the leaves of a joint probability tree over an integer variable with a wide
    range.
    """

    log_probabilities: csr_array
    """
    The log-probability of every state with a non-zero probability, for every node.

    A state that is not stored has probability zero. A stored zero is the log-probability
    of a state with probability one.
    """

    @classmethod
    def from_entries(
        cls,
        rows: NodeIndices,
        columns: StateIndices,
        probabilities: TableEntryValues,
        shape: Tuple[int, int],
    ) -> Self:
        stored = probabilities > 0
        table = csr_array(
            (probabilities[stored], (rows[stored], columns[stored])), shape=shape
        )
        table.sum_duplicates()
        table.sort_indices()
        return cls(csr_array((np.log(table.data), table.indices, table.indptr), shape))

    def entries(self) -> Tuple[NodeIndices, StateIndices, TableEntryValues]:
        return (
            self.rows_of_entries,
            self.probabilities.indices.astype(np.int64),
            self.probabilities.data,
        )

    @functools.cached_property
    def probabilities(self) -> csr_array:
        """
        :return: The probability of every stored state, in linear space.
        """
        return csr_array(
            (
                np.exp(self.log_probabilities.data),
                self.log_probabilities.indices,
                self.log_probabilities.indptr,
            ),
            self.log_probabilities.shape,
        )

    @functools.cached_property
    def rows_of_entries(self) -> NodeIndices:
        """
        :return: The row of every stored entry.
        """
        return np.repeat(
            np.arange(self.number_of_nodes), np.diff(self.log_probabilities.indptr)
        )

    @functools.cached_property
    def by_column(self) -> csc_array:
        """
        :return: The table stored column by column, for looking up states.
        """
        return csc_array(self.probabilities)

    @property
    def number_of_nodes(self) -> int:
        return self.log_probabilities.shape[0]

    @property
    def number_of_states(self) -> int:
        return self.log_probabilities.shape[1]

    @property
    def number_of_stored_entries(self) -> int:
        return int(self.log_probabilities.nnz)

    def log_probabilities_of_states(self, indices: StateIndices) -> SampleNodeValues:
        result = np.full((len(indices), self.number_of_nodes), -np.inf)
        known = indices >= 0
        if known.any():
            # densify only the columns of the states that occur, once per state
            states, positions = np.unique(indices[known], return_inverse=True)
            with np.errstate(divide="ignore"):
                columns = np.log(self.by_column[:, states].toarray())
            result[known] = columns[:, positions].T
        return result

    def probability_of_states(self, selected: StateMask) -> NodeValues:
        rows, columns, probabilities = self.entries()
        return np.bincount(
            rows,
            weights=probabilities * selected[columns],
            minlength=self.number_of_nodes,
        )

    def dot(self, values: StateValues) -> NodeValues:
        return self.probabilities @ values

    def probabilities_of_node(self, node: int) -> Tuple[StateIndices, TableEntryValues]:
        start, end = self.probabilities.indptr[node : node + 2]
        return (
            self.probabilities.indices[start:end].astype(np.int64),
            self.probabilities.data[start:end].copy(),
        )

    def dense_probabilities(self) -> NodeStateValues:
        return self.probabilities.toarray()

    def restricted_to(self, selected: StateMask) -> Self:
        rows, columns, probabilities = self.entries()
        kept = selected[columns]
        return self.from_entries(
            rows[kept], columns[kept], probabilities[kept], self.log_probabilities.shape
        )

    def normalized(self) -> Self:
        rows, columns, probabilities = self.entries()
        totals = np.bincount(
            rows, weights=probabilities, minlength=self.number_of_nodes
        )
        return self.from_entries(
            rows, columns, probabilities / totals[rows], self.log_probabilities.shape
        )

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(csr_array(self.log_probabilities[np.flatnonzero(mask)]))

    @classmethod
    def concatenate(cls, tables: List[Self]) -> Self:
        return cls(csr_array(vstack([table.log_probabilities for table in tables])))

    def copy(self) -> Self:
        return self.__class__(self.log_probabilities.copy())

    def has_equal_probabilities(self, other: Self) -> bool:
        return (self.log_probabilities != other.log_probabilities).nnz == 0
