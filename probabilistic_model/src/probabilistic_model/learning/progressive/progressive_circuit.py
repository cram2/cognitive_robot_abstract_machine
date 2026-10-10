"""
The structure of a progressive probabilistic circuit: its columns and how they connect.
"""

from __future__ import annotations

# %% imports
import copy
import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from probabilistic_model.learning.progressive.exceptions import (
    ColumnStructureMismatchError,
    UnregisteredColumnError,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    Unit,
)


# %% circuit column
@dataclass
class CircuitColumn:
    """
    The copy of the template that models one task inside a progressive circuit.

    .. note::

        The column records its units when it is created, so it must be created before
        edges from it to other columns exist.
    """

    task_name: str
    """
    Readable name of the task.
    """

    root: Unit
    """
    Root unit of the column.
    """

    unit_indices: frozenset[int] = field(init=False)
    """
    Indices of the units the column owns.
    """

    sample_count: int = field(default=0, init=False)
    """
    Number of rows the column was last learned from;
    :attr:`ProgressiveProbabilisticCircuit.root` weights the column by its share of
    these rows.
    """

    def __post_init__(self):
        """
        Record the units the column owns: its root and every unit below it.
        """
        descendants = self.root.probabilistic_circuit.descendants(self.root)
        self.unit_indices = frozenset(
            [self.root.index] + [unit.index for unit in descendants]
        )

    def contains(self, unit: Unit) -> bool:
        """
        Check whether a unit belongs to the column.

        :param unit: The unit to check.
        :return: Whether the column owns the unit.
        """
        return unit.index in self.unit_indices

    def children_of(self, unit: Unit) -> list[Unit]:
        """
        Get the children of a unit inside the column.

        :param unit: A unit the column owns.
        :return: The children of the unit that the column owns, without those in other
            columns.
        """
        return [child for child in unit.subcircuits if self.contains(child)]


# %% aligned units
@dataclass(frozen=True)
class AlignedUnits:
    """
    Two units at the same structural position in two columns.
    """

    left: Unit
    """
    The unit of the first column.
    """

    right: Unit
    """
    The unit of the second column.
    """

    def match(self) -> bool:
        """
        Check whether the two units have the same structure.

        :return: Whether both units have the same type and model the same variables.
        """
        return type(self.left) is type(self.right) and tuple(
            self.left.variables
        ) == tuple(self.right.variables)


# %% progressive probabilistic circuit
@dataclass
class ProgressiveProbabilisticCircuit:
    """
    A probabilistic circuit that learns tasks one after another, one column per task, as
    progressive neural networks do.

    Every column is a copy of :attr:`template`. Each sum unit of a new column also mixes
    the aligned sum units of every earlier column, so the new column can reuse them
    while they stay unchanged.
    """

    template: ProbabilisticCircuit
    """
    Circuit every column is copied from, structure and initial parameters.
    """

    earlier_column_share: float = 0.5
    """
    Share of the start weight of every sum unit of a new column that goes to the aligned
    sum units of the earlier columns, split equally among them; the column's own
    children keep the rest in the proportions of :attr:`template`.
    """

    columns: list[CircuitColumn] = field(default_factory=list, init=False)
    """
    The columns, oldest first.
    """

    circuit: ProbabilisticCircuit = field(init=False)
    """
    The circuit holding :attr:`root` and every column.
    """

    root: SumUnit = field(init=False)
    """
    Sum unit mixing the roots of all columns.
    """

    def __post_init__(self):
        """
        Create the empty circuit and its root, which the columns are added below.
        """
        self.circuit = ProbabilisticCircuit()
        self.root = SumUnit(probabilistic_circuit=self.circuit)

    def validate_column(self, column: CircuitColumn) -> None:
        """
        Check that a column was added to this progressive circuit.

        :param column: The column to check.
        :raises UnregisteredColumnError: If the column belongs to another progressive
            circuit.
        """
        if column not in self.columns:
            raise UnregisteredColumnError(column)

    def add_column(self, task_name: str) -> CircuitColumn:
        """
        Add a column for a new task, connected to every earlier column.

        Every sum unit of the new column also mixes the aligned sum unit of each earlier
        column, starting with :attr:`earlier_column_share` of its weight. The circuit
        stays unchanged if the template no longer matches the earlier columns.

        :param task_name: Readable name of the task.
        :return: The new column.
        :raises ColumnStructureMismatchError: If the template differs in structure from
            the earlier columns.
        """
        template_copy = copy.deepcopy(self.template)
        aligned_sum_units = self._aligned_sum_units(
            CircuitColumn(task_name=task_name, root=template_copy.root)
        )
        mounted_units = self.circuit.mount(template_copy.root)
        column = CircuitColumn(
            task_name=task_name, root=mounted_units[template_copy.root.index]
        )
        self._connect_to_earlier_columns(
            [
                AlignedUnits(mounted_units[aligned.left.index], aligned.right)
                for aligned in aligned_sum_units
            ]
        )
        self.columns.append(column)
        self.root.add_subcircuit(column.root, log_weight=0.0)
        self.weight_root_by_sample_count()
        return column

    def _connect_to_earlier_columns(
        self, aligned_sum_units: list[AlignedUnits]
    ) -> None:
        """
        Make every earlier sum unit a child of its aligned sum unit of the new column,
        with :attr:`earlier_column_share` of the start weight split equally among the
        earlier columns.

        :param aligned_sum_units: The sum units of the new column, already in the
            circuit, paired with the aligned sum unit of each earlier column.
        """
        if not aligned_sum_units:
            return
        own_log_share = math.log(1 - self.earlier_column_share)
        for sum_unit in {aligned.left for aligned in aligned_sum_units}:
            sum_unit.normalize()
            for log_weight, child in sum_unit.log_weighted_subcircuits:
                self.circuit.add_edge(
                    sum_unit, child, log_weight=log_weight + own_log_share
                )
        earlier_column_log_weight = math.log(
            self.earlier_column_share / len(self.columns)
        )
        for aligned in aligned_sum_units:
            self.circuit.add_edge(
                aligned.left, aligned.right, log_weight=earlier_column_log_weight
            )

    def _aligned_sum_units(self, column: CircuitColumn) -> list[AlignedUnits]:
        """
        Pair every sum unit of a column with the aligned sum unit of each earlier
        column, earliest column first.

        :param column: A column that is not part of the circuit yet.
        :return: The pairs, the unit of ``column`` on the left.
        :raises ColumnStructureMismatchError: If the column differs in structure from an
            earlier column.
        """
        return [
            aligned
            for earlier_column in self.columns
            for aligned in self.aligned_units(column, earlier_column)
            if isinstance(aligned.left, SumUnit)
        ]

    def restrict_root_to(self, column: CircuitColumn) -> None:
        """
        Give the whole weight of :attr:`root` to one column.

        :param column: The column that gets the whole weight.
        :raises UnregisteredColumnError: If the column belongs to another progressive
            circuit.
        """
        self.validate_column(column)
        for other_column in self.columns:
            self.circuit.add_edge(
                self.root,
                other_column.root,
                log_weight=0.0 if other_column is column else -math.inf,
            )

    def weight_root_by_sample_count(self) -> None:
        """
        Weight every column below :attr:`root` by its share of all rows the columns were
        learned from, so columns that were not learned get no weight.

        Before any column was learned, every column gets the same weight.
        """
        total_sample_count = sum(column.sample_count for column in self.columns)
        if total_sample_count == 0:
            for column in self.columns:
                self.circuit.add_edge(self.root, column.root, log_weight=0.0)
            self.root.normalize()
            return
        for column in self.columns:
            log_weight = (
                math.log(column.sample_count / total_sample_count)
                if column.sample_count > 0
                else -math.inf
            )
            self.circuit.add_edge(self.root, column.root, log_weight=log_weight)

    def aligned_units(
        self, left: CircuitColumn, right: CircuitColumn
    ) -> Iterator[AlignedUnits]:
        """
        Walk two columns in parallel, without following edges between columns.

        :param left: The first column.
        :param right: The second column.
        :return: The units at matching positions, starting with the roots.
        :raises ColumnStructureMismatchError: If the columns differ in structure.
        """
        queue: deque[AlignedUnits] = deque([AlignedUnits(left.root, right.root)])
        visited: set[AlignedUnits] = set()
        while queue:
            aligned = queue.popleft()
            if aligned in visited:
                continue
            visited.add(aligned)
            left_children = left.children_of(aligned.left)
            right_children = right.children_of(aligned.right)
            if not aligned.match() or len(left_children) != len(right_children):
                raise ColumnStructureMismatchError(aligned.left, aligned.right)
            yield aligned
            queue.extend(
                AlignedUnits(left_child, right_child)
                for left_child, right_child in zip(left_children, right_children)
            )

    def units_of(self, column: CircuitColumn) -> set[Unit]:
        """
        Get the units of a column from the circuit.

        :param column: A column of this progressive circuit.
        :return: The units the column owns.
        """
        return {self.circuit.graph[index] for index in column.unit_indices}
