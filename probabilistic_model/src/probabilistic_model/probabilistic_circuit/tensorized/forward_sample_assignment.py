from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, Dict, Iterable, List, Self

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    SampleRows,
)

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
        Layer,
    )


@dataclass
class SampleRowsOfNode:
    """
    The rows of the sample array that one node has to fill.

    A node with several parents receives one chunk of rows from each of them.
    """

    chunks: List[SampleRows] = field(default_factory=list)
    """
    The chunks of rows received so far.
    """

    def add(self, rows: SampleRows):
        """
        :param rows: Rows the node has to fill as well.
        """
        self.chunks.append(rows)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def rows(self) -> SampleRows:
        """
        :return: All rows the node has to fill.
        """
        return np.concatenate(self.chunks)


@dataclass
class ForwardSampleAssignment:
    """
    Bookkeeping for a top-down sampling pass over a circuit.

    A layer routes the output rows assigned to each of its nodes to the nodes of its
    child layers; a child layer that is shared by several parents accumulates rows from
    each of them before it is its own turn to route them further.
    """

    rows_by_layer: Dict[int, List[SampleRowsOfNode]]
    """
    For every layer, keyed by its id, the rows assigned to each of its nodes so far.
    """

    @classmethod
    def for_layers(cls, layers: Iterable[Layer]) -> Self:
        """
        :param layers: Every layer that will be visited during the pass.
        :return: An assignment without any rows for every node of every layer.
        """
        return cls(
            {
                id(layer): [SampleRowsOfNode() for _ in range(layer.number_of_nodes)]
                for layer in layers
            }
        )

    def assign(self, layer: Layer, node: int, rows: SampleRows):
        """
        Route output rows to one node of a layer.

        :param layer: The layer the node belongs to.
        :param node: The index of the node within that layer.
        :param rows: The output rows drawn from that node.
        """
        self.rows_by_layer[id(layer)][node].add(rows)

    def rows_of(self, layer: Layer) -> List[SampleRowsOfNode]:
        """
        :param layer: The layer to read the assignment of.
        :return: The rows assigned to every node of that layer so far.
        """
        return self.rows_by_layer[id(layer)]
