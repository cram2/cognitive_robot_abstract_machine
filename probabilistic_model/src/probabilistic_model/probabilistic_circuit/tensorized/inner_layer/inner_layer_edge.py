from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import Iterator

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
)


@dataclass
class InnerLayerEdge:
    """
    One edge of an inner layer: a node of that layer, and the node of one of its child
    layers that it points at.
    """

    node: int
    """
    The index of the node inside the layer the edge belongs to.
    """

    child_layer_index: int
    """
    The index of the child layer the edge points into, within
    :attr:`InnerLayer.child_layers`.
    """

    child_node: int
    """
    The index of the node inside that child layer.
    """


@dataclass
class InnerLayerEdges:
    """
    Several edges of an inner layer, stored as one array per field of
    :class:`InnerLayerEdge`.

    The ``k``-th entry of every array belongs to the ``k``-th edge.
    """

    nodes: NodeIndices
    """
    The node of every edge, see :attr:`InnerLayerEdge.node`.
    """

    child_layer_indices: npt.NDArray[np.int64]
    """
    The child layer of every edge, see :attr:`InnerLayerEdge.child_layer_index`.
    """

    child_nodes: NodeIndices
    """
    The child node of every edge, see :attr:`InnerLayerEdge.child_node`.
    """

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[InnerLayerEdge]:
        for node, child_layer_index, child_node in zip(
            self.nodes, self.child_layer_indices, self.child_nodes
        ):
            yield InnerLayerEdge(int(node), int(child_layer_index), int(child_node))
