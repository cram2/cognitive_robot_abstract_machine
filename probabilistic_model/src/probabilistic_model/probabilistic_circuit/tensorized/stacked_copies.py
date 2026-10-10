"""
Stacking copies of one layer graph that differ only in their parameters, such as the
copies a structural pass makes of a circuit for several points, into one layer graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, Dict, List

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    CopiesNotAlignedError,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdges,
)

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
        InnerLayer,
        Layer,
    )


@dataclass
class StackedLayer:
    """
    A layer that holds the nodes of every copy of one layer.

    Copy ``c`` occupies the block of nodes ``c * nodes_per_copy`` up to ``(c + 1) *
    nodes_per_copy``, unless all copies are equal, in which case they share one block.
    """

    layer: Layer
    """
    The layer.
    """

    nodes_per_copy: int
    """
    The number of nodes of one copy.
    """

    is_shared: bool
    """
    Whether all copies are equal and share one block of nodes.
    """

    @classmethod
    def shared(cls, layer: Layer) -> StackedLayer:
        """
        :param layer: A layer that every copy has unchanged.
        :return: The layer, shared by all copies.
        """
        return cls(layer, layer.number_of_nodes, True)

    def nodes_of_copy(self, copy: int | NodeIndices, nodes: NodeIndices) -> NodeIndices:
        """
        :param copy: The index of a copy, or of the copy of every node.
        :param nodes: Nodes of that copy, as indices into one copy of the layer.
        :return: The same nodes, as indices into this layer.
        """
        if self.is_shared:
            return nodes
        return nodes + copy * self.nodes_per_copy

    @classmethod
    def of_input_layer_copies(cls, copies: List[Layer]) -> StackedLayer:
        """
        :param copies: The copies of a layer without child layers.
        :return: The copies as one layer, shared if they are all equal.
        """
        first = copies[0]
        parameters = first.to_json()
        if all(copy.to_json() == parameters for copy in copies[1:]):
            return cls.shared(first)
        return cls(type(first).concatenate(copies), first.number_of_nodes, False)

    @classmethod
    def of_inner_layer_copies(
        cls, copies: List[InnerLayer], stacked_child_layers: List[StackedLayer]
    ) -> StackedLayer:
        """
        :param copies: The copies of an inner layer, one per copy of the graph.
        :param stacked_child_layers: The stacked copies of each of its child layers.
        :return: The copies as one layer, shared if they and their children are equal.
        """
        first = copies[0]
        if all(child.is_shared for child in stacked_child_layers) and all(
            copy.has_equal_edges(first) for copy in copies[1:]
        ):
            return cls.shared(
                first.with_edges(
                    [child.layer for child in stacked_child_layers],
                    first.inner_layer_edges,
                    [first],
                )
            )

        edges = [copy.inner_layer_edges for copy in copies]
        stacked_edges = InnerLayerEdges(
            np.concatenate(
                [
                    copy_edges.nodes + index * first.number_of_nodes
                    for index, copy_edges in enumerate(edges)
                ]
            ),
            np.concatenate([copy_edges.child_layer_indices for copy_edges in edges]),
            np.concatenate(
                [
                    cls.child_nodes_of_copy(index, copy_edges, stacked_child_layers)
                    for index, copy_edges in enumerate(edges)
                ]
            ),
        )
        layer = first.with_edges(
            [child.layer for child in stacked_child_layers], stacked_edges, copies
        )
        return cls(layer, first.number_of_nodes, False)

    @staticmethod
    def child_nodes_of_copy(
        copy: int, edges: InnerLayerEdges, stacked_child_layers: List[StackedLayer]
    ) -> NodeIndices:
        """
        :param copy: The index of a copy.
        :param edges: The edges of that copy of an inner layer.
        :param stacked_child_layers: The stacked copies of each child layer.
        :return: The child node of every edge, as an index into its stacked child layer.
        """
        result = edges.child_nodes.copy()
        for child_layer_index, child in enumerate(stacked_child_layers):
            into_child = edges.child_layer_indices == child_layer_index
            result[into_child] = child.nodes_of_copy(copy, result[into_child])
        return result


@dataclass
class AlignedCopiesStacker:
    """
    Stacks copies of one layer graph into one layer graph whose layers hold the nodes of
    every copy.

    The copies must share their structure: the same types of layers, connected the same
    way, with the same number of nodes. They may differ in their parameters, as the
    copies do that a structural pass makes of one circuit without pruning it. A layer
    that is equal in every copy is shared rather than repeated.
    """

    stacked_layers: Dict[int, StackedLayer] = field(default_factory=dict)
    """
    The result for every layer stacked so far, by the identity of its first copy, so
    that a layer with several parents is stacked once.
    """

    def stack(self, copies: List[Layer]) -> StackedLayer:
        """
        :param copies: The copies of a layer, one per copy of the graph.
        :return: The copies, and the copies of everything below them, as one layer.
        :raises CopiesNotAlignedError: If the copies differ in their structure.
        """
        key = id(copies[0])
        if key not in self.stacked_layers:
            self.stacked_layers[key] = self.stack_unseen(copies)
        return self.stacked_layers[key]

    def stack_unseen(self, copies: List[Layer]) -> StackedLayer:
        """
        :param copies: The copies of a layer that was not stacked yet.
        :return: The copies as one layer.
        """
        first = copies[0]
        if any(
            type(copy) is not type(first)
            or copy.number_of_nodes != first.number_of_nodes
            or len(copy.child_layers) != len(first.child_layers)
            for copy in copies
        ):
            raise CopiesNotAlignedError(
                tuple(type(copy) for copy in copies),
                tuple(copy.number_of_nodes for copy in copies),
            )
        stacked_child_layers = [
            self.stack([copy.child_layers[index] for copy in copies])
            for index in range(len(first.child_layers))
        ]
        return type(first).stacked(copies, stacked_child_layers)
