"""
Stacking copies of one layer graph that differ only in their parameters, such as the
copies a structural pass makes of a circuit for several points, into one layer graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

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
        LeafLayer,
    )

# %% stacked layers


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
    def of_leaf_layer_copies(cls, copies: list[LeafLayer]) -> StackedLayer:
        """
        :param copies: The copies of a leaf layer.
        :return: The copies as one layer, shared if they are all equal.
        """
        first = copies[0]
        if all(copy.has_equal_parameters(first) for copy in copies[1:]):
            return cls.shared(first)
        return cls(type(first).concatenate(copies), first.number_of_nodes, False)

    @classmethod
    def of_inner_layer_copies(
        cls, copies: list[InnerLayer], stacked_child_layers: list[StackedLayer]
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
        copy: int, edges: InnerLayerEdges, stacked_child_layers: list[StackedLayer]
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


# %% stacking


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

    stacked_by_first_copy: dict[Layer, StackedLayer] = field(default_factory=dict)
    """
    The result for every layer stacked so far, by its copy in the first graph.

    The copies are aligned, so the copy in the first graph determines the copies in all
    other graphs: they are the layers at the same position. Keying by it alone lets a
    layer with several parents be stacked once, however it is reached.
    """

    def stack(self, copies: list[Layer], path: tuple[int, ...] = ()) -> StackedLayer:
        """
        :param copies: The copies of a layer, one per copy of the graph.
        :param path: The index of the child layer taken at every step from the roots
            of the copies down to ``copies``, to report where copies are not aligned.
        :return: The copies, and the copies of everything below them, as one layer.
        :raises CopiesNotAlignedError: If the copies differ in their structure.
        """
        first_copy = copies[0]
        if first_copy not in self.stacked_by_first_copy:
            self.stacked_by_first_copy[first_copy] = self.stack_unseen(copies, path)
        return self.stacked_by_first_copy[first_copy]

    def stack_unseen(self, copies: list[Layer], path: tuple[int, ...]) -> StackedLayer:
        """
        :param copies: The copies of a layer that was not stacked yet.
        :param path: The path of :meth:`stack` to the copies.
        :return: The copies as one layer.
        """
        first = copies[0]
        if any(
            type(copy) is not type(first)
            or copy.number_of_nodes != first.number_of_nodes
            or len(copy.child_layers) != len(first.child_layers)
            for copy in copies
        ):
            raise CopiesNotAlignedError(tuple(copies), path)
        stacked_child_layers = [
            self.stack([copy.child_layers[index] for copy in copies], (*path, index))
            for index in range(len(first.child_layers))
        ]
        return type(first).stacked(copies, stacked_child_layers)
