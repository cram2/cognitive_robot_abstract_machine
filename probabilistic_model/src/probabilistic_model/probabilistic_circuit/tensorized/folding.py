"""
Folding a layered circuit: every layer of one type at one height becomes one layer, so
that a circuit such as a RAT-SPN computes in a handful of batched operations.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import numpy as np
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
    Layer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
    SparseEdgeMatrix,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdges,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    UnivariateInputLayer,
)

# %% folded layers


@dataclass(kw_only=True)
class FoldedLayer:
    """
    The layer that layers of a circuit were folded into, and where their nodes went.
    """

    layer: Layer
    """
    The folded layer.
    """

    node_positions: dict[int, NodeIndices]
    """
    The node of :attr:`layer` that every node of every folded layer became, keyed by the
    :func:`id` of the folded layer.
    """


# %% folding


@dataclass(kw_only=True)
class LayerFolding:
    """
    Folds the layers of a circuit with host parameters: the univariate input layers of
    one type, the product layers and the sum layers, each at one height, become one
    layer each, the sums storing their edges as a :class:`BlockEdgeMatrix`.

    Any other layer keeps a layer of its own. The folded circuit computes what the
    circuit computes.
    """

    root: Layer
    """
    The root of the circuit to fold.
    """

    heights: dict[int, int] = field(init=False, default_factory=dict)
    """
    The longest number of layers from every layer down to a leaf, keyed by the
    :func:`id` of the layer.
    """

    folded_by_layer: dict[int, FoldedLayer] = field(init=False, default_factory=dict)
    """
    The folded layer of every layer, keyed by the :func:`id` of the layer.
    """

    def __post_init__(self) -> None:
        """
        Compute the height of every layer, children first.
        """
        for layer in reversed(self.root.all_layers()):
            self.heights[id(layer)] = 1 + max(
                (self.heights[id(child)] for child in layer.child_layers), default=-1
            )

    def folded_root(self) -> Layer:
        """
        :return: The root of the folded circuit, whose node ``i`` is node ``i`` of the
            root.
        """
        groups: dict[tuple[int, type | int], list[Layer]] = {}
        for layer in reversed(self.root.all_layers()):
            foldable = isinstance(layer, (UnivariateInputLayer, SumLayer, ProductLayer))
            kind = type(layer) if foldable else id(layer)
            groups.setdefault((self.heights[id(layer)], kind), []).append(layer)
        # every child is lower than its parents, so folding by height folds it first
        for key in sorted(groups, key=lambda group_key: group_key[0]):
            layers = groups[key]
            folded = self.fold(layers=layers)
            for layer in layers:
                self.folded_by_layer[id(layer)] = folded
        return self.folded_by_layer[id(self.root)].layer

    def fold(self, *, layers: list[Layer]) -> FoldedLayer:
        """
        :param layers: Layers of one type at one height, whose children are folded.
        :return: The layers as one layer.
        """
        first = layers[0]
        if isinstance(first, UnivariateInputLayer):
            return self.fold_input_layers(layers=layers)
        if isinstance(first, InnerLayer):
            return self.fold_inner_layers(layers=layers)
        return FoldedLayer(
            layer=first, node_positions={id(first): np.arange(first.number_of_nodes)}
        )

    @staticmethod
    def fold_input_layers(*, layers: list[UnivariateInputLayer]) -> FoldedLayer:
        """
        :param layers: Univariate input layers of one type.
        :return: Their nodes as one layer, sorted by variable.
        """
        joined = type(layers[0]).concatenate(layers)
        order = np.argsort(joined.node_variables, kind="stable")
        position_of_joined = np.argsort(order)
        offsets = np.cumsum([0] + [layer.number_of_nodes for layer in layers])
        return FoldedLayer(
            layer=joined.select_nodes(order),
            node_positions={
                id(layer): position_of_joined[offset : offset + layer.number_of_nodes]
                for layer, offset in zip(layers, offsets)
            },
        )

    def fold_inner_layers(self, *, layers: list[InnerLayer]) -> FoldedLayer:
        """
        :param layers: Sum layers or product layers, whose children are folded.
        :return: Their nodes as one layer, the nodes of ``layers[k]`` after those of the
            layers before it.
        """
        child_layers: list[Layer] = []
        index_of_child_layer: dict[int, int] = {}
        nodes, child_layer_indices, child_nodes, log_weights = [], [], [], []
        offset = 0
        node_positions = {}
        for layer in layers:
            edges = layer.inner_layer_edges
            child_index_of_edge = np.empty(len(edges.nodes), dtype=np.int64)
            child_node_of_edge = np.empty(len(edges.nodes), dtype=np.int64)
            for index, child in enumerate(layer.child_layers):
                folded_child = self.folded_by_layer[id(child)]
                key = id(folded_child.layer)
                if key not in index_of_child_layer:
                    index_of_child_layer[key] = len(child_layers)
                    child_layers.append(folded_child.layer)
                of_child = edges.child_layer_indices == index
                child_index_of_edge[of_child] = index_of_child_layer[key]
                child_node_of_edge[of_child] = folded_child.node_positions[id(child)][
                    edges.child_nodes[of_child]
                ]
            nodes.append(edges.nodes + offset)
            child_layer_indices.append(child_index_of_edge)
            child_nodes.append(child_node_of_edge)
            if isinstance(layer, SumLayer):
                log_weights.append(np.asarray(layer.edge_log_weights))
            node_positions[id(layer)] = np.arange(layer.number_of_nodes) + offset
            offset += layer.number_of_nodes

        folded_edges = InnerLayerEdges(
            np.concatenate(nodes),
            np.concatenate(child_layer_indices),
            np.concatenate(child_nodes),
        )
        if isinstance(layers[0], ProductLayer):
            layer = ProductLayer.from_edges(
                child_layers, folded_edges, offset, SparseEdgeMatrix
            )
        else:
            column_offsets = np.cumsum(
                [0] + [child.number_of_nodes for child in child_layers]
            )
            layer = SumLayer.from_entries(
                child_layers,
                SparseEntries(
                    np.concatenate(log_weights),
                    folded_edges.nodes,
                    column_offsets[folded_edges.child_layer_indices]
                    + folded_edges.child_nodes,
                ),
                offset,
                BlockEdgeMatrix,
            )
        return FoldedLayer(layer=layer, node_positions=node_positions)
