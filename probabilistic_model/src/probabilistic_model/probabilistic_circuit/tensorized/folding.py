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
class PositionsInFoldedLayer:
    """
    Where the nodes of one layer of a circuit went in the folded circuit.
    """

    folded_layer: Layer
    """
    The folded layer that holds them.
    """

    positions: NodeIndices
    """
    The node of :attr:`folded_layer` that every node of the original layer became.
    """


@dataclass(kw_only=True)
class LayersFoldedTogether:
    """
    Layers at one height that fold into one layer: the univariate input layers of one
    type, the product layers or the sum layers.

    Any other layer folds alone.
    """

    height: int
    """
    The longest number of layers from the layers down to a leaf.
    """

    layers: list[Layer]
    """
    The layers, at least one.
    """

    @property
    def is_foldable(self) -> bool:
        """
        :return: Whether layers other than the first may join these layers.
        """
        return isinstance(
            self.layers[0], (UnivariateInputLayer, SumLayer, ProductLayer)
        )

    def admits(self, *, layer: Layer, height: int) -> bool:
        """
        :param layer: A layer of the circuit.
        :param height: The height of the layer.
        :return: Whether the layer folds together with these layers.
        """
        return (
            self.is_foldable
            and height == self.height
            and type(layer) is type(self.layers[0])
        )


@dataclass(kw_only=True)
class FoldedChildLayers:
    """
    The folded layers that the children of inner layers folded together became, each
    once, in the order the inner layers first point to them.
    """

    positions_in_folded_layers: dict[Layer, PositionsInFoldedLayer]
    """
    Where the nodes of every folded layer of the circuit went.
    """

    layers: list[Layer] = field(default_factory=list)
    """
    The folded child layers.
    """

    def index_of(self, *, folded_child: Layer) -> int:
        """
        :param folded_child: A folded layer, added to :attr:`layers` if it is not one.
        :return: Its index in :attr:`layers`.
        """
        for index, layer in enumerate(self.layers):
            if layer is folded_child:
                return index
        self.layers.append(folded_child)
        return len(self.layers) - 1

    def edges_of(self, *, layer: InnerLayer, first_node: int) -> InnerLayerEdges:
        """
        :param layer: An inner layer folded with others, whose children are folded.
        :param first_node: The node of the folded layer the first node of ``layer``
            becomes.
        :return: The edges of ``layer`` in the folded layer, into :attr:`layers`.
        """
        edges = layer.inner_layer_edges
        child_layer_indices = np.empty_like(edges.child_layer_indices)
        child_nodes = np.empty_like(edges.child_nodes)
        for index, child in enumerate(layer.child_layers):
            child_positions = self.positions_in_folded_layers[child]
            of_child = edges.child_layer_indices == index
            child_layer_indices[of_child] = self.index_of(
                folded_child=child_positions.folded_layer
            )
            child_nodes[of_child] = child_positions.positions[
                edges.child_nodes[of_child]
            ]
        return InnerLayerEdges(
            edges.nodes + first_node, child_layer_indices, child_nodes
        )

    @property
    def column_offsets(self) -> NodeIndices:
        """
        :return: The column of the first node of every layer of :attr:`layers` in the
            edge matrix of a sum layer over them.
        """
        return np.cumsum([0] + [layer.number_of_nodes for layer in self.layers])


# %% folding


@dataclass(kw_only=True)
class LayerFolding:
    """
    Folds the layers of a circuit with host parameters: the layers of every
    :class:`LayersFoldedTogether` become one layer, the sums storing their edges as a
    :class:`BlockEdgeMatrix`.

    The folded circuit computes what the circuit computes.
    """

    root: Layer
    """
    The root of the circuit to fold.
    """

    heights: dict[Layer, int] = field(init=False, default_factory=dict)
    """
    The longest number of layers from every layer down to a leaf.
    """

    positions_in_folded_layers: dict[Layer, PositionsInFoldedLayer] = field(
        init=False, default_factory=dict
    )
    """
    Where the nodes of every layer folded so far went.
    """

    def __post_init__(self) -> None:
        """
        Compute the height of every layer, children first.
        """
        for layer in reversed(self.root.all_layers()):
            self.heights[layer] = 1 + max(
                (self.heights[child] for child in layer.child_layers), default=-1
            )

    @property
    def layer_groups(self) -> list[LayersFoldedTogether]:
        """
        :return: The layers of the circuit, split into those folded together, lowest
            first, so that every child is folded before its parents.
        """
        groups: list[LayersFoldedTogether] = []
        for layer in reversed(self.root.all_layers()):
            height = self.heights[layer]
            group = next(
                (group for group in groups if group.admits(layer=layer, height=height)),
                None,
            )
            if group is None:
                groups.append(LayersFoldedTogether(height=height, layers=[layer]))
            else:
                group.layers.append(layer)
        return sorted(groups, key=lambda group: group.height)

    def folded_root(self) -> Layer:
        """
        :return: The root of the folded circuit, whose node ``i`` is node ``i`` of the
            root.
        """
        for group in self.layer_groups:
            self.positions_in_folded_layers.update(self.fold(layers=group.layers))
        return self.positions_in_folded_layers[self.root].folded_layer

    def fold(self, *, layers: list[Layer]) -> dict[Layer, PositionsInFoldedLayer]:
        """
        :param layers: The layers of a :class:`LayersFoldedTogether`, whose children are folded.
        :return: Where the nodes of every layer went in the one layer they fold into.
        """
        first = layers[0]
        if isinstance(first, UnivariateInputLayer):
            return self.fold_input_layers(layers=layers)
        if isinstance(first, InnerLayer):
            return self.fold_inner_layers(layers=layers)
        return {
            first: PositionsInFoldedLayer(
                folded_layer=first, positions=np.arange(first.number_of_nodes)
            )
        }

    @staticmethod
    def fold_input_layers(
        *, layers: list[UnivariateInputLayer]
    ) -> dict[Layer, PositionsInFoldedLayer]:
        """
        :param layers: Univariate input layers of one type.
        :return: Where their nodes went in one layer, sorted by variable.
        """
        joined = type(layers[0]).concatenate(layers)
        order = np.argsort(joined.node_variables, kind="stable")
        folded = joined.select_nodes(order)
        position_in_folded = np.argsort(order)
        offsets = np.cumsum([0] + [layer.number_of_nodes for layer in layers])
        return {
            layer: PositionsInFoldedLayer(
                folded_layer=folded,
                positions=position_in_folded[offset : offset + layer.number_of_nodes],
            )
            for layer, offset in zip(layers, offsets)
        }

    def fold_inner_layers(
        self, *, layers: list[InnerLayer]
    ) -> dict[Layer, PositionsInFoldedLayer]:
        """
        :param layers: Sum layers or product layers, whose children are folded.
        :return: Where their nodes went in one layer, the nodes of ``layers[k]`` after
            those of the layers before it.
        """
        children = FoldedChildLayers(
            positions_in_folded_layers=self.positions_in_folded_layers
        )
        offsets = np.cumsum([0] + [layer.number_of_nodes for layer in layers])
        number_of_nodes = int(offsets[-1])
        edges = InnerLayerEdges.concatenate(
            [
                children.edges_of(layer=layer, first_node=int(offset))
                for layer, offset in zip(layers, offsets)
            ]
        )
        if isinstance(layers[0], ProductLayer):
            folded = ProductLayer.from_edges(
                children.layers, edges, number_of_nodes, SparseEdgeMatrix
            )
        else:
            folded = SumLayer.from_entries(
                children.layers,
                SparseEntries(
                    np.concatenate(
                        [np.asarray(layer.edge_log_weights) for layer in layers]
                    ),
                    edges.nodes,
                    children.column_offsets[edges.child_layer_indices]
                    + edges.child_nodes,
                ),
                number_of_nodes,
                BlockEdgeMatrix,
            )
        return {
            layer: PositionsInFoldedLayer(
                folded_layer=folded, positions=np.arange(layer.number_of_nodes) + offset
            )
            for layer, offset in zip(layers, offsets)
        }
