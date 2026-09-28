from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Dict

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeMask,
    NodeValues,
)

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
        Layer,
    )


@dataclass
class LayerWithLogProbabilities:
    """
    A layer that a structural query created, with the log-probability of the query
    under each of its nodes.
    """

    layer: Layer
    """
    The created layer.
    """

    log_probabilities: NodeValues
    """
    The log-probability of every node of the layer.
    """


@dataclass
class LogProbabilitiesOfLayers:
    """
    The log-probabilities that a structural query recorded for the layers it created.

    A structural query keeps the number of nodes of every layer it rewrites, so that the
    edges of the parents stay valid, and marks the nodes that became impossible with a
    log-probability of ``-inf``. The prune pass that follows reads them back from here.
    """

    by_layer_id: Dict[int, NodeValues] = field(default_factory=dict)
    """
    The log-probabilities of the nodes of every recorded layer, keyed by the
    :func:`id` of the layer.
    """

    def record(self, result: LayerWithLogProbabilities) -> LayerWithLogProbabilities:
        """
        :param result: A layer created by the query and the log-probabilities of its
            nodes.
        :return: The same result.
        """
        self.by_layer_id[id(result.layer)] = result.log_probabilities
        return result

    def alive_nodes_of(self, layer: Layer) -> NodeMask:
        """
        :param layer: A layer of the circuit the query created.
        :return: Which nodes of the layer are still possible. A layer that nothing was
            recorded for was not rewritten, so all of its nodes are.
        """
        log_probabilities = self.by_layer_id.get(id(layer))
        if log_probabilities is None:
            return np.ones(layer.number_of_nodes, dtype=bool)
        return log_probabilities > -np.inf


@dataclass
class StructuralQuery:
    """
    The arguments that every layer of a structural query, a truncation or a
    conditioning, shares.
    """

    variables: SortedSet
    """
    The variables of the circuit.
    """

    singleton_allowed: bool = False
    """
    Whether a truncation to a singleton creates a Dirac delta rather than an impossible
    node.
    """

    log_probabilities: LogProbabilitiesOfLayers = field(
        default_factory=LogProbabilitiesOfLayers
    )
    """
    The log-probabilities of the nodes of every layer the query created.
    """
