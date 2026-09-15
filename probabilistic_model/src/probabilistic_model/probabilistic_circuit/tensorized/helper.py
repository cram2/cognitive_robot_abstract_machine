from __future__ import annotations

import numpy as np
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous, Variable
from sortedcontainers import SortedSet
from typing_extensions import Dict, Iterable, List, Optional

from probabilistic_model.probabilistic_circuit.tensorized.inner_layer import (
    Layer,
    ProductLayer,
    SparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import SparseArray
from probabilistic_model.probabilistic_circuit.rx import helper as rx_helper


def uniform_measure_of_simple_event(simple_event: SimpleEvent) -> LayeredProbabilisticCircuit:
    """
    Create the uniform measure over a simple event as a layered circuit.

    :param simple_event: The simple event.
    :return: The circuit describing the uniform measure.
    """
    return LayeredProbabilisticCircuit.from_rustworkx(
        rx_helper.uniform_measure_of_simple_event(simple_event)
    )


def uniform_measure_of_event(event: Event) -> LayeredProbabilisticCircuit:
    """
    Create the uniform measure over an event as a layered circuit.

    :param event: The event.
    :return: The circuit describing the uniform measure.
    """
    return LayeredProbabilisticCircuit.from_rustworkx(rx_helper.uniform_measure_of_event(event))


def fully_factorized(
    variables: Iterable[Variable],
    means: Optional[Dict[Continuous, float]] = None,
    variances: Optional[Dict[Continuous, float]] = None,
) -> LayeredProbabilisticCircuit:
    """
    Create a fully factorized distribution over a set of variables as a layered circuit.

    :param variables: The variables.
    :param means: The means of the normal distributions of the numeric variables.
    :param variances: The variances of the normal distributions of the numeric variables.
    :return: The circuit.
    """
    return LayeredProbabilisticCircuit.from_rustworkx(
        rx_helper.fully_factorized(variables, means, variances)
    )


def product_of(variables: SortedSet, child_layers: List[Layer]) -> ProductLayer:
    """
    Create a product layer with a single node that multiplies one node of every child
    layer.

    :param variables: The variables of the circuit the layer belongs to.
    :param child_layers: The child layers, each contributing its first node.
    :return: The product layer.
    """
    edges = SparseArray.from_coordinates(
        np.arange(len(child_layers)),
        np.zeros(len(child_layers), dtype=np.int64),
        np.zeros(len(child_layers), dtype=np.int64),
        (len(child_layers), 1),
    )
    return ProductLayer(child_layers, edges)


def mixture_of(
    child_layers: List[Layer], log_weights: Iterable[float]
) -> SparseSumLayer:
    """
    Create a sum layer with a single node that mixes the first node of every child layer.

    :param child_layers: The child layers, each contributing its first node.
    :param log_weights: The logarithmic weight of every child layer.
    :return: The sum layer.
    """
    weights = list(log_weights)
    if len(weights) != len(child_layers):
        raise ValueError(
            "The number of weights has to match the number of child layers."
        )
    return SparseSumLayer(
        child_layers,
        [
            SparseArray.from_coordinates(
                np.array([0]),
                np.array([0]),
                np.array([log_weight], dtype=float),
                (1, child_layer.number_of_nodes),
            )
            for child_layer, log_weight in zip(child_layers, weights)
        ],
    )
