from __future__ import annotations

import numpy as np
from random_events.product_algebra import Event, SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Integer, Symbolic, Variable
from sortedcontainers import SortedSet

from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.exceptions import (
    EventMixesPointsAndIntervalsError,
    UnboundedEventError,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    Layer,
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
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    IntegerLayer,
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.utils import (
    MissingDict,
    event_compatible_for_truncation_with_singletons,
)

# %% uniform measures


def uniform_measure_of_event(event: Event) -> LayeredProbabilisticCircuit:
    """
    :param event: The event to spread the measure over.
    :return: The circuit that is uniform over the event.
    :raises UnboundedEventError: If the event does not bound a continuous variable
        from below and above.
    :raises EventMixesPointsAndIntervalsError: If a continuous variable takes points
        and intervals in the box bounding the event.
    """
    uniform_measure = uniform_measure_of_simple_event(event.bounding_box())
    uniform_measure.log_truncated_in_place(event)
    return uniform_measure


def uniform_measure_of_simple_event(
    simple_event: SimpleEvent,
) -> LayeredProbabilisticCircuit:
    """
    :param simple_event: The simple event to spread the measure over.
    :return: The circuit that is uniform over the simple event.
    :raises UnboundedEventError: If the simple event does not bound a continuous
        variable from below and above.
    :raises EventMixesPointsAndIntervalsError: If a continuous variable takes points
        and intervals.
    """
    if not event_compatible_for_truncation_with_singletons(
        simple_event.as_composite_set()
    ):
        raise EventMixesPointsAndIntervalsError(simple_event)
    variables = SortedSet(simple_event.keys())
    return LayeredProbabilisticCircuit(
        variables,
        ProductLayer.product_of(
            [
                uniform_layer_of_assignment(
                    variables.index(variable), variable, assignment
                )
                for variable, assignment in simple_event.items()
            ]
        ),
    )


def uniform_layer_of_assignment(
    variable_index: int, variable: Variable, assignment: AbstractCompositeSet
) -> Layer:
    """
    :param variable_index: The index of the variable in the circuit.
    :param variable: The variable.
    :param assignment: The values the variable may take.
    :return: A layer with one node that is uniform over the values.
    :raises UnboundedEventError: If the variable is continuous and the values are
        not bounded from below and above.
    """
    if isinstance(variable, Symbolic):
        return SymbolicLayer.from_distributions(
            variable_index,
            [
                SymbolicDistribution(
                    variable=variable,
                    probabilities=MissingDict(
                        float,
                        {
                            hash(value): 1 / len(assignment.simple_sets)
                            for value in assignment
                        },
                    ),
                )
            ],
        )
    if isinstance(variable, Integer):
        return IntegerLayer.from_distributions(
            variable_index,
            [
                IntegerDistribution(
                    variable=variable,
                    probabilities=MissingDict(
                        float,
                        {
                            value.lower: 1 / len(assignment.simple_sets)
                            for value in assignment
                        },
                    ),
                )
            ],
        )
    if not (
        np.isfinite(assignment.simple_sets[0].lower)
        and np.isfinite(assignment.simple_sets[-1].upper)
    ):
        raise UnboundedEventError(variable=variable)
    pieces = list(assignment)
    if all(piece.is_singleton() for piece in pieces):
        pieces_layer = DiracDeltaLayer.from_distributions(
            variable_index,
            [
                DiracDeltaDistribution(
                    variable=variable, location=piece.lower, density_cap=1.0
                )
                for piece in pieces
            ],
        )
        log_weights = np.zeros(len(pieces))
    else:
        pieces_layer = UniformLayer.from_distributions(
            variable_index,
            [
                UniformDistribution(variable=variable, interval=piece)
                for piece in pieces
            ],
        )
        log_weights = np.log([piece.upper - piece.lower for piece in pieces])
    number_of_pieces = len(pieces)
    mixture = SumLayer.from_edges(
        [pieces_layer],
        InnerLayerEdges(
            np.zeros(number_of_pieces, dtype=np.int64),
            np.zeros(number_of_pieces, dtype=np.int64),
            np.arange(number_of_pieces),
        ),
        log_weights,
        1,
    )
    mixture.normalize_own()
    return mixture
