"""
The log-likelihood of a layered circuit with jax parameters, compiled by :func:`jax.jit`
into one program for every shape of events.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import jax
from typing_extensions import Callable

from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    SampleArray,
)

# %% compiled log-likelihood


@dataclass(kw_only=True)
class CompiledLogLikelihood:
    """
    :meth:`LayeredProbabilisticCircuit.log_likelihood` of a circuit with jax parameters,
    compiled once for every shape of events and reused for every later call.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The circuit, with jax parameters.
    """

    compiled: Callable[..., jax.Array] = field(init=False, repr=False)
    """
    :meth:`log_likelihood_under_parameters`, compiled.
    """

    def __post_init__(self) -> None:
        """
        Compile :meth:`log_likelihood_under_parameters`.
        """
        self.compiled = jax.jit(self.log_likelihood_under_parameters)

    def log_likelihood_under_parameters(
        self, *, parameters: list[jax.Array], encoded_events: jax.Array
    ) -> jax.Array:
        """
        :param parameters: Arrays in the order and with the shapes of the circuit's
            :attr:`~probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base.Layer.circuit_parameters`.
        :param encoded_events: Events encoded as the circuit encodes them.
        :return: The log-likelihood of every event under the circuit with these
            parameters.
        """
        root = self.circuit.root.copy_with_circuit_parameters(parameters)
        return root.log_likelihood_of_nodes(encoded_events)[:, 0]

    def log_likelihood(self, *, encoded_events: SampleArray) -> jax.Array:
        """
        :param encoded_events: Events encoded as the circuit encodes them.
        :return: The log-likelihood of every event under the circuit.
        """
        return self.compiled(
            parameters=self.circuit.root.circuit_parameters,
            encoded_events=encoded_events,
        )
