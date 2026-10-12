"""
The conversion of a circuit into any other representation of circuits, through every
representation in between.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Type, TypeVar

from probabilistic_model.adapters.converter import Converter
from probabilistic_model.adapters.exceptions import NoConversionPathError
from probabilistic_model.adapters.jax_tensorized.jax_to_tensorized import (
    DifferentiableLayeredCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.jax_tensorized.tensorized_to_jax import (
    LayeredCircuitToDifferentiableLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)

Representation = TypeVar("Representation")


@dataclass
class CircuitRepresentations:
    """
    The representations of circuits and the converters of whole circuits between them.

    A circuit converts into every representation that a chain of these converters
    reaches, for instance a rustworkx circuit into a differentiable circuit through the
    layered circuit.
    """

    converters: List[Type[Converter]] = field(
        default_factory=lambda: [
            RustworkxCircuitToLayeredCircuitConverter,
            LayeredCircuitToRustworkxCircuitConverter,
            DifferentiableLayeredCircuitToLayeredCircuitConverter,
            LayeredCircuitToDifferentiableLayeredCircuitConverter,
        ]
    )
    """
    The converters of whole circuits from one representation into another.
    """

    def converters_between(
        self, circuit_type: Type, representation: Type
    ) -> List[Type[Converter]]:
        """
        :param circuit_type: The representation to convert from. A subclass of a
            representation, such as a learned rustworkx circuit, converts like it.
        :param representation: The representation to convert into.
        :return: The shortest chain of converters from one representation into the
            other, empty if the circuit already has the representation.
        :raises NoConversionPathError: If no chain of converters connects them.
        """
        chains: Dict[Type, List[Type[Converter]]] = {circuit_type: []}
        unvisited = deque([circuit_type])
        while unvisited:
            current = unvisited.popleft()
            if issubclass(current, representation):
                return chains[current]
            for converter in self.converters:
                if issubclass(current, converter.input_type()) and (
                    converter.output_type() not in chains
                ):
                    chains[converter.output_type()] = chains[current] + [converter]
                    unvisited.append(converter.output_type())
        raise NoConversionPathError(
            circuit_type=circuit_type, representation=representation
        )

    def convert(
        self, circuit: Any, representation: Type[Representation]
    ) -> Representation:
        """
        :param circuit: A circuit in any representation.
        :param representation: The representation to convert the circuit into.
        :return: The circuit in that representation, the circuit itself if it already
            has it.
        """
        for converter in self.converters_between(type(circuit), representation):
            circuit = converter.convert(circuit)
        return circuit
