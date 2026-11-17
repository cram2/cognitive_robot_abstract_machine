"""
How the ORM of the experiments stores classes whose fields it cannot map one by one.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from krrood.adapters.json_serializer import JSONData, from_json, to_json
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping

from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% circuits


@dataclass(eq=False, kw_only=True)
class LayeredProbabilisticCircuitMapping(
    AlternativeMapping[LayeredProbabilisticCircuit]
):
    """
    A layered circuit, stored as its JSON form.
    """

    circuit_json: JSONData
    """
    The JSON form of the circuit.
    """

    @classmethod
    def from_domain_object(
        cls, *, obj: LayeredProbabilisticCircuit
    ) -> LayeredProbabilisticCircuitMapping:
        return cls(circuit_json=to_json(obj))

    def to_domain_object(self) -> LayeredProbabilisticCircuit:
        return from_json(self.circuit_json)
