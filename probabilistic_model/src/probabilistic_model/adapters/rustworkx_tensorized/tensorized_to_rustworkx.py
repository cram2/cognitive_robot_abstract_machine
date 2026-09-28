from __future__ import annotations

from dataclasses import dataclass, field

from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, List

from probabilistic_model.adapters.rustworkx_tensorized.converter import (
    TensorizedToRustworkxConverter,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    Unit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    InputLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)


@dataclass
class RustworkxCircuitBuilder:
    """
    The state of converting the layers of one layered circuit into one rustworkx
    circuit, which converts every layer once.
    """

    variables: SortedSet
    """
    The variables of the circuit, in the order the layers index them.
    """

    circuit: ProbabilisticCircuit = field(default_factory=ProbabilisticCircuit)
    """
    The circuit the units are created in.
    """

    units_by_layer: Dict[int, List[Unit]] = field(default_factory=dict)
    """
    The units created for every layer converted so far, keyed by the id of the layer.
    """

    def units_of(self, layer: Layer) -> List[Unit]:
        """
        :param layer: A layer of the circuit.
        :return: One unit per node of the layer, in the order of its nodes.
        """
        if id(layer) not in self.units_by_layer:
            self.units_by_layer[id(layer)] = TensorizedToRustworkxConverter.convert(
                layer, self
            )
        return self.units_by_layer[id(layer)]


class SumLayerToSumUnitsConverter(
    TensorizedToRustworkxConverter[SumLayer, List[SumUnit]]
):

    @classmethod
    def convert(cls, data: SumLayer, builder: RustworkxCircuitBuilder) -> List[SumUnit]:
        units = [
            SumUnit(probabilistic_circuit=builder.circuit)
            for _ in range(data.number_of_nodes)
        ]
        child_units = [
            builder.units_of(child_layer) for child_layer in data.child_layers
        ]
        for edge, log_weight in zip(data.iterate_edges(), data.log_weights.data):
            units[edge.node].add_subcircuit(
                child_units[edge.child_layer_index][edge.child_node], float(log_weight)
            )
        for unit in units:
            unit.normalize()
        return units


class ProductLayerToProductUnitsConverter(
    TensorizedToRustworkxConverter[ProductLayer, List[ProductUnit]]
):

    @classmethod
    def convert(
        cls, data: ProductLayer, builder: RustworkxCircuitBuilder
    ) -> List[ProductUnit]:
        units = [
            ProductUnit(probabilistic_circuit=builder.circuit)
            for _ in range(data.number_of_nodes)
        ]
        child_units = [
            builder.units_of(child_layer) for child_layer in data.child_layers
        ]
        for edge in data.iterate_edges():
            units[edge.node].add_subcircuit(
                child_units[edge.child_layer_index][edge.child_node]
            )
        return units


class InputLayerToLeavesConverter(
    TensorizedToRustworkxConverter[InputLayer, List[Unit]]
):
    """
    Convert any input layer into one leaf per node, through the distributions of its
    nodes.
    """

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        return isinstance(data, InputLayer)

    @classmethod
    def convert(cls, data: InputLayer, builder: RustworkxCircuitBuilder) -> List[Unit]:
        return [
            leaf(distribution, builder.circuit)
            for distribution in data.node_distributions(
                builder.variables[data.variable]
            )
        ]


class LayeredCircuitToRustworkxCircuitConverter(
    TensorizedToRustworkxConverter[LayeredProbabilisticCircuit, ProbabilisticCircuit]
):

    @classmethod
    def convert(cls, data: LayeredProbabilisticCircuit) -> ProbabilisticCircuit:
        builder = RustworkxCircuitBuilder(data.variables)
        builder.units_of(data.root)
        return builder.circuit
