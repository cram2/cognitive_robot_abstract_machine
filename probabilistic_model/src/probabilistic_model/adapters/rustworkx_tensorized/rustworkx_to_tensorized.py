from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, List, Tuple, Type

from probabilistic_model.adapters.rustworkx_tensorized.converter import (
    InputType,
    OutputType,
    RustworkxToTensorizedConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.exceptions import (
    NotExactlyOneRootError,
)
from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    Unit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
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
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.row_grouped_sparse_array import (
    RowGroupedSparseArray,
    SparseEntries,
)


@dataclass
class ConvertedLayer:
    """
    A layer created from units of a rustworkx circuit, together with those units.
    """

    layer: Layer
    """
    The created layer.
    """

    units: List[Unit]
    """
    The units the layer was created from, in the order of its nodes.
    """

    node_of_unit: Dict[int, int]
    """
    A map from the hash of a unit to the index of its node in the layer.
    """

    @classmethod
    def of_units(cls, layer: Layer, units: List[Unit]) -> ConvertedLayer:
        """
        :param layer: The layer created from the units.
        :param units: The units, in the order of the nodes of the layer.
        :return: The converted layer.
        """
        return cls(
            layer, units, {hash(unit): index for index, unit in enumerate(units)}
        )


class UnitsToLayerConverter(RustworkxToTensorizedConverter[InputType, OutputType]):
    """
    Base class for converters of a group of units of one type and scope into one layer.

    The input type is the unit class, or the distribution class for leaves.
    """

    @staticmethod
    def type_of_unit(unit: Unit) -> Type:
        """
        :param unit: A unit of a rustworkx circuit.
        :return: The distribution class of a leaf, or the class of any other unit.
        """
        return type(unit.distribution) if unit.is_leaf else type(unit)

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        return (
            isinstance(data, list)
            and len(data) > 0
            and cls.type_of_unit(data[0]) is cls.input_type()
        )


class SumUnitsToSumLayerConverter(UnitsToLayerConverter[SumUnit, SumLayer]):

    @classmethod
    def convert(
        cls, data: List[SumUnit], converted_layers: List[ConvertedLayer]
    ) -> ConvertedLayer:
        variables = np.array(
            [
                data[0].probabilistic_circuit.variables.index(variable)
                for variable in data[0].variables
            ]
        )

        child_layers = []
        rows, columns, values = [], [], []
        for converted_layer in converted_layers:
            if not np.array_equal(converted_layer.layer.variables, variables):
                continue
            offset = sum(layer.number_of_nodes for layer in child_layers)
            found = False
            for index, unit in enumerate(data):
                for log_weight, subcircuit in unit.log_weighted_subcircuits:
                    if hash(subcircuit) in converted_layer.node_of_unit:
                        rows.append(index)
                        columns.append(
                            offset + converted_layer.node_of_unit[hash(subcircuit)]
                        )
                        values.append(log_weight)
                        found = True
            if found:
                child_layers.append(converted_layer.layer)

        log_weights = RowGroupedSparseArray.from_entries(
            SparseEntries(np.array(values, dtype=float), rows, columns),
            (len(data), sum(layer.number_of_nodes for layer in child_layers)),
        )
        return ConvertedLayer.of_units(SumLayer(child_layers, log_weights), data)


class ProductUnitsToProductLayerConverter(
    UnitsToLayerConverter[ProductUnit, ProductLayer]
):

    @classmethod
    def convert(
        cls, data: List[ProductUnit], converted_layers: List[ConvertedLayer]
    ) -> ConvertedLayer:
        used_layers: List[ConvertedLayer] = []
        row_of_layer: Dict[int, int] = {}
        rows, columns, values = [], [], []

        for unit_index, unit in enumerate(data):
            subcircuit_hashes = {hash(subcircuit) for subcircuit in unit.subcircuits}
            for layer_index, converted_layer in enumerate(converted_layers):
                for subcircuit_hash in subcircuit_hashes:
                    if subcircuit_hash not in converted_layer.node_of_unit:
                        continue
                    if layer_index not in row_of_layer:
                        row_of_layer[layer_index] = len(used_layers)
                        used_layers.append(converted_layer)
                    rows.append(row_of_layer[layer_index])
                    columns.append(unit_index)
                    values.append(converted_layer.node_of_unit[subcircuit_hash])

        edges = SparseEntries(
            np.array(values, dtype=np.int64), rows, columns
        ).to_coo_array((len(used_layers), len(data)))
        layer = ProductLayer(
            [converted_layer.layer for converted_layer in used_layers], edges
        )
        return ConvertedLayer.of_units(layer, data)


class LeavesToInputLayerConverter(UnitsToLayerConverter[InputType, OutputType]):
    """
    Base class for converters of leaves whose distributions are of one class into the
    input layer that holds that class.
    """

    @classmethod
    def convert(
        cls, data: List[Unit], converted_layers: List[ConvertedLayer]
    ) -> ConvertedLayer:
        layer = cls.output_type().from_distributions(
            data[0].probabilistic_circuit.variables.index(data[0].variable),
            [unit.distribution for unit in data],
        )
        return ConvertedLayer.of_units(layer, data)


class DiracDeltaLeavesToDiracDeltaLayerConverter(
    LeavesToInputLayerConverter[DiracDeltaDistribution, DiracDeltaLayer]
): ...


class UniformLeavesToUniformLayerConverter(
    LeavesToInputLayerConverter[UniformDistribution, UniformLayer]
): ...


class GaussianLeavesToGaussianLayerConverter(
    LeavesToInputLayerConverter[GaussianDistribution, GaussianLayer]
): ...


class TruncatedGaussianLeavesToTruncatedGaussianLayerConverter(
    LeavesToInputLayerConverter[TruncatedGaussianDistribution, TruncatedGaussianLayer]
): ...


class SymbolicLeavesToSymbolicLayerConverter(
    LeavesToInputLayerConverter[SymbolicDistribution, SymbolicLayer]
): ...


class IntegerLeavesToIntegerLayerConverter(
    LeavesToInputLayerConverter[IntegerDistribution, IntegerLayer]
): ...


class RustworkxCircuitToLayeredCircuitConverter(
    RustworkxToTensorizedConverter[ProbabilisticCircuit, LayeredProbabilisticCircuit]
):
    """
    Convert a rustworkx circuit into a layered circuit, one level of the rustworkx
    circuit at a time from the leaves up.
    """

    @classmethod
    def convert(cls, data: ProbabilisticCircuit) -> LayeredProbabilisticCircuit:
        converted_layers: List[ConvertedLayer] = []
        for units in reversed(list(data.layers)):
            converted_layers = [
                RustworkxToTensorizedConverter.convert(group, converted_layers)
                for group in cls.groups_of_level(units)
            ] + converted_layers

        roots = [
            converted_layer
            for converted_layer in converted_layers
            if converted_layer.units[0] is data.root
        ]
        if len(roots) != 1:
            raise NotExactlyOneRootError(number_of_roots=len(roots))
        return LayeredProbabilisticCircuit(SortedSet(data.variables), roots[0].layer)

    @staticmethod
    def groups_of_level(units: List[Unit]) -> List[List[Unit]]:
        """
        :param units: The units of one level of a rustworkx circuit.
        :return: The units grouped by exact type and scope, one group per layer.
        """
        groups: Dict[Tuple[Type, Tuple], List[Unit]] = {}
        for unit in units:
            key = (UnitsToLayerConverter.type_of_unit(unit), tuple(unit.variables))
            groups.setdefault(key, []).append(unit)
        return list(groups.values())
