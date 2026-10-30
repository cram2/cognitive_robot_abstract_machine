from __future__ import annotations

from abc import abstractmethod

import jax
import numpy as np
from jax.experimental.sparse import BCOO
from random_events.interval import Bound
from random_events.variable import Integer, Symbolic
from scipy.sparse import coo_array
from sortedcontainers import SortedSet

from probabilistic_model.adapters.converter import LayerConversion
from probabilistic_model.adapters.exceptions import CannotConvertError
from probabilistic_model.adapters.jax_tensorized.converter import (
    InputType,
    JaxToTensorizedConverter,
)
from probabilistic_model.adapters.jax_tensorized.utils import (
    columns_of_domain_elements,
    to_numpy,
)
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    DifferentiableGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableDenseSumLayer,
    DifferentiableLayer,
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
    DifferentiableSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableDiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.jax.uniform_layer import (
    DifferentiableUniformLayer,
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
    DiscreteLayer,
    IntegerLayer,
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
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
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)

LayersToTensorized = LayerConversion[DifferentiableLayer, Layer]
"""
The conversion of the layers of one differentiable circuit into layers of the
``tensorized`` package.
"""


# %% inner layers


class DifferentiableSumLayerToSumLayerConverter(
    JaxToTensorizedConverter[InputType, SumLayer]
):
    """
    Base class for converters of the differentiable sum layers, which keep one weight
    matrix per child layer, into one sum layer with a single sparse weight matrix whose
    columns are the nodes of all child layers in order.

    Training leaves the weights unnormalized, so the converted weights are normalized.
    """

    @staticmethod
    @abstractmethod
    def entries_of(log_weights: BCOO | jax.Array) -> SparseEntries:
        """
        :param log_weights: The logarithmic weights of the edges into one child layer.
        :return: The stored entries of the weights, the columns indexing the nodes of
            that child layer.
        """
        raise NotImplementedError

    @classmethod
    def convert(
        cls, data: DifferentiableSumLayer, conversion: LayersToTensorized
    ) -> SumLayer:
        child_layers = [
            conversion.convert(child_layer) for child_layer in data.child_layers
        ]
        entries = []
        offset = 0
        for log_weights, child_layer in zip(data.log_weights, child_layers):
            child_entries = cls.entries_of(log_weights)
            child_entries.columns = child_entries.columns + offset
            entries.append(child_entries)
            offset += child_layer.number_of_nodes

        layer = SumLayer(
            child_layers,
            RowGroupedSparseArray.from_entries(
                SparseEntries.concatenate(entries), (data.number_of_nodes, offset)
            ),
        )
        layer.normalize_own()
        return layer


class DifferentiableSparseSumLayerToSumLayerConverter(
    DifferentiableSumLayerToSumLayerConverter[DifferentiableSparseSumLayer]
):

    @staticmethod
    def entries_of(log_weights: BCOO) -> SparseEntries:
        indices = np.asarray(log_weights.indices, dtype=np.int64)
        return SparseEntries(to_numpy(log_weights.data), indices[:, 0], indices[:, 1])


class DifferentiableDenseSumLayerToSumLayerConverter(
    DifferentiableSumLayerToSumLayerConverter[DifferentiableDenseSumLayer]
):

    @staticmethod
    def entries_of(log_weights: jax.Array) -> SparseEntries:
        rows, columns = np.indices(log_weights.shape, dtype=np.int64)
        return SparseEntries(
            to_numpy(log_weights).ravel(), rows.ravel(), columns.ravel()
        )


class DifferentiableProductLayerToProductLayerConverter(
    JaxToTensorizedConverter[DifferentiableProductLayer, ProductLayer]
):
    """
    Both layers store their edges as the same sparse matrix of child node indices.
    """

    @classmethod
    def convert(
        cls, data: DifferentiableProductLayer, conversion: LayersToTensorized
    ) -> ProductLayer:
        indices = np.asarray(data.edges.indices, dtype=np.int64)
        edges = coo_array(
            (
                np.asarray(data.edges.data, dtype=np.int64),
                (indices[:, 0], indices[:, 1]),
            ),
            shape=data.edges.shape,
        )
        return ProductLayer(
            [conversion.convert(child_layer) for child_layer in data.child_layers],
            edges,
        )


# %% input layers


class DifferentiableGaussianLayerToGaussianLayerConverter(
    JaxToTensorizedConverter[DifferentiableGaussianLayer, GaussianLayer]
):
    """
    The scale of a differentiable Gaussian layer includes its minimum scale.
    """

    @classmethod
    def convert(
        cls, data: DifferentiableGaussianLayer, conversion: LayersToTensorized
    ) -> GaussianLayer:
        return GaussianLayer(
            data.variable, to_numpy(data.location), to_numpy(data.scale)
        )


class DifferentiableUniformLayerToUniformLayerConverter(
    JaxToTensorizedConverter[DifferentiableUniformLayer, UniformLayer]
):
    """
    A differentiable uniform layer treats every interval as open.
    """

    @classmethod
    def convert(
        cls, data: DifferentiableUniformLayer, conversion: LayersToTensorized
    ) -> UniformLayer:
        interval = to_numpy(data.interval)
        return UniformLayer(
            data.variable,
            interval,
            np.full(interval.shape, int(Bound.OPEN), dtype=np.int64),
        )


class DifferentiableDiracDeltaLayerToDiracDeltaLayerConverter(
    JaxToTensorizedConverter[DifferentiableDiracDeltaLayer, DiracDeltaLayer]
):

    @classmethod
    def convert(
        cls, data: DifferentiableDiracDeltaLayer, conversion: LayersToTensorized
    ) -> DiracDeltaLayer:
        return DiracDeltaLayer(
            data.variable, to_numpy(data.location), to_numpy(data.density_cap)
        )


class DifferentiableDiscreteLayerToDiscreteLayerConverter(
    JaxToTensorizedConverter[DifferentiableDiscreteLayer, DiscreteLayer]
):
    """
    A differentiable discrete layer looks the probability of a value up in the column
    with the value as its index: the value itself for an integer variable and the hash
    of the domain element for a symbolic variable.

    The variable decides whether the layer becomes a symbolic or an integer layer.
    """

    @classmethod
    def convert(
        cls, data: DifferentiableDiscreteLayer, conversion: LayersToTensorized
    ) -> DiscreteLayer:
        variable = conversion.variables[data.variable]
        log_probabilities = to_numpy(data.normalized_log_probabilities)
        if isinstance(variable, Symbolic):
            columns = columns_of_domain_elements(variable)
            return SymbolicLayer(
                data.variable,
                np.arange(len(columns), dtype=np.int64),
                DenseProbabilityTable(log_probabilities[:, columns]),
                SymbolicEncoding(variable).hashes,
            )
        if isinstance(variable, Integer):
            return IntegerLayer(
                data.variable,
                np.arange(log_probabilities.shape[1], dtype=np.int64),
                DenseProbabilityTable(log_probabilities),
            )
        raise CannotConvertError(data_type=type(variable))


# %% circuit


class DifferentiableLayeredCircuitToLayeredCircuitConverter(
    JaxToTensorizedConverter[DifferentiableLayeredCircuit, LayeredProbabilisticCircuit]
):
    """
    Convert a differentiable circuit, usually after training it, into a layered circuit
    that answers every query.

    A classification circuit has one root node per class and stays differentiable, so
    no subclass of a differentiable circuit is converted.
    """

    @classmethod
    def convert(cls, data: DifferentiableLayeredCircuit) -> LayeredProbabilisticCircuit:
        if not cls.can_convert(data):
            raise CannotConvertError(data_type=type(data))
        variables = SortedSet(data.variables)
        conversion = LayersToTensorized(JaxToTensorizedConverter, variables)
        return LayeredProbabilisticCircuit(variables, conversion.convert(data.root))
