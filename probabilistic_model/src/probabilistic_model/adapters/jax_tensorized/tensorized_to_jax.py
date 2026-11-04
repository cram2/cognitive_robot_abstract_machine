from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from random_events.variable import Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.adapters.converter import LayerConversion
from probabilistic_model.adapters.jax_tensorized.converter import (
    TensorizedToJaxConverter,
)
from probabilistic_model.adapters.jax_tensorized.exceptions import (
    StatesAreNotColumnIndicesError,
)
from probabilistic_model.adapters.jax_tensorized.utils import (
    columns_of_domain_elements,
    discrete_layer_of,
    sparse_matrix,
)
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    DifferentiableGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableLayer,
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
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
    IntegerLayer,
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

LayersToDifferentiable = LayerConversion[Layer, DifferentiableLayer]
"""
The conversion of the layers of one layered circuit into differentiable layers.
"""


# %% inner layers


class SumLayerToDifferentiableSparseSumLayerConverter(
    TensorizedToJaxConverter[SumLayer, DifferentiableSparseSumLayer]
):
    """
    Split the single weight matrix of a sum layer into one sparse weight matrix per
    child layer.
    """

    @classmethod
    def convert(
        cls, data: SumLayer, conversion: LayersToDifferentiable
    ) -> DifferentiableSparseSumLayer:
        offsets = data.column_offsets
        rows = data.log_weights.rows
        columns = data.log_weights.columns
        log_weights = data.normalized_edge_log_weights
        weights_per_child_layer = []
        for index, child_layer in enumerate(data.child_layers):
            into_child_layer = (columns >= offsets[index]) & (
                columns < offsets[index + 1]
            )
            weights_per_child_layer.append(
                sparse_matrix(
                    log_weights[into_child_layer],
                    rows[into_child_layer],
                    columns[into_child_layer] - offsets[index],
                    (data.number_of_nodes, child_layer.number_of_nodes),
                )
            )
        return DifferentiableSparseSumLayer(
            [conversion.convert(child_layer) for child_layer in data.child_layers],
            weights_per_child_layer,
        )


class ProductLayerToDifferentiableProductLayerConverter(
    TensorizedToJaxConverter[ProductLayer, DifferentiableProductLayer]
):
    """
    Both layers store their edges as the same sparse matrix of child node indices.
    """

    @classmethod
    def convert(
        cls, data: ProductLayer, conversion: LayersToDifferentiable
    ) -> DifferentiableProductLayer:
        edges = sparse_matrix(
            data.edges.data.astype(np.int32),
            data.edges.row,
            data.edges.col,
            data.edges.shape,
        )
        return DifferentiableProductLayer(
            [conversion.convert(child_layer) for child_layer in data.child_layers],
            edges,
        )


# %% input layers


class GaussianLayerToDifferentiableGaussianLayerConverter(
    TensorizedToJaxConverter[GaussianLayer, DifferentiableGaussianLayer]
):
    """
    The converted layer has no minimum scale, so that it has the same scale.
    """

    @classmethod
    def convert(
        cls, data: GaussianLayer, conversion: LayersToDifferentiable
    ) -> DifferentiableGaussianLayer:
        return DifferentiableGaussianLayer(
            data.variable,
            location=jnp.asarray(data.location),
            log_scale=jnp.log(jnp.asarray(data.scale)),
            minimum_scale=jnp.zeros(data.number_of_nodes),
        )


class UniformLayerToDifferentiableUniformLayerConverter(
    TensorizedToJaxConverter[UniformLayer, DifferentiableUniformLayer]
):
    """
    A differentiable uniform layer treats every interval as open, so a closed bound is
    lost.
    """

    @classmethod
    def convert(
        cls, data: UniformLayer, conversion: LayersToDifferentiable
    ) -> DifferentiableUniformLayer:
        return DifferentiableUniformLayer(data.variable, jnp.asarray(data.interval))


class DiracDeltaLayerToDifferentiableDiracDeltaLayerConverter(
    TensorizedToJaxConverter[DiracDeltaLayer, DifferentiableDiracDeltaLayer]
):
    """
    A differentiable Dirac delta layer compares a value with its location exactly, so
    the tolerance is lost.
    """

    @classmethod
    def convert(
        cls, data: DiracDeltaLayer, conversion: LayersToDifferentiable
    ) -> DifferentiableDiracDeltaLayer:
        return DifferentiableDiracDeltaLayer(
            data.variable, jnp.asarray(data.location), jnp.asarray(data.density_cap)
        )


class SymbolicLayerToDifferentiableDiscreteLayerConverter(
    TensorizedToJaxConverter[SymbolicLayer, DifferentiableDiscreteLayer]
):
    """
    Lay the states of a symbolic layer out as the columns of a probability table with
    one column per domain element, the column of an element being its hash.
    """

    @classmethod
    def convert(
        cls, data: SymbolicLayer, conversion: LayersToDifferentiable
    ) -> DifferentiableDiscreteLayer:
        variable: Symbolic = conversion.variables[data.variable]
        columns = columns_of_domain_elements(variable)
        return discrete_layer_of(data, columns[data.states], len(columns))


class IntegerLayerToDifferentiableDiscreteLayerConverter(
    TensorizedToJaxConverter[IntegerLayer, DifferentiableDiscreteLayer]
):
    """
    Lay the states of an integer layer out as the columns of a probability table that
    reaches up to the largest state.
    """

    @classmethod
    def convert(
        cls, data: IntegerLayer, conversion: LayersToDifferentiable
    ) -> DifferentiableDiscreteLayer:
        if (data.states < 0).any():
            raise StatesAreNotColumnIndicesError(
                variable=conversion.variables[data.variable], states=data.states
            )
        return discrete_layer_of(data, data.states, int(data.states.max()) + 1)


# %% circuit


class LayeredCircuitToDifferentiableLayeredCircuitConverter(
    TensorizedToJaxConverter[LayeredProbabilisticCircuit, DifferentiableLayeredCircuit]
):
    """
    Convert a layered circuit into a differentiable circuit, so that a circuit learned
    in another way can be refined by gradient descent.
    """

    @classmethod
    def convert(cls, data: LayeredProbabilisticCircuit) -> DifferentiableLayeredCircuit:
        conversion = LayersToDifferentiable(TensorizedToJaxConverter, data.variables)
        return DifferentiableLayeredCircuit(
            SortedSet(data.variables), conversion.convert(data.root)
        )
