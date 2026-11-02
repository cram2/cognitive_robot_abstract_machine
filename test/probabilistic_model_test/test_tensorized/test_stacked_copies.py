from dataclasses import dataclass

import numpy as np
import pytest
from random_events.interval import Bound, closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous, Integer
from sortedcontainers import SortedSet
from typing_extensions import Callable

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
)
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.distributions.truncated_multivariate_gaussian import (
    TruncatedMultivariateGaussianDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    CopiesNotAlignedError,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    Layer,
    LeafLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    IntegerLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.multivariate_gaussian_layer import (
    MultivariateGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.truncated_multivariate_gaussian_layer import (
    TruncatedMultivariateGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
    ProbabilityTable,
    SparseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import QueryCache
from probabilistic_model.probabilistic_circuit.tensorized.stacked_copies import (
    AlignedCopiesStacker,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    StructuralQuery,
)
from probabilistic_model.utils import MissingDict


@dataclass
class Variables:
    """
    The variables of the circuits and layers of these tests.
    """

    x: Continuous
    y: Continuous
    count: Integer


@pytest.fixture
def variables() -> Variables:
    return Variables(Continuous("x"), Continuous("y"), Integer("count"))


# %% stacking conditioned copies


def mixture_of(
    variables: Variables,
    components: list[tuple[float, tuple[float, float], dict[int, float]]],
) -> LayeredProbabilisticCircuit:
    """
    :param variables: The variables of the tests.
    :param components: The weight, the bounds of the uniform distribution over ``x``
        and the probabilities of ``count`` of every component.
    :return: A mixture of products of a uniform distribution over ``x`` and a
        distribution over ``count``.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, (lower, upper), probabilities in components:
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=variables.x, interval=closed(lower, upper).simple_sets[0]
                ),
                circuit,
            )
        )
        product.add_subcircuit(
            leaf(
                IntegerDistribution(
                    variable=variables.count,
                    probabilities=MissingDict(float, probabilities),
                ),
                circuit,
            )
        )
    return RustworkxCircuitToLayeredCircuitConverter.convert(circuit)


@pytest.fixture
def layered_circuit(variables: Variables) -> LayeredProbabilisticCircuit:
    """
    A mixture of two products of a uniform distribution over ``x`` and a distribution
    over ``count``.
    """
    return mixture_of(
        variables,
        [
            (0.3, (0.0, 1.0), {0: 0.9, 1: 0.1}),
            (0.7, (2.0, 3.0), {0: 0.2, 1: 0.8}),
        ],
    )


def conditioned_on(
    layered_circuit: LayeredProbabilisticCircuit, variables: Variables, value: int
) -> Layer:
    """
    :param layered_circuit: The circuit of :func:`layered_circuit`.
    :param variables: The variables of the tests.
    :param value: A value of ``count``.
    :return: The root layer of the circuit conditioned on the value, without ``count``.
    """
    conditioned = layered_circuit.root.log_conditional_of_point(
        {variables.count: value},
        StructuralQuery(layered_circuit.variables),
        cache=QueryCache(),
    )
    kept = np.array([variable == variables.x for variable in layered_circuit.variables])
    return conditioned.layer.marginal(kept)


class TestStackingConditionedCopies:
    """
    Copies of one circuit conditioned on different points share their structure, so they
    stack into one layer graph whose root holds one node per copy.
    """

    def test_node_of_every_copy_is_that_copy(self, layered_circuit, variables):
        copies = [conditioned_on(layered_circuit, variables, value) for value in (0, 1)]
        stacked = AlignedCopiesStacker().stack(copies)
        stacked.layer.normalize()
        events = np.array([[0.0, 0.5], [0.0, 2.5], [0.0, 1.5]])
        log_likelihoods = stacked.layer.log_likelihood_of_nodes(events)
        for index, copy in enumerate(copies):
            copy.normalize()
            np.testing.assert_allclose(
                log_likelihoods[:, index], copy.log_likelihood_of_nodes(events)[:, 0]
            )

    def test_layers_that_conditioning_leaves_unchanged_are_shared(
        self, layered_circuit, variables
    ):
        copies = [conditioned_on(layered_circuit, variables, value) for value in (0, 1)]
        stacked = AlignedCopiesStacker().stack(copies)
        assert not stacked.is_shared
        [uniform_layer] = [
            layer
            for layer in stacked.layer.all_layers()
            if isinstance(layer, UniformLayer)
        ]
        [original_uniform_layer] = [
            layer for layer in layered_circuit.layers if isinstance(layer, UniformLayer)
        ]
        assert uniform_layer.number_of_nodes == original_uniform_layer.number_of_nodes

    def test_equal_copies_are_one_shared_copy(self, layered_circuit):
        stacked = AlignedCopiesStacker().stack(
            [layered_circuit.root, layered_circuit.root.__deepcopy__({})]
        )
        assert stacked.is_shared
        assert stacked.layer.number_of_nodes == layered_circuit.root.number_of_nodes


class TestCopiesNotAligned:
    """
    Copies that differ in their structure are rejected with the layers that differ and
    the path to them.
    """

    def test_differing_roots_are_reported_at_the_empty_path(self, layered_circuit):
        root = layered_circuit.root
        with pytest.raises(CopiesNotAlignedError) as error:
            AlignedCopiesStacker().stack([root, root, root.child_layers[0]])
        assert error.value.path == ()
        assert error.value.misaligned_copies == (2,)
        assert error.value.copies == (root, root, root.child_layers[0])

    def test_differing_child_layers_are_reported_at_their_path(
        self, layered_circuit, variables
    ):
        with_three_components = mixture_of(
            variables,
            [
                (0.2, (0.0, 1.0), {0: 0.9, 1: 0.1}),
                (0.3, (2.0, 3.0), {0: 0.2, 1: 0.8}),
                (0.5, (4.0, 5.0), {0: 0.5, 1: 0.5}),
            ],
        )
        with pytest.raises(CopiesNotAlignedError) as error:
            AlignedCopiesStacker().stack(
                [layered_circuit.root, with_three_components.root]
            )
        assert error.value.path == (0,)
        assert error.value.misaligned_copies == (1,)
        assert [copy.number_of_nodes for copy in error.value.copies] == [2, 3]
        assert "through the child layers [0]" in str(error.value)


# %% equal parameters of leaf layers


LeafLayerOf = Callable[[Variables, float], LeafLayer]
"""
Builds a leaf layer of one node over the variables of the tests from one parameter.
"""


def uniform_layer_over(variables: Variables, upper: float) -> UniformLayer:
    """
    :param variables: The variables of the tests.
    :param upper: The upper bound of the support.
    :return: A layer of one uniform distribution over ``x`` from zero to the bound.
    """
    return UniformLayer.from_distributions(
        0,
        [
            UniformDistribution(
                variable=variables.x, interval=closed(0.0, upper).simple_sets[0]
            )
        ],
    )


def gaussian_layer_with_scale(variables: Variables, scale: float) -> GaussianLayer:
    """
    :param variables: The variables of the tests.
    :param scale: The scale of the Gaussian.
    :return: A layer of one Gaussian over ``x`` with mean zero.
    """
    return GaussianLayer(0, np.zeros(1), np.full(1, scale))


def truncated_gaussian_layer_over(
    variables: Variables, upper: float
) -> TruncatedGaussianLayer:
    """
    :param variables: The variables of the tests.
    :param upper: The upper bound of the support.
    :return: A layer of one standard Gaussian over ``x``, truncated from zero to the
        bound.
    """
    return TruncatedGaussianLayer(
        0,
        np.array([[0.0, upper]]),
        np.full((1, 2), int(Bound.CLOSED), dtype=np.int64),
        np.zeros(1),
        np.ones(1),
    )


def dirac_delta_layer_with_density_cap(
    variables: Variables, density_cap: float
) -> DiracDeltaLayer:
    """
    :param variables: The variables of the tests.
    :param density_cap: The density at the location.
    :return: A layer of one Dirac delta over ``x`` at zero.
    """
    return DiracDeltaLayer.from_distributions(
        0,
        [
            DiracDeltaDistribution(
                variable=variables.x, location=0.0, density_cap=density_cap
            )
        ],
    )


def integer_layer_of(
    variables: Variables,
    probability_of_zero: float,
    table_type: type[ProbabilityTable] = DenseProbabilityTable,
) -> IntegerLayer:
    """
    :param variables: The variables of the tests.
    :param probability_of_zero: The probability of ``count`` being zero, the rest is on
        one.
    :param table_type: The type of the probability table.
    :return: A layer of one distribution over ``count``.
    """
    return IntegerLayer.from_distributions(
        0,
        [
            IntegerDistribution(
                variable=variables.count,
                probabilities=MissingDict(
                    float, {0: probability_of_zero, 1: 1.0 - probability_of_zero}
                ),
            )
        ],
        table_type,
    )


def integer_layer_with_sparse_table_of(
    variables: Variables, probability_of_zero: float
) -> IntegerLayer:
    """
    Like :func:`integer_layer_of`, with the probabilities in a sparse table.
    """
    return integer_layer_of(variables, probability_of_zero, SparseProbabilityTable)


def integer_point_mass_at(variables: Variables, value: float) -> IntegerLayer:
    """
    :param variables: The variables of the tests.
    :param value: The value of ``count`` that gets all of the mass.
    :return: A layer of one point mass over ``count``, whose only state is the value.
    """
    return LayeredProbabilisticCircuit.point_masses_layer(
        0,
        [
            IntegerDistribution(
                variable=variables.count,
                probabilities=MissingDict(float, {int(value): 1.0}),
            )
        ],
    )


def gaussian_over_x_and_y(
    variables: Variables, variance_of_x: float
) -> MultivariateGaussianDistribution:
    """
    :param variables: The variables of the tests.
    :param variance_of_x: The variance of ``x``.
    :return: A correlated Gaussian over ``x`` and ``y``.
    """
    return MultivariateGaussianDistribution(
        variables=(variables.x, variables.y),
        mean=np.zeros(2),
        covariance=Covariance.from_matrix([[variance_of_x, 0.5], [0.5, 1.0]]),
    )


def multivariate_gaussian_layer_with_variance_of_x(
    variables: Variables, variance_of_x: float
) -> MultivariateGaussianLayer:
    """
    :param variables: The variables of the tests.
    :param variance_of_x: The variance of ``x``.
    :return: A layer of one correlated Gaussian over ``x`` and ``y``.
    """
    return MultivariateGaussianLayer.from_distributions(
        SortedSet([variables.x, variables.y]),
        [gaussian_over_x_and_y(variables, variance_of_x)],
    )


def truncated_multivariate_gaussian_layer_over(
    variables: Variables, upper: float
) -> TruncatedMultivariateGaussianLayer:
    """
    :param variables: The variables of the tests.
    :param upper: The upper bound of the box on ``x``.
    :return: A layer of one correlated Gaussian over ``x`` and ``y``, truncated to a box.
    """
    return TruncatedMultivariateGaussianLayer.from_distributions(
        SortedSet([variables.x, variables.y]),
        [
            TruncatedMultivariateGaussianDistribution(
                untruncated=gaussian_over_x_and_y(variables, 1.0),
                box=SimpleEvent.from_data(
                    {variables.x: closed(-1.0, upper), variables.y: closed(-1.0, 1.0)}
                ),
            )
        ],
    )


@pytest.mark.parametrize(
    "layer_of,parameter,changed_parameter",
    [
        pytest.param(uniform_layer_over, 1.0, 2.0, id="uniform"),
        pytest.param(gaussian_layer_with_scale, 1.0, 2.0, id="gaussian"),
        pytest.param(truncated_gaussian_layer_over, 1.0, 2.0, id="truncated gaussian"),
        pytest.param(dirac_delta_layer_with_density_cap, 1.0, 2.0, id="dirac delta"),
        pytest.param(integer_layer_of, 0.3, 0.4, id="integer"),
        pytest.param(
            integer_layer_with_sparse_table_of,
            0.3,
            0.4,
            id="integer with a sparse table",
        ),
        pytest.param(
            integer_point_mass_at, 0, 2, id="integer point mass on another state"
        ),
        pytest.param(
            multivariate_gaussian_layer_with_variance_of_x,
            1.0,
            2.0,
            id="multivariate gaussian",
        ),
        pytest.param(
            truncated_multivariate_gaussian_layer_over,
            1.0,
            2.0,
            id="truncated multivariate gaussian",
        ),
    ],
)
class TestEqualParametersOfEveryLeafLayerType:
    """
    Two copies of a leaf layer are equal exactly when every node has the same parameters
    in both.
    """

    def test_a_layer_has_the_parameters_of_its_copy(
        self,
        variables: Variables,
        layer_of: LeafLayerOf,
        parameter: float,
        changed_parameter: float,
    ):
        layer = layer_of(variables, parameter)
        assert layer.has_equal_parameters(layer.__deepcopy__())

    def test_a_changed_parameter_is_unequal(
        self,
        variables: Variables,
        layer_of: LeafLayerOf,
        parameter: float,
        changed_parameter: float,
    ):
        assert not layer_of(variables, parameter).has_equal_parameters(
            layer_of(variables, changed_parameter)
        )


class TestStackingCopiesOfLeafLayers:
    """
    Copies of a leaf layer are shared exactly when their parameters are equal.
    """

    def test_tables_of_different_types_are_unequal(self, variables):
        assert not integer_layer_of(variables, 0.3).has_equal_parameters(
            integer_layer_with_sparse_table_of(variables, 0.3)
        )

    def test_equal_copies_of_a_leaf_layer_are_shared(self, variables):
        layer = uniform_layer_over(variables, 1.0)
        stacked = AlignedCopiesStacker().stack([layer, layer.__deepcopy__()])
        assert stacked.is_shared
        assert stacked.layer is layer


# %% attaching a child layer


def test_only_the_given_nodes_multiply_the_attached_child_layer(variables):
    uniform_layer = UniformLayer.from_distributions(
        0,
        [
            UniformDistribution(
                variable=variables.x, interval=closed(0.0, 1.0).simple_sets[0]
            ),
            UniformDistribution(
                variable=variables.x, interval=closed(0.0, 2.0).simple_sets[0]
            ),
        ],
    )
    product_layer = ProductLayer.node_wise_product_of([uniform_layer])
    point_mass = DiracDeltaLayer.from_distributions(
        1, [DiracDeltaDistribution(variable=variables.y, location=3.0, density_cap=2.0)]
    )
    product_layer.attach_child_layer(point_mass, np.array([1]), np.array([0]))

    events = np.array([[0.5, 3.0]])
    np.testing.assert_allclose(
        product_layer.log_likelihood_of_nodes(events),
        [[np.log(1.0), np.log(0.5) + np.log(2.0)]],
    )
    np.testing.assert_array_equal(product_layer.variables, [0, 1])


# %% point masses


def test_every_point_mass_puts_its_mass_on_its_value(variables):
    values = [0, 2]
    layer = LayeredProbabilisticCircuit.point_masses_layer(
        0,
        [
            IntegerDistribution(
                variable=variables.count, probabilities=MissingDict(float, {value: 1.0})
            )
            for value in values
        ],
    )
    np.testing.assert_array_equal(
        layer.log_likelihood_of_nodes(np.array([[value] for value in values])),
        [[0.0, -np.inf], [-np.inf, 0.0]],
    )
