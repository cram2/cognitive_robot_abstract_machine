import numpy as np
import pytest
from random_events.interval import Bound, closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous, Integer
from sortedcontainers import SortedSet

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

x = Continuous("x")
y = Continuous("y")
n = Integer("n")

# %% stacking conditioned copies


@pytest.fixture
def layered_circuit() -> LayeredProbabilisticCircuit:
    """
    A mixture of two products of a uniform distribution over ``x`` and a distribution
    over ``n``.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, (lower, upper), probabilities in (
        (0.3, (0.0, 1.0), {0: 0.9, 1: 0.1}),
        (0.7, (2.0, 3.0), {0: 0.2, 1: 0.8}),
    ):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=x, interval=closed(lower, upper).simple_sets[0]
                ),
                circuit,
            )
        )
        product.add_subcircuit(
            leaf(
                IntegerDistribution(
                    variable=n, probabilities=MissingDict(float, probabilities)
                ),
                circuit,
            )
        )
    return RustworkxCircuitToLayeredCircuitConverter.convert(circuit)


def conditioned_on(layered_circuit: LayeredProbabilisticCircuit, value: int) -> Layer:
    """
    :param layered_circuit: The circuit of :func:`layered_circuit`.
    :param value: A value of ``n``.
    :return: The root layer of the circuit conditioned on the value, without ``n``.
    """
    conditioned = layered_circuit.root.log_conditional_of_point(
        {n: value}, StructuralQuery(layered_circuit.variables), cache=QueryCache()
    )
    kept = np.array([variable == x for variable in layered_circuit.variables])
    return conditioned.layer.marginal(kept)


class TestStackingConditionedCopies:
    """
    Copies of one circuit conditioned on different points share their structure, so they
    stack into one layer graph whose root holds one node per copy.
    """

    def test_node_of_every_copy_is_that_copy(self, layered_circuit):
        copies = [conditioned_on(layered_circuit, value) for value in (0, 1)]
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
        self, layered_circuit
    ):
        copies = [conditioned_on(layered_circuit, value) for value in (0, 1)]
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

    def test_copies_of_different_structure_are_rejected(self, layered_circuit):
        with pytest.raises(CopiesNotAlignedError):
            AlignedCopiesStacker().stack(
                [layered_circuit.root, layered_circuit.root.child_layers[0]]
            )


# %% equal parameters of leaf layers


def uniform_layer_over(upper: float) -> UniformLayer:
    """
    :param upper: The upper bound of the support.
    :return: A layer of one uniform distribution over ``x`` from zero to the bound.
    """
    return UniformLayer.from_distributions(
        0,
        [UniformDistribution(variable=x, interval=closed(0.0, upper).simple_sets[0])],
    )


def truncated_gaussian_layer_over(upper: float) -> TruncatedGaussianLayer:
    """
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


def integer_layer_of(
    probability_of_zero: float,
    table_type: type[ProbabilityTable] = DenseProbabilityTable,
) -> IntegerLayer:
    """
    :param probability_of_zero: The probability of ``n`` being zero, the rest is on one.
    :param table_type: The type of the probability table.
    :return: A layer of one distribution over ``n``.
    """
    return IntegerLayer.from_distributions(
        0,
        [
            IntegerDistribution(
                variable=n,
                probabilities=MissingDict(
                    float, {0: probability_of_zero, 1: 1.0 - probability_of_zero}
                ),
            )
        ],
        table_type,
    )


def gaussian_over_x_and_y(variance_of_x: float) -> MultivariateGaussianDistribution:
    """
    :param variance_of_x: The variance of ``x``.
    :return: A correlated Gaussian over ``x`` and ``y``.
    """
    return MultivariateGaussianDistribution(
        variables=(x, y),
        mean=np.zeros(2),
        covariance=Covariance.from_matrix([[variance_of_x, 0.5], [0.5, 1.0]]),
    )


def truncated_multivariate_gaussian_layer_over(
    upper: float,
) -> TruncatedMultivariateGaussianLayer:
    """
    :param upper: The upper bound of the box on ``x``.
    :return: A layer of one correlated Gaussian over ``x`` and ``y``, truncated to a box.
    """
    return TruncatedMultivariateGaussianLayer.from_distributions(
        SortedSet([x, y]),
        [
            TruncatedMultivariateGaussianDistribution(
                untruncated=gaussian_over_x_and_y(1.0),
                box=SimpleEvent.from_data(
                    {x: closed(-1.0, upper), y: closed(-1.0, 1.0)}
                ),
            )
        ],
    )


@pytest.mark.parametrize(
    "layer,changed",
    [
        pytest.param(uniform_layer_over(1.0), uniform_layer_over(2.0), id="uniform"),
        pytest.param(
            GaussianLayer(0, np.zeros(1), np.ones(1)),
            GaussianLayer(0, np.zeros(1), np.full(1, 2.0)),
            id="gaussian",
        ),
        pytest.param(
            truncated_gaussian_layer_over(1.0),
            truncated_gaussian_layer_over(2.0),
            id="truncated gaussian",
        ),
        pytest.param(
            DiracDeltaLayer.from_distributions(
                0, [DiracDeltaDistribution(variable=x, location=0.0, density_cap=1.0)]
            ),
            DiracDeltaLayer.from_distributions(
                0, [DiracDeltaDistribution(variable=x, location=0.0, density_cap=2.0)]
            ),
            id="dirac delta",
        ),
        pytest.param(integer_layer_of(0.3), integer_layer_of(0.4), id="integer"),
        pytest.param(
            integer_layer_of(0.3, SparseProbabilityTable),
            integer_layer_of(0.4, SparseProbabilityTable),
            id="integer with a sparse table",
        ),
        pytest.param(
            MultivariateGaussianLayer.from_distributions(
                SortedSet([x, y]), [gaussian_over_x_and_y(1.0)]
            ),
            MultivariateGaussianLayer.from_distributions(
                SortedSet([x, y]), [gaussian_over_x_and_y(2.0)]
            ),
            id="multivariate gaussian",
        ),
        pytest.param(
            truncated_multivariate_gaussian_layer_over(1.0),
            truncated_multivariate_gaussian_layer_over(2.0),
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
        self, layer: LeafLayer, changed: LeafLayer
    ):
        assert layer.has_equal_parameters(layer.__deepcopy__())

    def test_a_changed_parameter_is_unequal(self, layer: LeafLayer, changed: LeafLayer):
        assert not layer.has_equal_parameters(changed)


class TestStackingCopiesOfLeafLayers:
    """
    Copies of a leaf layer are shared exactly when their parameters are equal.
    """

    def test_tables_of_different_types_are_unequal(self):
        assert not integer_layer_of(0.3).has_equal_parameters(
            integer_layer_of(0.3, SparseProbabilityTable)
        )

    def test_equal_copies_of_a_leaf_layer_are_shared(self):
        layer = uniform_layer_over(1.0)
        stacked = AlignedCopiesStacker().stack([layer, layer.__deepcopy__()])
        assert stacked.is_shared
        assert stacked.layer is layer


# %% attaching a child layer


def test_only_the_given_nodes_multiply_the_attached_child_layer():
    uniform_layer = UniformLayer.from_distributions(
        0,
        [
            UniformDistribution(variable=x, interval=closed(0.0, 1.0).simple_sets[0]),
            UniformDistribution(variable=x, interval=closed(0.0, 2.0).simple_sets[0]),
        ],
    )
    product_layer = ProductLayer.node_wise_product_of([uniform_layer])
    point_mass = DiracDeltaLayer.from_distributions(
        1, [DiracDeltaDistribution(variable=y, location=3.0, density_cap=2.0)]
    )
    product_layer.attach_child_layer(point_mass, np.array([1]), np.array([0]))

    events = np.array([[0.5, 3.0]])
    np.testing.assert_allclose(
        product_layer.log_likelihood_of_nodes(events),
        [[np.log(1.0), np.log(0.5) + np.log(2.0)]],
    )
    np.testing.assert_array_equal(product_layer.variables, [0, 1])


# %% point masses


def test_every_point_mass_puts_its_mass_on_its_value():
    values = [0, 2]
    layer = LayeredProbabilisticCircuit.point_masses_layer(
        0,
        [
            IntegerDistribution(
                variable=n, probabilities=MissingDict(float, {value: 1.0})
            )
            for value in values
        ],
    )
    np.testing.assert_array_equal(
        layer.log_likelihood_of_nodes(np.array([[value] for value in values])),
        [[0.0, -np.inf], [-np.inf, 0.0]],
    )
