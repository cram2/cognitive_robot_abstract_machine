"""
Tests that the numeric queries of the univariate input layers run on every supported
array backend and agree with numpy there.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from array_api_compat import array_namespace
from random_events.interval import Bound, closed, open
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Continuous, Integer, Symbolic, Variable
from sortedcontainers import SortedSet
from typing_extensions import Any, Callable

from probabilistic_model.distributions.distributions import (
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    UnivariateInputLayer,
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
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)
from probabilistic_model.utils import MissingDict

from .test_layered_probabilistic_circuit import Letter

# %% layers of every type, built with the arrays of a backend


@dataclass(kw_only=True)
class InputLayerQueryCase:
    """
    An input layer of one type, which can be built with the parameters of any array
    backend, and the values and the event its numeric queries are asked for.
    """

    build: Callable[[ModuleType], UnivariateInputLayer]
    """
    Build the layer with the parameters created by an array namespace.
    """

    variable: Variable
    """
    The variable of the layer, the only variable of the circuit.
    """

    values: list[float]
    """
    The values of the variable to evaluate the layer at.
    """

    assignment: AbstractCompositeSet
    """
    The assignment of the variable of the simple event to compute the probability of.
    """

    @property
    def variables(self) -> SortedSet:
        """
        :return: The variables of the circuit.
        """
        return SortedSet([self.variable])

    def events(self, *, namespace: ModuleType) -> Any:
        """
        :param namespace: An array namespace.
        :return: The values as events of shape (#values, 1), an array of that namespace.
        """
        return namespace.asarray(np.array(self.values, dtype=float).reshape(-1, 1))

    @property
    def simple_event(self) -> SimpleEvent:
        """
        :return: The simple event to compute the probability of.
        """
        return SimpleEvent.from_data({self.variable: self.assignment})

    @property
    def moment_query(self) -> MomentQuery:
        """
        :return: The second moment of the variable, centered at a half.
        """
        return MomentQuery.from_maps(
            {self.variable: 2}, {self.variable: 0.5}, self.variables
        )


def convert_table_to_namespace(layer: DiscreteLayer, namespace: ModuleType) -> Any:
    """
    :param layer: A discrete layer with a dense table of numpy arrays.
    :param namespace: An array namespace.
    :return: The layer with its log-probabilities in an array of that namespace.
    """
    return dataclasses.replace(
        layer,
        table=DenseProbabilityTable(namespace.asarray(layer.table.log_probabilities)),
    )


@pytest.fixture
def gaussian_case() -> InputLayerQueryCase:
    """
    Gaussians at -1 and 2.
    """
    return InputLayerQueryCase(
        build=lambda namespace: GaussianLayer(
            0, namespace.asarray([-1.0, 2.0]), namespace.asarray([0.5, 1.5])
        ),
        variable=Continuous("x"),
        values=[-3.0, -1.0, 0.0, 0.5, 2.0, 7.0],
        assignment=closed(-1.0, 0.5) | closed(1.0, 3.0),
    )


@pytest.fixture
def truncated_gaussian_case() -> InputLayerQueryCase:
    """
    Three truncated Gaussians whose supports have different kinds of bounds.
    """
    return InputLayerQueryCase(
        build=lambda namespace: TruncatedGaussianLayer(
            0,
            namespace.asarray([[-1.0, 0.5], [0.0, 3.0], [2.0, 10.0]]),
            np.array(
                [
                    [int(Bound.CLOSED), int(Bound.OPEN)],
                    [int(Bound.OPEN), int(Bound.CLOSED)],
                    [int(Bound.CLOSED), int(Bound.CLOSED)],
                ],
                dtype=np.int64,
            ),
            namespace.asarray([0.0, 2.5, 0.0]),
            namespace.asarray([1.0, 0.5, 2.0]),
        ),
        variable=Continuous("x"),
        values=[-2.0, -1.0, 0.0, 0.5, 2.0, 3.0, 11.0],
        assignment=closed(-0.5, 2.5),
    )


@pytest.fixture
def uniform_case() -> InputLayerQueryCase:
    """
    Uniform distributions over [0, 2) and (1, 5].
    """
    return InputLayerQueryCase(
        build=lambda namespace: UniformLayer(
            0,
            namespace.asarray([[0.0, 2.0], [1.0, 5.0]]),
            np.array(
                [
                    [int(Bound.CLOSED), int(Bound.OPEN)],
                    [int(Bound.OPEN), int(Bound.CLOSED)],
                ],
                dtype=np.int64,
            ),
        ),
        variable=Continuous("x"),
        values=[-1.0, 0.0, 1.0, 1.5, 2.0, 5.0, 6.0],
        assignment=open(0.5, 1.5) | closed(4.0, 8.0),
    )


@pytest.fixture
def dirac_delta_case() -> InputLayerQueryCase:
    """
    Dirac deltas at 0.5 and 3.
    """
    return InputLayerQueryCase(
        build=lambda namespace: DiracDeltaLayer(
            0, namespace.asarray([0.5, 3.0]), namespace.asarray([1.0, 2.0])
        ),
        variable=Continuous("x"),
        values=[0.0, 0.5, 1.0, 3.0, 4.0],
        assignment=closed(0.0, 1.0),
    )


@pytest.fixture
def integer_case() -> InputLayerQueryCase:
    """
    Two distributions over the integers that share some of their states.
    """
    count = Integer("n")
    distributions = [
        IntegerDistribution(
            variable=count, probabilities=MissingDict(float, {1: 0.2, 2: 0.5, 4: 0.3})
        ),
        IntegerDistribution(
            variable=count, probabilities=MissingDict(float, {2: 0.6, 3: 0.4})
        ),
    ]
    layer = IntegerLayer.from_distributions(0, distributions)
    return InputLayerQueryCase(
        build=lambda namespace: convert_table_to_namespace(layer, namespace),
        variable=count,
        values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        assignment=closed(2, 3),
    )


@pytest.fixture
def symbolic_case() -> InputLayerQueryCase:
    """
    Two categorical distributions over :class:`Letter`.
    """
    letter = Symbolic(name="s", domain=Set.from_iterable(Letter))
    distributions = [
        SymbolicDistribution(
            variable=letter,
            probabilities=MissingDict(
                float, {hash(Letter.A): 0.3, hash(Letter.B): 0.7}
            ),
        ),
        SymbolicDistribution(
            variable=letter,
            probabilities=MissingDict(
                float, {hash(Letter.B): 0.4, hash(Letter.C): 0.6}
            ),
        ),
    ]
    layer = SymbolicLayer.from_distributions(0, distributions)
    return InputLayerQueryCase(
        build=lambda namespace: convert_table_to_namespace(layer, namespace),
        variable=letter,
        values=[0.0, 1.0, 2.0],
        assignment=Set.from_iterable([Letter.A, Letter.C]),
    )


@pytest.fixture(
    params=[
        "gaussian_case",
        "truncated_gaussian_case",
        "uniform_case",
        "dirac_delta_case",
        "integer_case",
        "symbolic_case",
    ]
)
def input_layer_case(request) -> InputLayerQueryCase:
    """
    An input layer of every univariate type.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "gaussian_case",
        "truncated_gaussian_case",
        "uniform_case",
        "dirac_delta_case",
        "integer_case",
    ]
)
def ordered_input_layer_case(request) -> InputLayerQueryCase:
    """
    An input layer of every univariate type over an ordered variable, which has a
    cumulative distribution function and a moment in closed form.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[np, jnp], ids=["numpy", "jax"])
def array_backend(request) -> ModuleType:
    """
    The array namespace of every supported backend, with jax computing in double
    precision.
    """
    with jax.enable_x64(True):
        yield request.param


def assert_result_of_backend_equals(
    result: Any, expected: np.ndarray, namespace: ModuleType
):
    """
    :param result: The result of a query of a layer with parameters of a backend.
    :param expected: The result of the same query of the layer with numpy parameters.
    :param namespace: The array namespace of the backend.
    """
    assert array_namespace(result) is array_namespace(namespace.asarray(0.0))
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12)


# %% numeric queries agree with numpy on every backend


def test_log_likelihood_of_nodes_on_every_backend(input_layer_case, array_backend):
    """
    The log-likelihood of a layer is an array of the backend of its parameters and
    agrees with that of the layer with numpy parameters.
    """
    case = input_layer_case
    assert_result_of_backend_equals(
        case.build(array_backend).log_likelihood_of_nodes(
            case.events(namespace=array_backend)
        ),
        case.build(np).log_likelihood_of_nodes(case.events(namespace=np)),
        array_backend,
    )


def test_cumulative_distribution_of_nodes_on_every_backend(
    ordered_input_layer_case, array_backend
):
    """
    The cumulative distribution of a layer is an array of the backend of its parameters
    and agrees with that of the layer with numpy parameters.
    """
    case = ordered_input_layer_case
    assert_result_of_backend_equals(
        case.build(array_backend).cumulative_distribution_of_nodes(
            case.events(namespace=array_backend)
        ),
        case.build(np).cumulative_distribution_of_nodes(case.events(namespace=np)),
        array_backend,
    )


def test_probability_of_simple_event_of_nodes_on_every_backend(
    input_layer_case, array_backend
):
    """
    The probability of a simple event under a layer is an array of the backend of its
    parameters and agrees with that of the layer with numpy parameters.
    """
    case = input_layer_case
    assert_result_of_backend_equals(
        case.build(array_backend).probability_of_simple_event_of_nodes(
            case.simple_event, case.variables
        ),
        case.build(np).probability_of_simple_event_of_nodes(
            case.simple_event, case.variables
        ),
        array_backend,
    )


def test_moment_of_nodes_on_every_backend(ordered_input_layer_case, array_backend):
    """
    The moment of a layer is an array of the backend of its parameters and agrees with
    that of the layer with numpy parameters.
    """
    case = ordered_input_layer_case
    assert_result_of_backend_equals(
        case.build(array_backend).moment_of_nodes(case.moment_query, case.variables),
        case.build(np).moment_of_nodes(case.moment_query, case.variables),
        array_backend,
    )


# %% numeric queries compile with jax


def test_log_likelihood_of_nodes_compiles_with_jax(input_layer_case):
    """
    The log-likelihood of a layer with jax parameters runs under :func:`jax.jit` and
    agrees with that of the layer with numpy parameters.
    """
    case = input_layer_case
    with jax.enable_x64(True):
        layer = case.build(jnp)
        result = jax.jit(lambda events: layer.log_likelihood_of_nodes(events))(
            case.events(namespace=jnp)
        )
    np.testing.assert_allclose(
        np.asarray(result),
        case.build(np).log_likelihood_of_nodes(case.events(namespace=np)),
        rtol=1e-12,
    )


def test_cumulative_distribution_of_nodes_compiles_with_jax(ordered_input_layer_case):
    """
    The cumulative distribution of a layer with jax parameters runs under
    :func:`jax.jit` and agrees with that of the layer with numpy parameters.
    """
    case = ordered_input_layer_case
    with jax.enable_x64(True):
        layer = case.build(jnp)
        result = jax.jit(lambda events: layer.cumulative_distribution_of_nodes(events))(
            case.events(namespace=jnp)
        )
    np.testing.assert_allclose(
        np.asarray(result),
        case.build(np).cumulative_distribution_of_nodes(case.events(namespace=np)),
        rtol=1e-12,
    )


# %% copies of layers that answered a query


def test_a_layer_that_answered_a_query_can_be_deep_copied(
    input_layer_case, array_backend
):
    """
    A deep copy of a layer whose backend a query has looked up answers that query on the
    same backend, like the original.
    """
    case = input_layer_case
    layer = case.build(array_backend)
    expected = layer.log_likelihood_of_nodes(case.events(namespace=array_backend))
    assert_result_of_backend_equals(
        copy.deepcopy(layer).log_likelihood_of_nodes(
            case.events(namespace=array_backend)
        ),
        np.asarray(expected),
        array_backend,
    )
