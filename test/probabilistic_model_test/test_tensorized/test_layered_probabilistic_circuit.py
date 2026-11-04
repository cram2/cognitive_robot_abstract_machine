"""
Cross validation of the layered numpy circuits against the rustworkx implementation.

Every query of a layered circuit has to return what the circuit of the ``rx`` package it
was converted from returns, so most tests here build a circuit with the ``rx`` classes,
convert it and compare the two answers.

Two queries are checked against a closed form instead of against ``rx``, because ``rx``
answers them incorrectly for circuits in which a subcircuit has more than one parent:

- ``marginal`` (and therefore the marginalization step inside ``conditional``) routes
  through ``simplify``, which merges a sum unit into its parent sum unit by adding an
  edge per grandchild. The graph is not a multigraph, so when both sum units point at the
  same grandchild the second ``add_subcircuit`` overwrites the weight of the first
  instead of adding to it, and that branch loses its mass.
- ``log_conditional_in_place`` reports the log-probability of whichever node is the root
  *after* that simplification rather than of the node the forward pass computed it for,
  so it returns the leftover value of an unrelated node when simplification replaces the
  root.

The values asserted below are derived by hand in the test that uses them and confirmed by
numerically integrating the joint density.
"""

from __future__ import annotations

import math
from enum import Enum, IntEnum

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.adapters.exceptions import CannotConvertError
from probabilistic_model.adapters.rustworkx_tensorized.exceptions import (
    NotExactlyOneRootError,
    UnitNotConvertedError,
)
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.exceptions import IntractableError, ShapeMismatchError
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RxCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    UndefinedCumulativeDistributionError,
)
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
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    SparseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)
from probabilistic_model.query_results import Truncation
from probabilistic_model.utils import MissingDict

from ..circuit_building import uniform_layer_of, uniform_leaf


class Letter(IntEnum):
    """
    The categories of a symbolic variable.
    """

    A = 0
    B = 1
    C = 2


class SparseLetter(IntEnum):
    """
    Domain elements whose hashes are not their positions in the domain.
    """

    LOW = 3
    MIDDLE = 7
    HIGH = 11


class SymbolWithLargeHash(Enum):
    """
    Domain elements whose hashes are larger than the hashes Python reduces integers to,
    so that hashing such a hash again does not give it back.
    """

    FIRST = 2**62 + 2**20
    SECOND = 2**62 + 2**21

    def __hash__(self) -> int:
        return self.value


class SymbolWithHashBeyondFloatPrecision(Enum):
    """
    Domain elements whose hashes need more significant bits than the mantissa of a float
    holds, so that storing such a hash as a float does not give it back.

    Their hashes still differ as floats, so the positions of the elements stay
    unambiguous.
    """

    FIRST = 2**62 + 1
    SECOND = 2**62 + 1025

    def __hash__(self) -> int:
        return self.value


class UnconvertibleUniformDistribution(UniformDistribution):
    """
    A distribution type that no converter handles: converters dispatch on the exact type
    of a distribution.
    """


def grid_of(lower: float, upper: float, number_of_points: int) -> np.ndarray:
    """
    :return: The points of a square grid over two variables.
    """
    axis = np.linspace(lower, upper, number_of_points)
    return np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)


def symbolic_distribution(
    variable: Symbolic, probabilities: dict[Enum, float]
) -> SymbolicDistribution:
    """
    :param variable: The variable of the distribution.
    :param probabilities: The probability of every domain element.
    :return: The distribution.
    """
    return SymbolicDistribution(
        variable=variable,
        probabilities=MissingDict(
            float,
            {
                hash(element): probability
                for element, probability in probabilities.items()
            },
        ),
    )


# %% variables


@pytest.fixture
def letter() -> Symbolic:
    """
    A symbolic variable over :class:`Letter`.
    """
    return Symbolic(name="s", domain=Set.from_iterable(Letter))


@pytest.fixture
def sparse_letter() -> Symbolic:
    """
    A symbolic variable over :class:`SparseLetter`.
    """
    return Symbolic(name="sparse", domain=Set.from_iterable(SparseLetter))


# %% circuits


@pytest.fixture
def overlapping_mixture(x, y) -> RxCircuit:
    """
    A mixture of two products of uniforms whose supports overlap, so the circuit is not
    deterministic.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, (x_lower, x_upper), (y_lower, y_upper) in [
        (0.4, (0, 1), (0, 2)),
        (0.6, (1, 3), (1, 2)),
    ]:
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(uniform_leaf(x, x_lower, x_upper, circuit))
        product.add_subcircuit(uniform_leaf(y, y_lower, y_upper, circuit))
    return circuit


@pytest.fixture
def deterministic_mixture(x, y) -> RxCircuit:
    """
    A mixture of two products of uniforms with disjoint supports.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, (x_lower, x_upper), (y_lower, y_upper) in [
        (0.3, (0, 1), (0, 1)),
        (0.7, (2, 3), (2, 4)),
    ]:
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(uniform_leaf(x, x_lower, x_upper, circuit))
        product.add_subcircuit(uniform_leaf(y, y_lower, y_upper, circuit))
    return circuit


@pytest.fixture
def mixed_circuit(x, count, letter) -> RxCircuit:
    """
    A circuit over a continuous, an integer and a symbolic variable.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, (lower, upper), integer_probabilities, letter_probabilities in (
        (0.4, (0, 2), {0: 0.2, 1: 0.8}, (0.5, 0.25, 0.25)),
        (0.6, (1, 4), {0: 0.7, 2: 0.3}, (0.1, 0.6, 0.3)),
    ):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(uniform_leaf(x, lower, upper, circuit))
        product.add_subcircuit(
            leaf(
                IntegerDistribution(
                    variable=count,
                    probabilities=MissingDict(float, integer_probabilities),
                ),
                circuit,
            )
        )
        product.add_subcircuit(
            leaf(
                symbolic_distribution(letter, dict(zip(Letter, letter_probabilities))),
                circuit,
            )
        )
    return circuit


@pytest.fixture
def gaussian_circuit(x, y) -> RxCircuit:
    """
    A mixture of products of Gaussians.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, location, scale in ((0.35, -1.0, 0.5), (0.65, 2.0, 1.5)):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(
            leaf(
                GaussianDistribution(variable=x, location=location, scale=scale),
                circuit,
            )
        )
        product.add_subcircuit(
            leaf(
                GaussianDistribution(variable=y, location=-location, scale=scale),
                circuit,
            )
        )
    return circuit


@pytest.fixture
def sparse_symbolic_circuit(x, sparse_letter) -> RxCircuit:
    """
    A mixture over a continuous variable and a symbolic variable whose domain elements
    hash to numbers that are not their positions in the domain.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, (lower, upper), letter_probabilities in (
        (0.3, (0, 2), {SparseLetter.LOW: 0.6, SparseLetter.HIGH: 0.4}),
        (0.7, (1, 3), {SparseLetter.MIDDLE: 0.5, SparseLetter.HIGH: 0.5}),
    ):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(uniform_leaf(x, lower, upper, circuit))
        product.add_subcircuit(
            leaf(symbolic_distribution(sparse_letter, letter_probabilities), circuit)
        )
    return circuit


@pytest.fixture(
    params=[
        "overlapping_mixture",
        "deterministic_mixture",
        "shared_children_circuit",
        "mixed_circuit",
        "gaussian_circuit",
    ]
)
def any_circuit(request) -> RxCircuit:
    """
    Every circuit of this module.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "overlapping_mixture",
        "deterministic_mixture",
        "shared_children_circuit",
        "gaussian_circuit",
    ]
)
def continuous_circuit(request) -> RxCircuit:
    """
    Every circuit of this module over continuous variables only.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "overlapping_mixture",
        "deterministic_mixture",
        "shared_children_circuit",
    ]
)
def uniform_circuit(request) -> RxCircuit:
    """
    Every circuit of this module over uniform distributions only.
    """
    return request.getfixturevalue(request.param)


# %% conversion


def test_every_circuit_converts_and_keeps_its_variables(any_circuit):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit)
    assert list(layered.variables) == list(any_circuit.variables)
    layered.validate()


def test_the_product_units_of_a_level_become_the_nodes_of_one_layer(
    shared_children_circuit,
):
    converted = RustworkxCircuitToLayeredCircuitConverter.convert_with_layers(
        shared_children_circuit
    )
    nodes = [
        converted.node_of(unit)
        for unit in shared_children_circuit.nodes()
        if isinstance(unit, ProductUnit)
    ]
    assert nodes[0].layer is nodes[1].layer
    assert isinstance(nodes[0].layer, ProductLayer)
    assert {node.node for node in nodes} == {0, 1}


def test_a_unit_of_another_circuit_was_not_converted(
    shared_children_circuit, overlapping_mixture
):
    converted = RustworkxCircuitToLayeredCircuitConverter.convert_with_layers(
        shared_children_circuit
    )
    with pytest.raises(UnitNotConvertedError):
        converted.node_of(overlapping_mixture.root)


def test_a_leaf_without_a_converter_is_reported(x):
    circuit = RxCircuit()
    leaf(
        UnconvertibleUniformDistribution(
            variable=x, interval=closed(0, 1).simple_sets[0]
        ),
        circuit,
    )
    with pytest.raises(CannotConvertError):
        RustworkxCircuitToLayeredCircuitConverter.convert(circuit)


def test_a_conversion_without_a_root_layer_is_reported(
    overlapping_mixture, monkeypatch
):
    monkeypatch.setattr(
        RustworkxCircuitToLayeredCircuitConverter,
        "groups_of_level",
        staticmethod(lambda units: []),
    )
    with pytest.raises(NotExactlyOneRootError):
        RustworkxCircuitToLayeredCircuitConverter.convert(overlapping_mixture)


def test_layer_types_of_a_uniform_mixture(overlapping_mixture):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(overlapping_mixture)
    assert isinstance(layered.root, SumLayer)
    assert layered.root.number_of_nodes == 1

    [product_layer] = layered.root.child_layers
    assert isinstance(product_layer, ProductLayer)
    assert product_layer.number_of_nodes == 2
    assert len(product_layer.child_layers) == 2
    for child_layer in product_layer.child_layers:
        assert isinstance(child_layer, UniformLayer)
        assert child_layer.number_of_nodes == 2


def test_shared_leaves_become_one_layer_with_two_nodes(shared_children_circuit):
    uniform_layers = [
        layer
        for layer in RustworkxCircuitToLayeredCircuitConverter.convert(
            shared_children_circuit
        ).layers
        if isinstance(layer, UniformLayer)
    ]
    # one layer per variable, each holding the two shared leaves
    assert len(uniform_layers) == 2
    assert {layer.number_of_nodes for layer in uniform_layers} == {2}


def test_gaussian_leaves_become_a_gaussian_layer(gaussian_circuit):
    assert any(
        isinstance(layer, GaussianLayer)
        for layer in RustworkxCircuitToLayeredCircuitConverter.convert(
            gaussian_circuit
        ).layers
    )


def test_discrete_leaves_become_discrete_layers(mixed_circuit):
    layers = RustworkxCircuitToLayeredCircuitConverter.convert(mixed_circuit).layers
    assert any(isinstance(layer, SymbolicLayer) for layer in layers)
    assert any(isinstance(layer, IntegerLayer) for layer in layers)


def test_round_trip_through_rustworkx_keeps_the_likelihood(continuous_circuit):
    samples = continuous_circuit.sample(200)
    np.testing.assert_allclose(
        LayeredCircuitToRustworkxCircuitConverter.convert(
            RustworkxCircuitToLayeredCircuitConverter.convert(continuous_circuit)
        ).log_likelihood(samples),
        continuous_circuit.log_likelihood(samples),
    )


# %% queries that agree with rustworkx


def test_log_likelihood(any_circuit):
    samples = any_circuit.sample(500)
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit).log_likelihood(
            samples
        ),
        any_circuit.log_likelihood(samples),
    )


def test_log_likelihood_outside_the_support_is_minus_infinity(deterministic_mixture):
    outside = np.array([[10.0, 10.0], [1.5, 1.5]])
    assert np.all(
        np.isneginf(
            RustworkxCircuitToLayeredCircuitConverter.convert(
                deterministic_mixture
            ).log_likelihood(outside)
        )
    )


def test_cumulative_distribution_function(continuous_circuit):
    samples = continuous_circuit.sample(200)
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(
            continuous_circuit
        ).cumulative_distribution_function(samples),
        continuous_circuit.cumulative_distribution_function(samples),
    )


def test_moment(continuous_circuit, x, y):
    order = VariableMap({x: 1, y: 1})
    center = VariableMap({x: 0.0, y: 0.0})
    expected = continuous_circuit.moment(order, center)
    moment = RustworkxCircuitToLayeredCircuitConverter.convert(
        continuous_circuit
    ).moment(order, center)
    assert set(moment.keys()) == set(expected.keys())
    for variable, value in expected.items():
        np.testing.assert_allclose(moment[variable], value)


def test_moment_of_a_subset_of_the_variables(overlapping_mixture, x, y):
    order = VariableMap({x: 1})
    center = VariableMap({x: 0.0})
    moment = RustworkxCircuitToLayeredCircuitConverter.convert(
        overlapping_mixture
    ).moment(order, center)
    expected = overlapping_mixture.moment(order, center)
    np.testing.assert_allclose(moment[x], expected[x])
    # the unrequested variable still reports, as it does in the rx circuit
    np.testing.assert_allclose(moment[y], expected[y])


def test_central_moment(overlapping_mixture, x):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(overlapping_mixture)
    mean = layered.moment(VariableMap({x: 1}), VariableMap({x: 0.0}))
    order = VariableMap({x: 2})
    center = VariableMap({x: mean[x]})

    variance = layered.moment(order, center)[x]
    np.testing.assert_allclose(variance, overlapping_mixture.moment(order, center)[x])
    # the variance of the mixture, against the samples it generates
    samples = overlapping_mixture.sample(200000)[:, 0]
    assert variance == pytest.approx(float(samples.var()), abs=5e-3)


def test_probability_of_a_simple_event(overlapping_mixture, x, y):
    event = SimpleEvent.from_data({x: closed(0.5, 2.0), y: closed(0.0, 1.5)})
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        overlapping_mixture
    ).probability_of_simple_event(event) == pytest.approx(
        overlapping_mixture.probability_of_simple_event(event)
    )


def test_probability_of_a_composite_event(overlapping_mixture, x):
    event = SimpleEvent.from_data(
        {x: closed(0.0, 0.5) | closed(2.0, 2.5)}
    ).as_composite_set()
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        overlapping_mixture
    ).probability(event.__deepcopy__()) == pytest.approx(
        overlapping_mixture.probability(event.__deepcopy__())
    )


def test_probability_of_a_mixed_event(mixed_circuit, x, count, letter):
    event = SimpleEvent.from_data(
        {
            x: closed(0.5, 2.5),
            count: closed(0, 0),
            letter: Set.from_iterable([Letter.A, Letter.C]),
        }
    )
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        mixed_circuit
    ).probability_of_simple_event(event) == pytest.approx(
        mixed_circuit.probability_of_simple_event(event)
    )


def test_support(any_circuit):
    assert (
        RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit).support
        == any_circuit.support
    )


def test_expectation_and_variance(continuous_circuit):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(continuous_circuit)
    for variable in continuous_circuit.variables:
        assert layered.expectation()[variable] == pytest.approx(
            continuous_circuit.expectation()[variable]
        )
        assert layered.variance()[variable] == pytest.approx(
            continuous_circuit.variance()[variable]
        )


@pytest.mark.parametrize(
    "circuit_fixture, variable_fixture",
    [("mixed_circuit", "count"), ("overlapping_mixture", "x")],
    ids=["integer variable", "subset of the variables"],
)
def test_expectation_of_one_variable(circuit_fixture, variable_fixture, request):
    circuit = request.getfixturevalue(circuit_fixture)
    variable = request.getfixturevalue(variable_fixture)
    assert RustworkxCircuitToLayeredCircuitConverter.convert(circuit).expectation(
        [variable]
    )[variable] == pytest.approx(circuit.expectation([variable])[variable])


def test_mode_of_a_deterministic_circuit(deterministic_mixture):
    rx_mode = deterministic_mixture.log_mode()
    mode = RustworkxCircuitToLayeredCircuitConverter.convert(
        deterministic_mixture
    ).log_mode()
    assert mode.event == rx_mode.event
    assert mode.log_likelihood == pytest.approx(rx_mode.log_likelihood)


def test_mode_of_a_non_deterministic_circuit_is_intractable(overlapping_mixture):
    with pytest.raises(IntractableError):
        RustworkxCircuitToLayeredCircuitConverter.convert(
            overlapping_mixture
        ).log_mode()


def test_determinism_matches_the_rustworkx_answer(any_circuit):
    assert (
        RustworkxCircuitToLayeredCircuitConverter.convert(
            any_circuit
        ).is_deterministic()
        == any_circuit.is_deterministic()
    )


def test_decomposability(any_circuit):
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        any_circuit
    ).is_decomposable()


# %% sampling


def test_samples_lie_in_the_support(any_circuit):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit)
    samples = layered.sample(500)
    assert samples.shape == (500, len(layered.variables))
    assert not np.any(np.isnan(samples))
    assert np.all(layered.log_likelihood(samples) > -np.inf)


@pytest.mark.parametrize(
    "circuit_fixture", ["overlapping_mixture", "shared_children_circuit"]
)
def test_sample_mean_approximates_the_expectation(circuit_fixture, request):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(
        request.getfixturevalue(circuit_fixture)
    )
    samples = layered.sample(20000)
    expectation = layered.expectation()
    for index, variable in enumerate(layered.variables):
        assert float(samples[:, index].mean()) == pytest.approx(
            float(expectation[variable]), abs=0.05
        )


# %% truncation


def assert_same_truncation(
    rx_circuit: RxCircuit, event: Event, grid: np.ndarray
) -> LayeredProbabilisticCircuit | None:
    """
    Assert that a circuit and its conversion to layers truncate to the same
    distribution.

    :param rx_circuit: A circuit of units.
    :param event: The event to truncate both circuits to.
    :param grid: The points the truncated circuits are compared on.
    :return: The truncated layered circuit.
    """
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
    rx_truncation = rx_circuit.truncated(event.__deepcopy__())
    truncation = layered.truncated(event.__deepcopy__())

    assert truncation.probability == pytest.approx(rx_truncation.probability)
    if rx_truncation.distribution is None:
        assert truncation.distribution is None
        return None

    truncation.distribution.validate()
    np.testing.assert_allclose(
        truncation.distribution.log_likelihood(grid),
        rx_truncation.distribution.log_likelihood(grid),
        atol=1e-10,
    )
    return truncation.distribution


def staircase_of_boxes(
    x: Continuous, y: Continuous, number_of_boxes: int, lower: float, upper: float
) -> Event:
    """
    A staircase of disjoint boxes with gaps between them.

    Each box takes its own window of both variables, which stops the product algebra
    from merging them back into one simple set, and the gaps stop it from splitting them
    at shared corners.
    """
    width = (upper - lower) / (2 * number_of_boxes)
    result = None
    for index in range(number_of_boxes):
        start = lower + 2 * index * width
        box = SimpleEvent.from_data(
            {x: closed(start, start + width), y: closed(start, start + width)}
        ).as_composite_set()
        result = box if result is None else result | box
    return result


def truncate_one_simple_set_at_a_time(
    layered: LayeredProbabilisticCircuit,
    event: Event,
    monkeypatch: pytest.MonkeyPatch,
) -> Truncation[LayeredProbabilisticCircuit]:
    """
    Truncate through the path that handles one simple set at a time.

    The patch is applied to the class: the public truncation works on a copy of the
    circuit, so an attribute set on the instance would not reach it.
    """
    with monkeypatch.context() as patch:
        patch.setattr(
            LayeredProbabilisticCircuit,
            "can_truncate_in_one_batch",
            lambda self, events, singleton_allowed=False: False,
        )
        return layered.truncated(event)


def test_truncation_to_a_simple_event(uniform_circuit, x, y):
    event = SimpleEvent.from_data(
        {x: closed(0.5, 2.5), y: closed(0.5, 1.75)}
    ).as_composite_set()
    assert_same_truncation(uniform_circuit, event, grid_of(-1, 4, 25))


def test_truncation_to_a_composite_event(uniform_circuit, x):
    event = SimpleEvent.from_data(
        {x: closed(0.0, 0.5) | closed(2.0, 2.6)}
    ).as_composite_set()
    assert_same_truncation(uniform_circuit, event, grid_of(-1, 4, 25))


def test_truncation_to_an_event_with_many_simple_sets(uniform_circuit, x, y):
    event = staircase_of_boxes(x, y, 8, 0.0, 3.0)
    assert len(event.simple_sets) == 8
    assert_same_truncation(uniform_circuit, event, grid_of(-1, 4, 30))


def test_truncation_of_a_gaussian_circuit_produces_truncated_gaussian_layers(
    gaussian_circuit, x, y
):
    event = SimpleEvent.from_data(
        {x: closed(-2.0, 1.0), y: closed(-1.0, 3.0)}
    ).as_composite_set()
    truncated = assert_same_truncation(gaussian_circuit, event, grid_of(-4, 6, 25))
    assert any(isinstance(layer, TruncatedGaussianLayer) for layer in truncated.layers)


def test_truncation_of_a_mixed_circuit(mixed_circuit, x, count, letter):
    event = SimpleEvent.from_data(
        {
            x: closed(0.5, 3.0),
            count: closed(0, 0),
            letter: Set.from_iterable([Letter.A, Letter.B]),
        }
    ).as_composite_set()
    rx_truncation = mixed_circuit.truncated(event.__deepcopy__())
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        mixed_circuit
    ).truncated(event.__deepcopy__())
    assert truncation.probability == pytest.approx(rx_truncation.probability)
    samples = truncation.distribution.sample(200)
    np.testing.assert_allclose(
        truncation.distribution.log_likelihood(samples),
        rx_truncation.distribution.log_likelihood(samples),
    )


def test_the_batched_pass_agrees_with_truncating_once_per_simple_set(
    uniform_circuit, x, y, monkeypatch
):
    """
    Truncating to all simple sets in one pass and truncating to them one at a time and
    mixing the results describe the same distribution.
    """
    grid = grid_of(-1, 4, 30)
    event = staircase_of_boxes(x, y, 6, 0.0, 3.0)
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(uniform_circuit)

    batched_truncation = layered.truncated(event.__deepcopy__())
    separate_truncation = truncate_one_simple_set_at_a_time(
        layered, event.__deepcopy__(), monkeypatch
    )

    assert batched_truncation.probability == pytest.approx(
        separate_truncation.probability
    )
    np.testing.assert_allclose(
        batched_truncation.distribution.log_likelihood(grid),
        separate_truncation.distribution.log_likelihood(grid),
        atol=1e-10,
    )


def test_the_batched_pass_keeps_the_number_of_layers(
    shared_children_circuit, x, y, monkeypatch
):
    """
    The point of truncating to every simple set in one pass: the circuit keeps its
    layers and their blocks grow, instead of getting one set of layers per simple set.
    """
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(shared_children_circuit)
    event = staircase_of_boxes(x, y, 10, 0.0, 2.0)

    truncated = layered.truncated(event.__deepcopy__()).distribution
    separate = truncate_one_simple_set_at_a_time(
        layered, event.__deepcopy__(), monkeypatch
    ).distribution

    # one mixing layer on top of the circuit's own layers, against one whole set of
    # layers per simple set
    assert len(truncated.layers) <= len(layered.layers) + 1
    assert len(separate.layers) > 4 * len(truncated.layers)
    # the same distribution, held in taller blocks rather than in more layers
    assert truncated.number_of_nodes >= separate.number_of_nodes - len(separate.layers)


def test_a_gaussian_circuit_batches_truncation_to_several_simple_sets(
    gaussian_circuit, x, y
):
    event = staircase_of_boxes(x, y, 4, -1.0, 3.0)
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        gaussian_circuit
    ).can_truncate_in_one_batch(list(event.__deepcopy__().simple_sets))
    assert_same_truncation(gaussian_circuit, event, grid_of(-4, 6, 25))


def test_a_batch_whose_simple_sets_disagree_on_the_layer_type_reports_it(
    gaussian_circuit, x, y
):
    """
    A batch a layer cannot be truncated in is reported, so that the circuit falls back
    to truncating once per simple set.

    Here one simple set bounds *x* and turns the Gaussian leaves over it into truncated
    Gaussians, while the other leaves *x* unbounded and keeps them Gaussian.
    """
    event = (
        SimpleEvent.from_data(
            {x: closed(0.0, 1.0), y: closed(0.0, 1.0)}
        ).as_composite_set()
        | SimpleEvent.from_data({y: closed(2.0, 3.0)}).as_composite_set()
    )
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_circuit)
    event.fill_missing_variables(set(layered.variables))
    assert not layered.can_truncate_in_one_batch(list(event.simple_sets))
    assert_same_truncation(gaussian_circuit, event, grid_of(-4, 6, 25))


def test_truncation_puts_all_mass_inside_the_event(overlapping_mixture, x, y):
    event = SimpleEvent.from_data(
        {x: closed(0.5, 2.5), y: closed(0.5, 1.75)}
    ).as_composite_set()
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(overlapping_mixture)
        .truncated(event.__deepcopy__())
        .distribution
    )
    assert truncated.probability(event.__deepcopy__()) == pytest.approx(1.0)


def test_truncation_to_an_impossible_event(deterministic_mixture, x):
    event = SimpleEvent.from_data({x: closed(10.0, 11.0)}).as_composite_set()
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        deterministic_mixture
    ).truncated(event)
    assert truncation.distribution is None
    assert truncation.probability == 0.0


def test_truncation_removes_the_impossible_nodes(deterministic_mixture, x):
    event = SimpleEvent.from_data({x: closed(0.0, 1.0)}).as_composite_set()
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        deterministic_mixture
    ).truncated(event)
    assert truncation.probability == pytest.approx(0.3)
    # the branch over [2, 3] x [2, 4] is gone
    for layer in truncation.distribution.layers:
        if isinstance(layer, UniformLayer):
            assert layer.number_of_nodes == 1


def test_truncation_does_not_change_the_original(deterministic_mixture, x):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(deterministic_mixture)
    before = layered.number_of_nodes
    layered.truncated(SimpleEvent.from_data({x: closed(0.0, 1.0)}).as_composite_set())
    assert layered.number_of_nodes == before


# %% conditioning


@pytest.mark.parametrize(
    "circuit_fixture, expected_probability",
    [("overlapping_mixture", 0.4), ("deterministic_mixture", 0.3)],
)
def test_conditioning_on_a_continuous_variable(
    circuit_fixture, expected_probability, request, x
):
    # p(x = 0.75) is the weight of the only branch whose support contains 0.75
    rx_circuit = request.getfixturevalue(circuit_fixture)
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
    point = {x: 0.75}
    rx_conditional = rx_circuit.conditional(point).distribution
    conditioning = layered.conditional(point)

    assert conditioning.likelihood == pytest.approx(expected_probability)
    assert list(conditioning.distribution.variables) == list(layered.variables)
    grid = np.stack([np.full(40, 0.75), np.linspace(-1, 4, 40)], axis=-1)
    np.testing.assert_allclose(
        conditioning.distribution.log_likelihood(grid),
        rx_conditional.log_likelihood(grid),
        atol=1e-10,
    )


def test_the_conditional_density_integrates_to_one(uniform_circuit, x):
    grid = np.linspace(-1, 5, 60001)
    conditional = (
        RustworkxCircuitToLayeredCircuitConverter.convert(uniform_circuit)
        .conditional({x: 0.75})
        .distribution
    )
    density = conditional.likelihood(
        np.stack([np.full_like(grid, 0.75), grid], axis=-1)
    )
    assert float(np.trapezoid(density, grid)) == pytest.approx(1.0, abs=1e-3)


def test_conditioning_a_circuit_with_shared_leaves(shared_children_circuit, x):
    """
    ``p(x=0.75, y) = 0.4 * sum_y_1(y) + 0.15 * sum_y_2(y)``, which is ``0.255`` on
    ``(0, 1)`` and ``0.295`` on ``(1, 2)``. Dividing by ``p(x=0.75) = 0.55`` gives the
    conditional density.
    """
    conditioning = RustworkxCircuitToLayeredCircuitConverter.convert(
        shared_children_circuit
    ).conditional({x: 0.75})
    assert conditioning.likelihood == pytest.approx(0.55)
    np.testing.assert_allclose(
        conditioning.distribution.likelihood(np.array([[0.75, 0.5], [0.75, 1.5]])),
        np.array([0.255 / 0.55, 0.295 / 0.55]),
    )


def test_conditioning_on_an_impossible_point(deterministic_mixture, x):
    conditioning = RustworkxCircuitToLayeredCircuitConverter.convert(
        deterministic_mixture
    ).conditional({x: 1.5})
    assert conditioning.distribution is None
    assert conditioning.likelihood == 0.0


def test_conditioning_on_a_symbolic_variable(mixed_circuit, letter):
    point = {letter: Letter.B}
    rx_conditioning = mixed_circuit.log_conditional(point)
    conditioning = RustworkxCircuitToLayeredCircuitConverter.convert(
        mixed_circuit
    ).log_conditional(point)
    assert conditioning.log_likelihood == pytest.approx(rx_conditioning.log_likelihood)
    samples = conditioning.distribution.sample(200)
    np.testing.assert_allclose(
        conditioning.distribution.log_likelihood(samples),
        rx_conditioning.distribution.log_likelihood(samples),
    )


def test_conditioning_on_an_integer_variable(mixed_circuit, x, count, letter):
    point = {count: 2}
    rx_conditioning = mixed_circuit.log_conditional(point)
    conditioning = RustworkxCircuitToLayeredCircuitConverter.convert(
        mixed_circuit
    ).log_conditional(point)

    # only the second component has n = 2
    assert conditioning.log_likelihood == pytest.approx(np.log(0.6 * 0.3))
    assert conditioning.log_likelihood == pytest.approx(rx_conditioning.log_likelihood)
    # the point mass is an integer layer, so it answers interval events of n
    assert any(
        isinstance(layer, IntegerLayer) for layer in conditioning.distribution.layers
    )
    event = SimpleEvent.from_data(
        {
            x: closed(1.5, 3.0),
            count: closed(1, 3),
            letter: Set.from_iterable([Letter.A]),
        }
    )
    assert conditioning.distribution.probability_of_simple_event(
        event
    ) == pytest.approx(rx_conditioning.distribution.probability_of_simple_event(event))


def test_conditioning_on_every_variable(deterministic_mixture, x, y):
    conditioning = RustworkxCircuitToLayeredCircuitConverter.convert(
        deterministic_mixture
    ).conditional({x: 0.5, y: 0.5})
    assert conditioning.likelihood == pytest.approx(0.3)
    np.testing.assert_allclose(
        conditioning.distribution.log_likelihood(np.array([[0.5, 0.5]])),
        np.array([0.0]),
    )


# %% marginals


@pytest.mark.parametrize(
    "circuit_fixture", ["overlapping_mixture", "deterministic_mixture"]
)
def test_marginal_of_one_variable(circuit_fixture, request, x):
    rx_circuit = request.getfixturevalue(circuit_fixture)
    marginal = RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit).marginal(
        [x]
    )
    assert list(marginal.variables) == [x]
    grid = np.linspace(-1, 4, 60).reshape(-1, 1)
    np.testing.assert_allclose(
        marginal.log_likelihood(grid),
        rx_circuit.marginal([x]).log_likelihood(grid),
        atol=1e-10,
    )


def test_marginal_of_a_circuit_with_shared_leaves(shared_children_circuit, x):
    """
    ``p(x) = 0.5 * sum_x_1(x) + 0.5 * sum_x_2(x)``, which is ``0.5 * 0.8 + 0.5 * 0.3``
    on ``(0, 1)`` and ``0.5 * 0.2 + 0.5 * 0.7`` on ``(1, 2)``.
    """
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(shared_children_circuit)
    marginal = layered.marginal([x])
    np.testing.assert_allclose(
        marginal.likelihood(np.array([[0.5], [1.5]])), np.array([0.55, 0.45])
    )

    # the same density, obtained by integrating y out of the joint
    grid = np.linspace(0, 2, 20001)
    for value, expected in ((0.5, 0.55), (1.5, 0.45)):
        joint = layered.likelihood(np.stack([np.full_like(grid, value), grid], axis=-1))
        assert float(np.trapezoid(joint, grid)) == pytest.approx(expected, abs=1e-3)


def test_marginal_of_a_discrete_variable(mixed_circuit, count):
    marginal = RustworkxCircuitToLayeredCircuitConverter.convert(
        mixed_circuit
    ).marginal([count])
    assert list(marginal.variables) == [count]
    grid = np.array([[0], [1], [2], [3]])
    np.testing.assert_allclose(
        marginal.log_likelihood(grid),
        mixed_circuit.marginal([count]).log_likelihood(grid),
    )


def test_marginal_of_a_variable_that_is_not_modeled(deterministic_mixture, z):
    assert (
        RustworkxCircuitToLayeredCircuitConverter.convert(
            deterministic_mixture
        ).marginal([z])
        is None
    )


def test_marginal_does_not_change_the_original(deterministic_mixture, x, y):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(deterministic_mixture)
    layered.marginal([x])
    assert list(layered.variables) == [x, y]


# %% serialization


def test_json_round_trip(any_circuit):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit)
    restored = from_json(to_json(layered))
    assert isinstance(restored, LayeredProbabilisticCircuit)
    assert list(restored.variables) == list(layered.variables)
    samples = any_circuit.sample(200)
    np.testing.assert_allclose(
        restored.log_likelihood(samples), layered.log_likelihood(samples)
    )


def test_json_round_trip_of_a_truncated_gaussian_layer(gaussian_circuit, x, y):
    event = SimpleEvent.from_data(
        {x: closed(-2.0, 1.0), y: closed(-1.0, 3.0)}
    ).as_composite_set()
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_circuit)
        .truncated(event)
        .distribution
    )
    restored = from_json(to_json(truncated))
    samples = truncated.sample(200)
    np.testing.assert_allclose(
        restored.log_likelihood(samples), truncated.log_likelihood(samples)
    )


def test_json_round_trip_of_a_conditioned_circuit(overlapping_mixture, x):
    # conditioning attaches a Dirac delta layer per conditioned variable under a new
    # product root
    conditioned = (
        RustworkxCircuitToLayeredCircuitConverter.convert(overlapping_mixture)
        .conditional({x: 0.5})
        .distribution
    )
    assert isinstance(conditioned.root, ProductLayer)
    assert any(isinstance(layer, DiracDeltaLayer) for layer in conditioned.layers)

    restored = from_json(to_json(conditioned))
    samples = conditioned.sample(200)
    np.testing.assert_allclose(
        restored.log_likelihood(samples), conditioned.log_likelihood(samples)
    )


# %% transformations


@pytest.mark.parametrize(
    "transform, upper_end_of_grid",
    [
        (lambda circuit, x, y: circuit.apply_translation({x: 1.5, y: -0.5}), 6),
        (lambda circuit, x, y: circuit.apply_scaling({x: 2.0, y: 3.0}), 14),
    ],
    ids=["translation", "scaling"],
)
def test_transformation_matches_rustworkx(
    deterministic_mixture, x, y, transform, upper_end_of_grid
):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(deterministic_mixture)
    transform(deterministic_mixture, x, y)
    transform(layered, x, y)
    grid = grid_of(-2, upper_end_of_grid, 20)
    np.testing.assert_allclose(
        layered.log_likelihood(grid), deterministic_mixture.log_likelihood(grid)
    )


def test_renaming_variables_with_a_prefix(deterministic_mixture):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(deterministic_mixture)
    samples = np.array([[0.5, 0.5], [2.5, 3.0]])
    before = layered.log_likelihood(samples)

    layered.rename_variables_with_prefix("world")
    assert [variable.name for variable in layered.variables] == ["world.x", "world.y"]
    np.testing.assert_allclose(layered.log_likelihood(samples), before)


def test_renaming_keeps_the_column_order(deterministic_mixture, x, y):
    # renaming may reorder the variables, and the layers have to follow
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(deterministic_mixture)
    samples = np.array([[0.5, 0.7], [2.5, 3.0]])
    before = layered.log_likelihood(samples)

    layered.update_variables({x: Continuous("z"), y: Continuous("a")})
    assert [variable.name for variable in layered.variables] == ["a", "z"]
    np.testing.assert_allclose(layered.log_likelihood(samples[:, ::-1]), before)


# %% simplification


def test_normalize_makes_the_weights_sum_to_one(overlapping_mixture):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(overlapping_mixture)
    root = layered.root
    root.log_weights.data = root.log_weights.data + 3.0
    samples = np.array([[0.5, 0.5], [2.0, 1.5]])
    before = layered.log_likelihood(samples)

    layered.normalize()
    np.testing.assert_allclose(
        root.log_normalization_constants, np.zeros(root.number_of_nodes), atol=1e-12
    )
    np.testing.assert_allclose(layered.log_likelihood(samples), before)


def test_simplify_keeps_the_likelihood(deterministic_mixture):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(deterministic_mixture)
    samples = np.array([[0.5, 0.5], [2.5, 3.0]])
    before = layered.log_likelihood(samples)
    number_of_layers = len(layered.layers)

    layered.simplify()
    assert len(layered.layers) <= number_of_layers
    np.testing.assert_allclose(layered.log_likelihood(samples), before)


# %% symbolic encoding


def test_encoding_maps_hashes_to_positions_and_back(sparse_letter):
    encoding = SymbolicEncoding(sparse_letter)
    hashes = np.array([hash(element) for element in SparseLetter])
    positions = encoding.indices_of_hashes(hashes)
    np.testing.assert_array_equal(positions, np.arange(len(SparseLetter)))
    np.testing.assert_array_equal(encoding.hashes_of_indices(positions), hashes)


def test_a_value_outside_of_the_domain_has_no_position(sparse_letter):
    encoding = SymbolicEncoding(sparse_letter)
    np.testing.assert_array_equal(
        encoding.indices_of_hashes(np.array([5.0, np.nan])), [-1, -1]
    )


@pytest.mark.parametrize(
    "domain", [SymbolWithLargeHash, SymbolWithHashBeyondFloatPrecision]
)
def test_a_distribution_over_elements_with_large_hashes_keeps_its_probabilities(
    domain,
):
    variable = Symbolic(name="large_hash", domain=Set.from_iterable(domain))
    distribution = symbolic_distribution(
        variable, {domain.FIRST: 0.25, domain.SECOND: 0.75}
    )
    layer = SymbolicLayer.from_distributions(0, [distribution])
    assert dict(layer.node_distribution(0, variable).probabilities) == dict(
        distribution.probabilities
    )


def test_the_support_of_a_symbolic_circuit_agrees_with_rustworkx():
    variable = Symbolic(
        name="beyond_float_precision",
        domain=Set.from_iterable(SymbolWithHashBeyondFloatPrecision),
    )
    circuit = RxCircuit()
    leaf(
        symbolic_distribution(
            variable,
            {
                SymbolWithHashBeyondFloatPrecision.FIRST: 0.25,
                SymbolWithHashBeyondFloatPrecision.SECOND: 0.75,
            },
        ),
        circuit,
    )
    assert (
        RustworkxCircuitToLayeredCircuitConverter.convert(circuit).support
        == circuit.support
    )


def test_symbolic_layer_stores_positions_in_the_domain(sparse_symbolic_circuit):
    [symbolic_layer] = [
        layer
        for layer in RustworkxCircuitToLayeredCircuitConverter.convert(
            sparse_symbolic_circuit
        ).layers
        if isinstance(layer, SymbolicLayer)
    ]
    assert set(symbolic_layer.states.tolist()) <= set(range(len(SparseLetter)))


def test_log_likelihood_of_a_sparse_symbolic_circuit_agrees_with_rustworkx(
    sparse_symbolic_circuit,
):
    events = np.array(
        [
            [1.5, hash(SparseLetter.LOW)],
            [1.5, hash(SparseLetter.MIDDLE)],
            [2.5, hash(SparseLetter.HIGH)],
            [1.5, 5.0],
        ]
    )
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(
            sparse_symbolic_circuit
        ).log_likelihood(events),
        sparse_symbolic_circuit.log_likelihood(events),
    )


def test_samples_are_hashes_of_domain_elements(sparse_symbolic_circuit, sparse_letter):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(sparse_symbolic_circuit)
    samples = layered.sample(500)
    column = layered.variables.index(sparse_letter)
    assert set(samples[:, column].tolist()) <= {
        float(hash(element)) for element in SparseLetter
    }
    assert np.isfinite(sparse_symbolic_circuit.log_likelihood(samples)).all()


def test_probability_of_an_event_of_a_sparse_symbolic_circuit_agrees_with_rustworkx(
    sparse_symbolic_circuit, x, sparse_letter
):
    event = SimpleEvent.from_data(
        {
            x: closed(0.5, 2.5),
            sparse_letter: Set.from_iterable([SparseLetter.LOW, SparseLetter.MIDDLE]),
        }
    )
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        sparse_symbolic_circuit
    ).probability_of_simple_event(event) == pytest.approx(
        sparse_symbolic_circuit.probability_of_simple_event(event)
    )


def test_conditioning_a_sparse_symbolic_circuit_agrees_with_rustworkx(
    sparse_symbolic_circuit, sparse_letter
):
    point = {sparse_letter: SparseLetter.HIGH}
    rx_conditioning = sparse_symbolic_circuit.log_conditional(point)
    conditioning = RustworkxCircuitToLayeredCircuitConverter.convert(
        sparse_symbolic_circuit
    ).log_conditional(point)
    assert conditioning.log_likelihood == pytest.approx(rx_conditioning.log_likelihood)
    samples = conditioning.distribution.sample(200)
    np.testing.assert_allclose(
        conditioning.distribution.log_likelihood(samples),
        rx_conditioning.distribution.log_likelihood(samples),
    )


# %% choice of probability table


@pytest.fixture(params=["mixed_circuit", "sparse_symbolic_circuit"])
def discrete_circuit(request) -> RxCircuit:
    """
    Every circuit of this module with discrete leaves.
    """
    return request.getfixturevalue(request.param)


def test_every_discrete_layer_is_stored_sparsely(discrete_circuit):
    sparse = RustworkxCircuitToLayeredCircuitConverter.convert(
        discrete_circuit
    ).store_discrete_probabilities_as(SparseProbabilityTable)
    discrete_layers = [
        layer for layer in sparse.layers if isinstance(layer, DiscreteLayer)
    ]
    assert len(discrete_layers) > 0
    for layer in discrete_layers:
        assert isinstance(layer.table, SparseProbabilityTable)


def test_sparse_and_dense_tables_answer_queries_alike(discrete_circuit, x):
    dense = RustworkxCircuitToLayeredCircuitConverter.convert(discrete_circuit)
    sparse = RustworkxCircuitToLayeredCircuitConverter.convert(
        discrete_circuit
    ).store_discrete_probabilities_as(SparseProbabilityTable)
    samples = dense.sample(300)
    np.testing.assert_allclose(
        sparse.log_likelihood(samples), dense.log_likelihood(samples)
    )
    event = SimpleEvent.from_data({x: closed(0.5, 2.5)})
    event.fill_missing_variables(dense.variables)
    assert sparse.probability_of_simple_event(event) == pytest.approx(
        dense.probability_of_simple_event(event)
    )
    dense_truncation = dense.truncated(event.as_composite_set())
    sparse_truncation = sparse.truncated(event.as_composite_set())
    assert sparse_truncation.probability == pytest.approx(dense_truncation.probability)
    np.testing.assert_allclose(
        sparse_truncation.distribution.log_likelihood(samples),
        dense_truncation.distribution.log_likelihood(samples),
    )


def test_samples_of_a_sparse_circuit_are_likely_under_the_dense_one(discrete_circuit):
    dense = RustworkxCircuitToLayeredCircuitConverter.convert(discrete_circuit)
    sparse = RustworkxCircuitToLayeredCircuitConverter.convert(
        discrete_circuit
    ).store_discrete_probabilities_as(SparseProbabilityTable)
    assert np.isfinite(dense.log_likelihood(sparse.sample(300))).all()


# %% layers outside of a circuit


def test_uniform_layer_likelihood():
    layer = uniform_layer_of(0, [(0, 1), (1, 3)])
    assert layer.number_of_nodes == 2
    np.testing.assert_allclose(
        layer.log_likelihood_of_nodes(np.array([[0.5], [2.0], [5.0]])),
        np.array([[0.0, -np.inf], [-np.inf, -np.log(2)], [-np.inf, -np.inf]]),
    )


def test_dirac_delta_layer_likelihood():
    layer = DiracDeltaLayer(0, np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        layer.log_likelihood_of_nodes(np.array([[0.0], [1.0], [2.0]])),
        np.array([[0.0, -np.inf], [-np.inf, np.log(2)], [-np.inf, -np.inf]]),
    )


def test_number_of_parameters(overlapping_mixture):
    # two weights on the root plus two bounds for each of the four uniform nodes
    assert (
        RustworkxCircuitToLayeredCircuitConverter.convert(
            overlapping_mixture
        ).number_of_parameters
        == 2 + 4 * 2
    )


def test_layers_visits_parents_first(shared_children_circuit):
    order = RustworkxCircuitToLayeredCircuitConverter.convert(
        shared_children_circuit
    ).layers
    positions = {id(layer): index for index, layer in enumerate(order)}
    for layer in order:
        for child_layer in layer.child_layers:
            assert positions[id(layer)] < positions[id(child_layer)]


def test_validate_detects_a_shape_mismatch():
    with pytest.raises(ShapeMismatchError):
        DiracDeltaLayer(0, np.array([0.0, 1.0]), np.array([1.0])).validate()


def test_gaussian_layer_likelihood(x):
    layer = GaussianLayer(0, np.array([0.0, 2.0]), np.array([1.0, 0.5]))
    points = np.array([[0.0], [2.0]])
    result = layer.log_likelihood_of_nodes(points)
    for node in range(2):
        np.testing.assert_allclose(
            result[:, node], layer.node_distribution(node, x).log_likelihood(points)
        )


def test_symbolic_layer_likelihood(letter):
    distributions = [
        symbolic_distribution(letter, dict(zip(Letter, probabilities)))
        for probabilities in ((0.5, 0.25, 0.25), (0.1, 0.6, 0.3))
    ]
    layer = SymbolicLayer.from_distributions(0, distributions)
    hashes = np.array([[hash(element)] for element in letter.domain.simple_sets])
    # a layer reads the positions of the domain elements, which the circuit encodes
    positions = SymbolicEncoding(letter).indices_of_hashes(hashes[:, 0]).reshape(-1, 1)
    result = layer.log_likelihood_of_nodes(positions)
    for node, distribution in enumerate(distributions):
        np.testing.assert_allclose(result[:, node], distribution.log_likelihood(hashes))


def test_a_symbolic_layer_has_no_cumulative_distribution(mixed_circuit):
    [symbolic_layer] = [
        layer
        for layer in RustworkxCircuitToLayeredCircuitConverter.convert(
            mixed_circuit
        ).layers
        if isinstance(layer, SymbolicLayer)
    ]
    with pytest.raises(UndefinedCumulativeDistributionError):
        symbolic_layer.cumulative_distribution_of_nodes(np.zeros((1, 3)))


def test_integer_layer_cumulative_distribution(count):
    distribution = IntegerDistribution(
        variable=count, probabilities=MissingDict(float, {0: 0.2, 1: 0.3, 2: 0.5})
    )
    layer = IntegerLayer.from_distributions(0, [distribution])
    points = np.array([[-1], [0], [1], [2], [3]])
    np.testing.assert_allclose(
        layer.cumulative_distribution_of_nodes(points)[:, 0],
        distribution.cumulative_distribution_function(points),
    )


# %% support of point masses


@pytest.fixture
def mixture_of_point_masses(x, y) -> RxCircuit:
    """
    A mixture of products of Dirac deltas at locations an interval cannot hold exactly.
    """
    generator = np.random.default_rng(0)
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for _ in range(20):
        product = ProductUnit(probabilistic_circuit=circuit)
        for variable in (x, y):
            product.add_subcircuit(
                leaf(
                    DiracDeltaDistribution(
                        variable=variable, location=float(generator.uniform(0, 5))
                    ),
                    circuit,
                )
            )
        root.add_subcircuit(product, math.log(1 / 20))
    return circuit


def test_the_support_of_point_masses_has_probability_one(mixture_of_point_masses):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(mixture_of_point_masses)
    assert layered.probability(layered.support) == pytest.approx(1.0, abs=1e-9)


def test_every_region_of_the_support_of_point_masses_agrees_with_rustworkx(
    mixture_of_point_masses,
):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(mixture_of_point_masses)
    for simple_set in mixture_of_point_masses.support.simple_sets:
        assert layered.probability(simple_set.as_composite_set()) == pytest.approx(
            mixture_of_point_masses.probability(simple_set.as_composite_set()),
            abs=1e-9,
        )
