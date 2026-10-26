"""
Cross validation of the multivariate Gaussian layers against the multivariate leaves of
the ``rx`` package.

The probabilities of boxes under a correlated Gaussian are integrated numerically by
:mod:`scipy.stats.multivariate_normal`, whose quasi Monte Carlo estimates differ from call
to call by about ``1e-5``, so every comparison that involves one allows for that.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from random_events.interval import closed, open as open_interval, reals, singleton
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import Continuous
from scipy.stats import norm

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.distributions.truncated_multivariate_gaussian import (
    MomentIntegration,
)
from probabilistic_model.exceptions import NoClosedFormError
from probabilistic_model.learning.gaussian_mixture.gaussian_mixture_model import (
    GaussianMixtureModel,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RxCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.multivariate_gaussian_layer import (
    MultivariateGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.truncated_multivariate_gaussian_layer import (
    TruncatedMultivariateGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from ..circuit_building import uniform_leaf

# %% circuits


@pytest.fixture
def numerical_integration_tolerance() -> float:
    """
    The absolute error allowed for a probability that was integrated numerically.
    """
    return 1e-4


@pytest.fixture
def first_component(x, y) -> MultivariateGaussianDistribution:
    """
    A correlated Gaussian whose variables are given in the reverse of the order the
    circuit sorts them in, so that a conversion that confused the two orders would swap
    them.
    """
    return MultivariateGaussianDistribution(
        variables=(y, x),
        mean=np.array([1.0, -0.5]),
        covariance=Covariance.from_matrix([[1.0, 0.6], [0.6, 2.0]]),
    )


@pytest.fixture
def second_component(x, y) -> MultivariateGaussianDistribution:
    """
    A Gaussian that is negatively correlated, with its variables in the circuit's order.
    """
    return MultivariateGaussianDistribution(
        variables=(x, y),
        mean=np.array([2.0, 1.5]),
        covariance=Covariance.from_matrix([[0.5, -0.2], [-0.2, 0.8]]),
    )


@pytest.fixture
def gaussian_mixture(first_component, second_component) -> RxCircuit:
    """
    A mixture of two correlated Gaussians over *x* and *y*.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(leaf(first_component, circuit), np.log(0.3))
    root.add_subcircuit(leaf(second_component, circuit), np.log(0.7))
    return circuit


@pytest.fixture
def single_gaussian(first_component) -> RxCircuit:
    """
    The first component on its own, which is deterministic.
    """
    circuit = RxCircuit()
    leaf(first_component, circuit)
    return circuit


@pytest.fixture
def gaussian_mixture_times_uniform(first_component, second_component, z) -> RxCircuit:
    """
    A mixture of products of a correlated Gaussian over *x* and *y* and a uniform
    distribution over *z*.
    """
    circuit = RxCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, component, (lower, upper) in (
        (0.4, first_component, (0.0, 1.0)),
        (0.6, second_component, (0.5, 3.0)),
    ):
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(weight))
        product.add_subcircuit(leaf(component, circuit))
        product.add_subcircuit(uniform_leaf(z, lower, upper, circuit))
    return circuit


@pytest.fixture
def three_variable_gaussian(x, y, z) -> RxCircuit:
    """
    One Gaussian over three correlated variables.
    """
    circuit = RxCircuit()
    leaf(
        MultivariateGaussianDistribution(
            variables=(z, x, y),
            mean=np.array([0.5, -1.0, 2.0]),
            covariance=Covariance.from_matrix(
                [[1.0, 0.3, -0.2], [0.3, 1.5, 0.4], [-0.2, 0.4, 0.7]]
            ),
        ),
        circuit,
    )
    return circuit


@pytest.fixture(
    params=[
        "gaussian_mixture",
        "single_gaussian",
        "gaussian_mixture_times_uniform",
        "three_variable_gaussian",
    ]
)
def any_circuit(request) -> RxCircuit:
    """
    Every circuit of this module.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture
def box_of_x_and_y(x, y) -> Event:
    """
    The box most truncations of this module use.
    """
    return SimpleEvent.from_data(
        {x: closed(-1.0, 2.0), y: closed(0.0, 1.5)}
    ).as_composite_set()


# %% conversion


def test_multivariate_gaussian_leaves_become_one_layer(gaussian_mixture):
    [gaussian_layer] = RustworkxCircuitToLayeredCircuitConverter.convert(
        gaussian_mixture
    ).root.child_layers
    assert isinstance(gaussian_layer, MultivariateGaussianLayer)
    assert gaussian_layer.number_of_nodes == 2
    np.testing.assert_array_equal(gaussian_layer.variables, [0, 1])


def test_the_parameters_are_laid_out_by_the_variables_of_the_circuit(
    single_gaussian, first_component, x, y
):
    gaussian_layer = RustworkxCircuitToLayeredCircuitConverter.convert(
        single_gaussian
    ).root
    in_circuit_order = [first_component.index_of(x), first_component.index_of(y)]
    np.testing.assert_allclose(
        gaussian_layer.mean[0], first_component.mean[in_circuit_order]
    )
    np.testing.assert_allclose(
        gaussian_layer.covariance.matrices[0],
        first_component.covariance.matrix[np.ix_(in_circuit_order, in_circuit_order)],
    )


def test_round_trip_through_rustworkx_keeps_the_likelihood(any_circuit):
    samples = any_circuit.sample(200)
    np.testing.assert_allclose(
        LayeredCircuitToRustworkxCircuitConverter.convert(
            RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit)
        ).log_likelihood(samples),
        any_circuit.log_likelihood(samples),
    )


def test_a_learned_gaussian_mixture_converts():
    generator = np.random.default_rng(0)
    data = pd.DataFrame(
        np.concatenate(
            [
                generator.multivariate_normal(
                    [0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=200
                ),
                generator.multivariate_normal(
                    [6.0, -4.0], [[0.5, -0.2], [-0.2, 2.0]], size=100
                ),
            ]
        ),
        columns=["y", "x"],
    )
    rx_circuit = GaussianMixtureModel().fit(data)
    samples = rx_circuit.sample(300)
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit).log_likelihood(
            samples
        ),
        rx_circuit.log_likelihood(samples),
    )


# %% queries that leave the circuit as it is


def test_log_likelihood(any_circuit):
    samples = any_circuit.sample(500)
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit).log_likelihood(
            samples
        ),
        any_circuit.log_likelihood(samples),
    )


def test_cumulative_distribution_function(any_circuit, numerical_integration_tolerance):
    samples = any_circuit.sample(20)
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(
            any_circuit
        ).cumulative_distribution_function(samples),
        any_circuit.cumulative_distribution_function(samples),
        atol=numerical_integration_tolerance,
    )


def test_probability_of_a_box(any_circuit, numerical_integration_tolerance, x, y, z):
    event = SimpleEvent.from_data(
        {x: closed(-1.0, 2.0), y: closed(0.0, 1.5), z: closed(0.2, 2.0)}
    )
    event.fill_missing_variables(set(any_circuit.variables))
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        any_circuit
    ).probability_of_simple_event(event) == pytest.approx(
        any_circuit.probability_of_simple_event(event),
        abs=numerical_integration_tolerance,
    )


@pytest.mark.parametrize(
    "x_interval, y_interval",
    [
        (closed(0.0, 1.0), reals()),
        (closed(-2.0, 0.0) | closed(1.0, 3.0), closed(-1.0, 2.0)),
    ],
    ids=["one variable bounded", "several intervals per variable"],
)
def test_probability_of_a_box_of_the_mixture(
    gaussian_mixture, numerical_integration_tolerance, x, y, x_interval, y_interval
):
    event = SimpleEvent.from_data({x: x_interval, y: y_interval})
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        gaussian_mixture
    ).probability_of_simple_event(event) == pytest.approx(
        gaussian_mixture.probability_of_simple_event(event),
        abs=numerical_integration_tolerance,
    )


def test_a_singleton_has_no_probability(gaussian_mixture, x, y):
    event = SimpleEvent.from_data({x: singleton(0.0), y: reals()})
    assert (
        RustworkxCircuitToLayeredCircuitConverter.convert(
            gaussian_mixture
        ).probability_of_simple_event(event)
        == 0.0
    )


def test_moment(any_circuit):
    expectation = any_circuit.expectation(any_circuit.variables)
    order = VariableMap({variable: 2 for variable in any_circuit.variables})
    expected = any_circuit.moment(order, expectation)
    moment = RustworkxCircuitToLayeredCircuitConverter.convert(any_circuit).moment(
        order, expectation
    )
    for variable in any_circuit.variables:
        assert moment[variable] == pytest.approx(expected[variable])


def test_expectation_next_to_another_leaf(
    gaussian_mixture_times_uniform, first_component, second_component, x, z
):
    expectation = RustworkxCircuitToLayeredCircuitConverter.convert(
        gaussian_mixture_times_uniform
    ).expectation([x, z])
    assert expectation[x] == pytest.approx(
        0.4 * first_component.mean[first_component.index_of(x)]
        + 0.6 * second_component.mean[second_component.index_of(x)]
    )
    # the means of the uniform distributions over [0, 1] and [0.5, 3]
    assert expectation[z] == pytest.approx(0.4 * 0.5 + 0.6 * 1.75)


@pytest.mark.parametrize(
    "kept_names",
    [["x"], ["z"], ["x", "y"], ["y", "z"]],
    ids=lambda names: "".join(names),
)
def test_marginal(three_variable_gaussian, kept_names):
    kept = [
        variable
        for variable in three_variable_gaussian.variables
        if variable.name in kept_names
    ]
    samples = three_variable_gaussian.sample(100)
    columns = [three_variable_gaussian.variables.index(variable) for variable in kept]
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(three_variable_gaussian)
        .marginal(kept)
        .log_likelihood(samples[:, columns]),
        three_variable_gaussian.marginal(kept).log_likelihood(samples[:, columns]),
    )


def test_mode_is_the_mean(single_gaussian):
    log_mode = RustworkxCircuitToLayeredCircuitConverter.convert(
        single_gaussian
    ).log_mode()
    expected_mode = single_gaussian.log_mode()
    assert log_mode.event == expected_mode.event
    assert log_mode.log_likelihood == pytest.approx(expected_mode.log_likelihood)


def test_samples_have_the_mean_of_the_circuit(gaussian_mixture):
    expectation = gaussian_mixture.expectation(gaussian_mixture.variables)
    np.testing.assert_allclose(
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .sample(20000)
        .mean(axis=0),
        [expectation[variable] for variable in gaussian_mixture.variables],
        atol=0.05,
    )


def test_samples_have_the_covariance_of_the_gaussian(
    single_gaussian, first_component, x, y
):
    in_circuit_order = [first_component.index_of(x), first_component.index_of(y)]
    np.testing.assert_allclose(
        np.cov(
            RustworkxCircuitToLayeredCircuitConverter.convert(single_gaussian)
            .sample(20000)
            .T
        ),
        first_component.covariance.matrix[np.ix_(in_circuit_order, in_circuit_order)],
        atol=0.05,
    )


def test_the_support_is_the_whole_space(gaussian_mixture):
    assert (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture).support
        == gaussian_mixture.support
    )


# %% truncation


@pytest.mark.parametrize(
    "circuit_fixture",
    ["gaussian_mixture", "single_gaussian", "gaussian_mixture_times_uniform"],
)
def test_truncation_to_a_box(
    circuit_fixture, request, box_of_x_and_y, numerical_integration_tolerance
):
    rx_circuit = request.getfixturevalue(circuit_fixture)
    rx_truncation = rx_circuit.truncated(box_of_x_and_y.__deepcopy__())
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        rx_circuit
    ).truncated(box_of_x_and_y.__deepcopy__())
    assert truncation.probability == pytest.approx(
        rx_truncation.probability, abs=numerical_integration_tolerance
    )
    assert any(
        isinstance(layer, TruncatedMultivariateGaussianLayer)
        for layer in truncation.distribution.layers
    )
    samples = rx_truncation.distribution.sample(200)
    np.testing.assert_allclose(
        truncation.distribution.log_likelihood(samples),
        rx_truncation.distribution.log_likelihood(samples),
        atol=1e-3,
    )


def test_likelihood_outside_the_box_is_minus_infinity(gaussian_mixture, x, y):
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(
            SimpleEvent.from_data(
                {x: closed(-1.0, 2.0), y: open_interval(0.0, 1.5)}
            ).as_composite_set()
        )
        .distribution
    )
    outside = np.array([[3.0, 1.0], [0.0, 1.5], [0.0, 0.0]])
    assert np.all(np.isneginf(truncated.log_likelihood(outside)))


def test_truncation_to_the_whole_space_keeps_the_gaussians(gaussian_mixture, x, y):
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        gaussian_mixture
    ).truncated(SimpleEvent.from_data({x: reals(), y: reals()}).as_composite_set())
    assert truncation.probability == pytest.approx(1.0)
    assert any(
        isinstance(layer, MultivariateGaussianLayer)
        for layer in truncation.distribution.layers
    )


def test_truncation_to_an_impossible_event(gaussian_mixture, x, y):
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        gaussian_mixture
    ).truncated(
        SimpleEvent.from_data({x: singleton(0.0), y: reals()}).as_composite_set()
    )
    assert truncation.distribution is None
    assert truncation.probability == 0.0


def test_truncation_to_a_box_far_in_the_tail_keeps_its_probability(
    single_gaussian, first_component, x, y
):
    position = first_component.index_of(y)
    mean = first_component.mean[position]
    deviation = np.sqrt(first_component.covariance.matrix[position, position])
    lower, upper = mean + 9 * deviation, mean + 10 * deviation
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        single_gaussian
    ).truncated(
        SimpleEvent.from_data({x: reals(), y: closed(lower, upper)}).as_composite_set()
    )
    assert truncation.distribution is not None
    assert np.log(truncation.probability) == pytest.approx(
        np.log(norm.sf(lower, mean, deviation) - norm.sf(upper, mean, deviation))
    )


def assert_truncation_agrees_with_rustworkx(
    rx_circuit: RxCircuit, event: Event, tolerance: float
) -> None:
    """
    :param rx_circuit: A circuit of units.
    :param event: The event to truncate it and its layered conversion to.
    :param tolerance: The absolute error allowed for the probability of the event.
    """
    rx_truncation = rx_circuit.truncated(event.__deepcopy__())
    truncation = RustworkxCircuitToLayeredCircuitConverter.convert(
        rx_circuit
    ).truncated(event.__deepcopy__())
    assert truncation.probability == pytest.approx(
        rx_truncation.probability, abs=tolerance
    )
    samples = rx_truncation.distribution.sample(200)
    np.testing.assert_allclose(
        truncation.distribution.log_likelihood(samples),
        rx_truncation.distribution.log_likelihood(samples),
        atol=1e-3,
    )


def test_truncation_to_several_intervals_of_one_variable(
    gaussian_mixture, numerical_integration_tolerance, x, y
):
    event = SimpleEvent.from_data(
        {x: closed(-2.0, 0.0) | closed(1.0, 3.0), y: closed(-1.0, 2.0)}
    ).as_composite_set()
    assert_truncation_agrees_with_rustworkx(
        gaussian_mixture, event, numerical_integration_tolerance
    )


def test_truncation_to_several_boxes_in_one_batch(
    gaussian_mixture_times_uniform, numerical_integration_tolerance, x, y, z
):
    event = (
        SimpleEvent.from_data(
            {x: closed(-1.0, 0.0), y: closed(-1.0, 0.0), z: closed(0.0, 1.0)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: closed(1.0, 2.0), y: closed(1.0, 2.0), z: closed(0.5, 2.5)}
        ).as_composite_set()
    )
    assert RustworkxCircuitToLayeredCircuitConverter.convert(
        gaussian_mixture_times_uniform
    ).can_truncate_in_one_batch(list(event.__deepcopy__().simple_sets))
    assert_truncation_agrees_with_rustworkx(
        gaussian_mixture_times_uniform, event, numerical_integration_tolerance
    )


def test_truncating_twice_intersects_the_boxes(
    gaussian_mixture, box_of_x_and_y, numerical_integration_tolerance, x, y
):
    second = SimpleEvent.from_data(
        {x: closed(0.0, 3.0), y: closed(-1.0, 1.0)}
    ).as_composite_set()

    rx_truncated = gaussian_mixture.truncated(
        box_of_x_and_y.__deepcopy__()
    ).distribution
    rx_truncation = rx_truncated.truncated(second.__deepcopy__())
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y.__deepcopy__())
        .distribution
    )
    truncation = truncated.truncated(second.__deepcopy__())

    assert truncation.probability == pytest.approx(
        rx_truncation.probability, abs=numerical_integration_tolerance
    )
    assert truncation.distribution.support == rx_truncation.distribution.support


def test_queries_of_a_truncated_circuit(
    gaussian_mixture, box_of_x_and_y, numerical_integration_tolerance, x, y
):
    rx_truncated = gaussian_mixture.truncated(
        box_of_x_and_y.__deepcopy__()
    ).distribution
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y.__deepcopy__())
        .distribution
    )

    points = np.array([[0.5, 0.5], [1.5, 1.0], [-2.0, 1.0], [3.0, 3.0]])
    np.testing.assert_allclose(
        truncated.cumulative_distribution_function(points),
        rx_truncated.cumulative_distribution_function(points),
        atol=numerical_integration_tolerance,
    )
    inner = SimpleEvent.from_data({x: closed(0.0, 1.0), y: closed(0.5, 3.0)})
    assert truncated.probability_of_simple_event(inner) == pytest.approx(
        rx_truncated.probability_of_simple_event(inner),
        abs=numerical_integration_tolerance,
    )
    assert truncated.support == rx_truncated.support


def test_mode_of_a_truncated_gaussian(single_gaussian, x, y):
    # the box excludes the mean, so the mode lies on its boundary
    event = SimpleEvent.from_data(
        {x: closed(1.0, 2.0), y: closed(-1.0, 0.0)}
    ).as_composite_set()
    rx_truncated = single_gaussian.truncated(event.__deepcopy__()).distribution
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(single_gaussian)
        .truncated(event.__deepcopy__())
        .distribution
    )
    log_mode = truncated.log_mode()
    expected_mode = rx_truncated.log_mode()
    assert log_mode.event == expected_mode.event
    assert log_mode.log_likelihood == pytest.approx(
        expected_mode.log_likelihood, abs=1e-3
    )


def test_samples_of_a_truncated_circuit_lie_in_the_box(
    gaussian_mixture, box_of_x_and_y
):
    rx_truncated = gaussian_mixture.truncated(
        box_of_x_and_y.__deepcopy__()
    ).distribution
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y.__deepcopy__())
        .distribution
    )
    samples = truncated.sample(5000)
    assert samples.shape == (5000, 2)
    assert np.all(np.isfinite(truncated.log_likelihood(samples)))
    np.testing.assert_allclose(
        samples.mean(axis=0), rx_truncated.sample(5000).mean(axis=0), atol=0.05
    )


def test_a_truncated_gaussian_has_no_marginal_over_some_of_its_variables(
    gaussian_mixture, box_of_x_and_y, x
):
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y)
        .distribution
    )
    with pytest.raises(NoClosedFormError):
        truncated.marginal([x])


def test_moments_of_a_truncated_gaussian_agree_with_rustworkx(
    gaussian_mixture, box_of_x_and_y, x, y
):
    rx_truncated = gaussian_mixture.truncated(
        box_of_x_and_y.__deepcopy__()
    ).distribution
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y.__deepcopy__())
        .distribution
    )
    expectation = truncated.expectation([x, y])
    variance = truncated.variance([x, y])
    expected_expectation = rx_truncated.expectation([x, y])
    expected_variance = rx_truncated.variance([x, y])
    for variable in (x, y):
        assert expectation[variable] == pytest.approx(expected_expectation[variable])
        assert variance[variable] == pytest.approx(expected_variance[variable])


def test_a_higher_moment_of_a_truncated_gaussian_agrees_with_rustworkx(
    gaussian_mixture, box_of_x_and_y, x, y
):
    rx_truncated = gaussian_mixture.truncated(
        box_of_x_and_y.__deepcopy__()
    ).distribution
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y.__deepcopy__())
        .distribution
    )
    order = VariableMap({x: 3, y: 4})
    center = VariableMap({x: 0.5, y: 0.0})
    moment = truncated.moment(order, center)
    expected = rx_truncated.moment(order, center)
    for variable in (x, y):
        assert moment[variable] == pytest.approx(expected[variable])


def test_the_expectation_of_a_truncated_gaussian_is_the_mean_of_its_samples(
    gaussian_mixture, box_of_x_and_y, x, y
):
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y)
        .distribution
    )
    expectation = truncated.expectation([x, y])
    np.testing.assert_allclose(
        [expectation[x], expectation[y]],
        truncated.sample(20000).mean(axis=0),
        atol=0.03,
    )


def test_the_way_to_integrate_moments_follows_the_distributions_through_the_layer(
    single_gaussian, box_of_x_and_y, x, y
):
    rx_truncated = single_gaussian.truncated(box_of_x_and_y).distribution
    [truncated_leaf] = rx_truncated.leaves
    moment_integration = MomentIntegration(
        deviations_integrated_over=6.0, panels=8, nodes_per_panel=12
    )
    truncated_leaf.distribution = dataclasses.replace(
        truncated_leaf.distribution, moment_integration=moment_integration
    )
    further = (
        RustworkxCircuitToLayeredCircuitConverter.convert(rx_truncated)
        .truncated(
            SimpleEvent.from_data(
                {x: closed(-0.5, 1.0), y: closed(0.0, 1.5)}
            ).as_composite_set()
        )
        .distribution
    )
    [restored] = LayeredCircuitToRustworkxCircuitConverter.convert(further).leaves
    assert restored.distribution.moment_integration == moment_integration


# %% conditioning


@pytest.mark.parametrize(
    "circuit_fixture",
    [
        "gaussian_mixture",
        "gaussian_mixture_times_uniform",
        "three_variable_gaussian",
    ],
)
def test_conditioning_on_some_of_the_variables(circuit_fixture, request, x):
    rx_circuit = request.getfixturevalue(circuit_fixture)
    point = {x: 0.5}
    rx_conditioning = rx_circuit.log_conditional(point)
    log_conditioning = RustworkxCircuitToLayeredCircuitConverter.convert(
        rx_circuit
    ).log_conditional(point)
    assert log_conditioning.log_likelihood == pytest.approx(
        rx_conditioning.log_likelihood
    )

    free = [variable for variable in rx_circuit.variables if variable != x]
    samples = rx_conditioning.distribution.marginal(free).sample(200)
    np.testing.assert_allclose(
        log_conditioning.distribution.marginal(free).log_likelihood(samples),
        rx_conditioning.distribution.marginal(free).log_likelihood(samples),
    )


def test_conditioning_on_every_variable_is_the_likelihood(gaussian_mixture, x, y):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
    log_conditioning = layered.log_conditional({x: 0.5, y: -0.25})
    assert log_conditioning.log_likelihood == pytest.approx(
        float(layered.log_likelihood(np.array([[0.5, -0.25]]))[0])
    )
    assert log_conditioning.distribution.sample(3).tolist() == [[0.5, -0.25]] * 3


@pytest.mark.parametrize("circuit_fixture", ["single_gaussian", "gaussian_mixture"])
def test_conditioning_a_truncated_circuit_agrees_with_rustworkx(
    circuit_fixture, request, box_of_x_and_y, x, y
):
    rx_circuit = request.getfixturevalue(circuit_fixture)
    point = {x: 0.5}
    values = np.array([[0.1], [0.7], [1.4]])
    rx_truncated = rx_circuit.truncated(box_of_x_and_y.__deepcopy__()).distribution
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(rx_circuit)
        .truncated(box_of_x_and_y.__deepcopy__())
        .distribution
    )

    expected_conditioning = rx_truncated.log_conditional(point)
    log_conditioning = truncated.log_conditional(point)
    assert log_conditioning.log_likelihood == pytest.approx(
        expected_conditioning.log_likelihood, abs=1e-3
    )
    np.testing.assert_allclose(
        log_conditioning.distribution.marginal([y]).log_likelihood(values),
        expected_conditioning.distribution.marginal([y]).log_likelihood(values),
        atol=1e-3,
    )


def test_conditioning_a_truncated_mixture_is_the_density_of_the_marginal(
    gaussian_mixture, box_of_x_and_y, x
):
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y)
        .distribution
    )
    log_probability = truncated.log_conditional({x: 0.5}).log_likelihood

    # the density of x at 0.5 integrates the joint density over the slice of y
    grid = np.linspace(0.0, 1.5, 20001)
    density = np.exp(
        truncated.log_likelihood(np.column_stack([np.full_like(grid, 0.5), grid]))
    )
    assert np.exp(log_probability) == pytest.approx(
        np.trapezoid(density, grid), abs=1e-3
    )


def test_conditioning_outside_the_box_is_impossible(
    gaussian_mixture, box_of_x_and_y, x
):
    truncated = (
        RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
        .truncated(box_of_x_and_y)
        .distribution
    )
    log_conditioning = truncated.log_conditional({x: 5.0})
    assert log_conditioning.distribution is None
    assert log_conditioning.log_likelihood == -np.inf


# %% changing the circuit in place


@pytest.fixture(params=[False, True], ids=["untruncated", "truncated"])
def layered_mixture(
    request, gaussian_mixture, box_of_x_and_y
) -> LayeredProbabilisticCircuit:
    """
    :func:`gaussian_mixture` as layers, as it is and truncated to
    :func:`box_of_x_and_y`.
    """
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
    if not request.param:
        return layered
    truncated = layered.truncated(box_of_x_and_y).distribution
    return truncated


def test_translation_moves_the_density(layered_mixture, x, y):
    samples = layered_mixture.sample(50)
    expected = layered_mixture.log_likelihood(samples)
    layered_mixture.apply_translation({x: 1.0, y: -2.0})
    np.testing.assert_allclose(
        layered_mixture.log_likelihood(samples + np.array([1.0, -2.0])), expected
    )


def test_scaling_scales_the_density(layered_mixture, x, y):
    samples = layered_mixture.sample(50)
    expected = layered_mixture.log_likelihood(samples) - np.log(2.0 * 3.0)
    layered_mixture.apply_scaling({x: 2.0, y: 3.0})
    np.testing.assert_allclose(
        layered_mixture.log_likelihood(samples * np.array([2.0, 3.0])), expected
    )


def test_renaming_that_reorders_the_variables(gaussian_mixture, x, y):
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(gaussian_mixture)
    samples = gaussian_mixture.sample(50)
    expected = layered.log_likelihood(samples)

    # a name that sorts after y moves x to the last column
    renamed_x = Continuous("z_x")
    layered.update_variables(VariableMap({x: renamed_x}))
    assert list(layered.variables) == [y, renamed_x]
    np.testing.assert_allclose(layered.log_likelihood(samples[:, ::-1]), expected)


# %% serialization


def test_json_round_trip(layered_mixture):
    points = np.array([[0.5, 0.5], [1.5, 1.0]])
    restored = from_json(to_json(layered_mixture.root))
    assert isinstance(restored, SumLayer)
    np.testing.assert_allclose(
        restored.log_likelihood_of_nodes(points)[:, 0],
        layered_mixture.log_likelihood(points),
    )
