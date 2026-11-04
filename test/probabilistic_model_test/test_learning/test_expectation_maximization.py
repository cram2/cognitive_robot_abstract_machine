"""
Tests for expectation maximization on the parameters of an existing circuit, and for
fitting circuits on dataframes with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
from random_events.interval import closed
from random_events.variable import Continuous
from sklearn.mixture import GaussianMixture

from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.learning.exceptions import (
    StartingCircuitVariablesMismatchError,
    UnsupportedLeafDistributionError,
)
from probabilistic_model.learning.expectation_maximization import (
    ExpectationMaximization,
    ExpectationMaximizationLearning,
    LearnableUnits,
)
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.helper import (
    factorized_gaussian_mixture,
    fully_factorized,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
    SumUnit,
    leaf,
)


# %% fixtures
@pytest.fixture
def iris() -> pd.DataFrame:
    """
    The four continuous measurements of the scikit-learn iris dataset.
    """
    return sklearn.datasets.load_iris(as_frame=True).data


@pytest.fixture
def iris_starting_circuit(iris) -> ProbabilisticCircuit:
    """
    An equally weighted mixture of three factorized standard Gaussians over the iris
    measurements, centered at one row of each species.
    """
    means = _iris_starting_means(iris)
    return factorized_gaussian_mixture(
        [Continuous(name) for name in iris.columns],
        means=means,
        scales=np.ones_like(means),
    )


# %% helpers
def _iris_starting_means(iris: pd.DataFrame) -> np.ndarray:
    """
    The first row of each iris species, one row per mixture component.
    """
    return iris.to_numpy()[[0, 50, 100]]


def _make_gaussian_leaf_circuit() -> ProbabilisticCircuit:
    """
    A circuit that is a single standard Gaussian leaf over ``x``.
    """
    circuit = ProbabilisticCircuit()
    leaf(
        GaussianDistribution(variable=Continuous("x"), location=0.0, scale=1.0), circuit
    )
    return circuit


def _make_mixture_circuit() -> ProbabilisticCircuit:
    """
    A uniform mixture of two Gaussian leaves over ``x``, centered at -1 and 1.
    """
    circuit = ProbabilisticCircuit()
    variable = Continuous("x")
    sum_root = SumUnit(probabilistic_circuit=circuit)
    for location in (-1.0, 1.0):
        component = leaf(
            GaussianDistribution(variable=variable, location=location, scale=1.0),
            circuit,
        )
        sum_root.add_subcircuit(component, log_weight=0.0)
    sum_root.normalize()
    return circuit


def _leaf_of(circuit: ProbabilisticCircuit, variable_name: str) -> LeafUnit:
    """
    The leaf of a circuit over the variable with the given name.
    """
    [result] = [
        unit
        for unit in circuit.leaves
        if unit.distribution.variable.name == variable_name
    ]
    return result


def _learn_all_units(
    circuit: ProbabilisticCircuit, data: np.ndarray, iterations: int = 1
) -> list[float]:
    """
    Learn every unit of a circuit with default expectation maximization.
    """
    return ExpectationMaximization().learn(
        circuit, data, LearnableUnits.from_units(circuit.nodes()), iterations
    )


# %% expectation maximization on an existing circuit
def test_learn_fits_gaussian_leaf_to_data():
    circuit = _make_gaussian_leaf_circuit()
    data = np.random.default_rng(42).normal(loc=5.0, scale=0.5, size=(100, 1))

    _learn_all_units(circuit, data)

    [gaussian_leaf] = circuit.leaves
    assert gaussian_leaf.distribution.location == pytest.approx(np.mean(data))
    assert gaussian_leaf.distribution.scale == pytest.approx(np.std(data))


def test_learn_updates_only_learnable_units():
    circuit = fully_factorized([Continuous("x"), Continuous("y")])
    learned_leaf = _leaf_of(circuit, "x")
    frozen_leaf = _leaf_of(circuit, "y")
    frozen_location = frozen_leaf.distribution.location
    frozen_scale = frozen_leaf.distribution.scale
    data = np.random.default_rng(42).normal(loc=5.0, scale=0.5, size=(100, 2))

    ExpectationMaximization().learn(
        circuit, data, LearnableUnits.from_units([learned_leaf])
    )

    x_values = data[
        :, circuit.variable_to_index_map[learned_leaf.distribution.variable]
    ]
    assert learned_leaf.distribution.location == pytest.approx(np.mean(x_values))
    assert frozen_leaf.distribution.location == frozen_location
    assert frozen_leaf.distribution.scale == frozen_scale


def test_learn_increases_likelihood_of_mixture():
    circuit = _make_mixture_circuit()
    generator = np.random.default_rng(42)
    data = np.concatenate(
        [
            generator.normal(loc=-3.0, scale=0.5, size=(50, 1)),
            generator.normal(loc=3.0, scale=0.5, size=(150, 1)),
        ]
    )

    history = _learn_all_units(circuit, data, iterations=5)

    assert len(history) == 5
    assert history[-1] > history[0]


def test_learn_never_decreases_likelihood_on_iris(iris, iris_starting_circuit):
    data = iris[[variable.name for variable in iris_starting_circuit.variables]]

    history = _learn_all_units(iris_starting_circuit, data.to_numpy(), iterations=10)

    assert np.all(np.diff(history) >= 0.0)


def test_learn_returns_empty_history_for_empty_data():
    history = _learn_all_units(_make_gaussian_leaf_circuit(), np.empty((0, 1)))

    assert history == []


def test_learnable_units_reject_truncated_gaussian_leaf():
    truncated_gaussian = TruncatedGaussianDistribution(
        variable=Continuous("x"),
        interval=closed(0.0, np.inf).simple_sets[0],
        location=0.0,
        scale=1.0,
    )
    circuit = ProbabilisticCircuit()
    leaf(truncated_gaussian, circuit)

    with pytest.raises(UnsupportedLeafDistributionError) as error:
        LearnableUnits.from_units(circuit.nodes())

    assert error.value.distribution == truncated_gaussian


# %% fitting a circuit on a dataframe
def test_fit_returns_starting_circuit_with_learned_parameters():
    data = pd.DataFrame(
        {"x": np.random.default_rng(42).normal(loc=5.0, scale=0.5, size=100)}
    )

    circuit = ExpectationMaximizationLearning(
        _make_gaussian_leaf_circuit(), iterations=1
    ).fit(data)

    [gaussian_leaf] = circuit.leaves
    assert gaussian_leaf.distribution.location == pytest.approx(data["x"].mean())
    assert gaussian_leaf.distribution.scale == pytest.approx(data["x"].std(ddof=0))


def test_fit_leaves_starting_circuit_unchanged():
    starting_circuit = _make_gaussian_leaf_circuit()
    [starting_leaf] = starting_circuit.leaves
    starting_location = starting_leaf.distribution.location
    starting_scale = starting_leaf.distribution.scale
    data = pd.DataFrame({"x": np.random.default_rng(42).normal(loc=5.0, size=100)})

    ExpectationMaximizationLearning(starting_circuit, iterations=1).fit(data)

    assert starting_leaf.distribution.location == starting_location
    assert starting_leaf.distribution.scale == starting_scale


def test_fit_reads_columns_by_variable_name():
    generator = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "y": generator.normal(loc=-5.0, size=100),
            "x": generator.normal(loc=5.0, size=100),
        }
    )

    circuit = ExpectationMaximizationLearning(
        fully_factorized([Continuous("x"), Continuous("y")]), iterations=1
    ).fit(data)

    assert _leaf_of(circuit, "x").distribution.location == pytest.approx(
        data["x"].mean()
    )
    assert _leaf_of(circuit, "y").distribution.location == pytest.approx(
        data["y"].mean()
    )


def test_fit_rejects_variables_other_than_those_of_starting_circuit():
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [3.0, 4.0, 5.0]})

    with pytest.raises(StartingCircuitVariablesMismatchError):
        ExpectationMaximizationLearning(
            _make_gaussian_leaf_circuit(), iterations=1
        ).fit(data, infer_variables_from_dataframe(data))


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_matches_diagonal_gaussian_mixture_of_scikit_learn(
    iris, iris_starting_circuit
):
    iterations = 10
    means = _iris_starting_means(iris)
    component_count = len(means)
    scikit_learn_mixture = GaussianMixture(
        n_components=component_count,
        covariance_type="diag",
        weights_init=np.full(component_count, 1 / component_count),
        means_init=means,
        precisions_init=np.ones_like(means),
        max_iter=iterations,
        tol=0.0,
        reg_covar=0.0,
    ).fit(iris.to_numpy())

    circuit = ExpectationMaximizationLearning(
        iris_starting_circuit, iterations=iterations
    ).fit(iris)

    data = iris[[variable.name for variable in circuit.variables]]
    assert circuit.log_likelihood(data.to_numpy()).mean() == pytest.approx(
        scikit_learn_mixture.score(iris.to_numpy())
    )
