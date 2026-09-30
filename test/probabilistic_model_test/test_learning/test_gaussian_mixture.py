"""
Tests for fitting circuits as Gaussian mixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from sklearn.mixture import GaussianMixture

from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from probabilistic_model.distributions.distributions import (
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.exceptions import NonContinuousVariableError
from probabilistic_model.learning.gaussian_mixture.covariance_type import (
    CovarianceType,
)
from probabilistic_model.learning.gaussian_mixture.gaussian_mixture_model import (
    GaussianMixtureModel,
)
from probabilistic_model.learning.gaussian_mixture.step_mix_model import StepMixModel
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    MultivariateLeaf,
    SumUnit,
)


@pytest.fixture
def two_clusters() -> pd.DataFrame:
    """
    Three hundred rows in two correlated clusters. The columns are named so that the
    circuit, which sorts its variables by name, lays them out in the reverse order of
    the dataframe; a conversion that confused the two orders would swap the columns.
    """
    generator = np.random.default_rng(0)
    first = generator.multivariate_normal(
        [0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=200
    )
    second = generator.multivariate_normal(
        [6.0, -4.0], [[0.5, -0.2], [-0.2, 2.0]], size=100
    )
    samples = np.concatenate([first, second])
    return pd.DataFrame({"y": samples[:, 0], "x": samples[:, 1]})


def _circuit_columns(circuit, data: pd.DataFrame) -> np.ndarray:
    """
    :return: The rows of the dataframe, with columns in the circuit's variable order.
    """
    return data[[variable.name for variable in circuit.variables]].to_numpy()


@pytest.mark.parametrize("covariance_type", CovarianceType)
def test_the_circuit_has_the_density_of_the_fitted_mixture(
    two_clusters, covariance_type
):
    method = GaussianMixtureModel(
        GaussianMixture(n_components=2, covariance_type=covariance_type, random_state=0)
    )

    circuit = method.fit(two_clusters)

    assert circuit.log_likelihood(
        _circuit_columns(circuit, two_clusters)
    ) == pytest.approx(method.model.score_samples(two_clusters.to_numpy()))


def test_the_root_weighs_one_component_per_mixture_component(two_clusters):
    method = GaussianMixtureModel(GaussianMixture(n_components=3, random_state=0))

    circuit = method.fit(two_clusters)

    assert isinstance(circuit.root, SumUnit)
    assert len(circuit.root.subcircuits) == 3
    assert np.sort(np.exp(circuit.root.log_weights)) == pytest.approx(
        np.sort(method.model.weights_)
    )


@pytest.mark.parametrize("covariance_type", CovarianceType)
def test_every_component_is_one_multivariate_gaussian_leaf(
    two_clusters, covariance_type
):
    circuit = GaussianMixtureModel(
        GaussianMixture(n_components=2, covariance_type=covariance_type, random_state=0)
    ).fit(two_clusters)

    for component in circuit.root.subcircuits:
        assert isinstance(component, MultivariateLeaf)
        assert isinstance(component.distribution, MultivariateGaussianDistribution)


def test_a_single_variable_is_fitted_into_one_dimensional_gaussians(two_clusters):
    circuit = GaussianMixtureModel(GaussianMixture(n_components=2, random_state=0)).fit(
        two_clusters[["y"]]
    )

    for component in circuit.root.subcircuits:
        assert isinstance(component.distribution, MultivariateGaussianDistribution)
        assert [variable.name for variable in component.distribution.variables] == ["y"]


def test_a_single_component_is_simplified_into_its_leaf(two_clusters):
    circuit = GaussianMixtureModel(GaussianMixture(n_components=1)).fit(two_clusters)

    assert isinstance(circuit.root, MultivariateLeaf)


def test_the_circuit_puts_all_probability_on_the_reals(two_clusters):
    circuit = GaussianMixtureModel(GaussianMixture(n_components=2, random_state=0)).fit(
        two_clusters
    )

    assert circuit.probability(
        circuit.universal_simple_event().as_composite_set()
    ) == pytest.approx(1.0)


def test_a_box_is_as_probable_as_the_fraction_of_samples_in_it(two_clusters):
    circuit = GaussianMixtureModel(GaussianMixture(n_components=2, random_state=0)).fit(
        two_clusters
    )
    x, y = circuit.variables
    event = SimpleEvent.from_data(
        {x: closed(-2.0, 2.0), y: closed(-2.0, 2.0)}
    ).as_composite_set()

    samples = circuit.sample(20_000)
    inside = np.all((samples >= -2.0) & (samples <= 2.0), axis=1).mean()

    assert circuit.probability(event) == pytest.approx(inside, abs=0.02)


def test_only_the_given_variables_are_fitted(two_clusters):
    variables = [
        annotated_variable
        for annotated_variable in infer_variables_from_dataframe(two_clusters)
        if annotated_variable.variable.name == "x"
    ]

    circuit = GaussianMixtureModel(GaussianMixture(n_components=2, random_state=0)).fit(
        two_clusters, variables
    )

    assert [variable.name for variable in circuit.variables] == ["x"]


def test_a_mixture_fitted_outside_is_converted(two_clusters):
    mixture = GaussianMixture(n_components=2, random_state=0).fit(
        two_clusters.to_numpy()
    )
    variables = [Continuous("y"), Continuous("x")]

    circuit = GaussianMixtureModel(mixture).to_probabilistic_circuit(variables)

    assert circuit.log_likelihood(
        _circuit_columns(circuit, two_clusters)
    ) == pytest.approx(mixture.score_samples(two_clusters.to_numpy()))


def test_each_fit_is_a_circuit_of_its_own(two_clusters):
    method = GaussianMixtureModel(GaussianMixture(n_components=2, random_state=0))

    first = method.fit(two_clusters)
    second = method.fit(two_clusters)

    assert first is not second
    assert len(first.nodes()) == len(second.nodes())


@pytest.mark.parametrize(
    "method",
    [
        GaussianMixtureModel(GaussianMixture(n_components=2, random_state=0)),
        StepMixModel(),
    ],
    ids=["scikit-learn", "StepMix"],
)
def test_refitting_leaves_an_earlier_circuit_unchanged(two_clusters, method):
    first = method.fit(two_clusters)
    before = first.log_likelihood(_circuit_columns(first, two_clusters))

    method.fit(two_clusters.iloc[:200] * 3.0)

    assert first.log_likelihood(_circuit_columns(first, two_clusters)) == pytest.approx(
        before
    )


@pytest.fixture
def labelled_clusters(two_clusters) -> pd.DataFrame:
    """
    The two clusters with a symbolic column naming each row's cluster and an integer
    column that is 1 in the first cluster and 2 or 3 in the second.
    """
    return two_clusters.assign(
        cluster=["first"] * 200 + ["second"] * 100,
        count=[1] * 200 + [2, 3] * 50,
    )


def _two_component_stepmix() -> StepMixModel:
    method = StepMixModel()
    method.model.set_params(n_components=2, random_state=0)
    return method


def _two_component_fit(data: pd.DataFrame):
    return _two_component_stepmix().fit(data)


def _variable(circuit, name: str):
    return next(variable for variable in circuit.variables if variable.name == name)


def test_every_component_gets_a_leaf_per_discrete_variable(labelled_clusters):
    circuit = _two_component_fit(labelled_clusters)

    for component in circuit.root.subcircuits:
        distributions = [child.distribution for child in component.subcircuits]
        assert sum(isinstance(d, SymbolicDistribution) for d in distributions) == 1
        assert sum(isinstance(d, IntegerDistribution) for d in distributions) == 1
        for distribution in distributions:
            if isinstance(distribution, (SymbolicDistribution, IntegerDistribution)):
                assert math.fsum(distribution.probabilities.values()) == pytest.approx(
                    1.0
                )


def test_scikit_learn_rejects_discrete_variables(labelled_clusters):
    method = GaussianMixtureModel(GaussianMixture(n_components=2, random_state=0))

    with pytest.raises(NonContinuousVariableError):
        method.fit(labelled_clusters)


def test_stepmix_fits_continuous_data_with_the_density_of_its_mixture(two_clusters):
    method = _two_component_stepmix()

    circuit = method.fit(two_clusters)

    assert circuit.log_likelihood(
        _circuit_columns(circuit, two_clusters)
    ).mean() == pytest.approx(method.model.score(two_clusters.to_numpy()))


def test_the_circuit_has_the_density_of_the_fitted_mixed_mixture(labelled_clusters):
    method = _two_component_stepmix()
    circuit = method.fit(labelled_clusters)
    cluster = _variable(circuit, "cluster")
    elements = list(cluster.domain.all_elements)
    stepmix_data = np.column_stack(
        [
            labelled_clusters[["y", "x"]].to_numpy(),
            [elements.index(value) for value in labelled_clusters["cluster"]],
            np.unique(labelled_clusters["count"], return_inverse=True)[1],
        ]
    )

    log_likelihood = circuit.log_likelihood(
        _circuit_columns(circuit, labelled_clusters)
    )

    assert log_likelihood.mean() == pytest.approx(method.model.score(stepmix_data))


def test_a_symbolic_variable_is_as_frequent_as_in_the_data(labelled_clusters):
    circuit = _two_component_fit(labelled_clusters)
    cluster = _variable(circuit, "cluster")

    first = SimpleEvent.from_data({cluster: "first"}).as_composite_set()

    assert circuit.probability(first) == pytest.approx(200 / 300, abs=1e-3)


def test_the_label_follows_its_cluster(labelled_clusters):
    circuit = _two_component_fit(labelled_clusters)
    x, y, cluster = (_variable(circuit, name) for name in ("x", "y", "cluster"))
    around_second = {x: closed(-7.0, -1.0), y: closed(4.0, 8.0)}

    labelled_second = SimpleEvent.from_data(
        {**around_second, cluster: "second"}
    ).as_composite_set()
    anywhere = SimpleEvent.from_data(around_second).as_composite_set()

    assert circuit.probability(labelled_second) / circuit.probability(
        anywhere
    ) == pytest.approx(1.0, abs=1e-3)


def test_stratifying_on_a_symbolic_variable_keeps_each_value_in_one_branch(
    labelled_clusters,
):
    circuit = StratifiedLearning(
        variables=["cluster"],
        method=_two_component_stepmix(),
    ).fit(labelled_clusters)
    cluster = _variable(circuit, "cluster")

    for value in ("first", "second"):
        event = SimpleEvent.from_data({cluster: value}).as_composite_set()
        branches_with_value = []
        for branch in circuit.root.subcircuits:
            branch_circuit = type(circuit)()
            branch_circuit.mount(branch)
            if branch_circuit.probability(event) > 0:
                branches_with_value.append(branch)
        assert len(branches_with_value) == 1


def test_data_without_a_continuous_column_is_a_mixture_of_categoricals(
    labelled_clusters,
):
    circuit = _two_component_fit(labelled_clusters[["cluster", "count"]])
    cluster, count = _variable(circuit, "cluster"), _variable(circuit, "count")

    first_with_one = SimpleEvent.from_data(
        {cluster: "first", count: 1}
    ).as_composite_set()

    assert {variable.name for variable in circuit.variables} == {"cluster", "count"}
    assert circuit.probability(first_with_one) == pytest.approx(200 / 300, abs=1e-3)


def test_the_weights_sum_to_one(two_clusters):
    circuit = GaussianMixtureModel(GaussianMixture(n_components=4, random_state=0)).fit(
        two_clusters
    )

    assert math.fsum(np.exp(circuit.root.log_weights)) == pytest.approx(1.0)
