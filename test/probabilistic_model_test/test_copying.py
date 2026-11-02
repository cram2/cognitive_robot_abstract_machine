from __future__ import annotations

import copy
from enum import StrEnum

import numpy as np
import pytest
from random_events.interval import SimpleInterval
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel

# %% fixtures


@pytest.fixture
def horizontal() -> Continuous:
    return Continuous("horizontal")


@pytest.fixture
def vertical() -> Continuous:
    return Continuous("vertical")


@pytest.fixture
def color() -> Symbolic:
    return Symbolic("color", domain=Set.from_iterable(["red", "green"]))


@pytest.fixture
def circuit(horizontal: Continuous, vertical: Continuous) -> ProbabilisticCircuit:
    """
    A mixture of two products whose leaves over the same variable share it.
    """
    result = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=result)
    for lower in (0.0, 2.0):
        product = ProductUnit(probabilistic_circuit=result)
        product.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=horizontal,
                    interval=SimpleInterval.from_data(lower, lower + 1),
                ),
                result,
            )
        )
        product.add_subcircuit(
            leaf(
                GaussianDistribution(variable=vertical, location=lower, scale=1.0),
                result,
            )
        )
        root.add_subcircuit(product, np.log(0.5))
    return result


class ModelKind(StrEnum):
    """
    The kinds of models every copy test runs on.
    """

    GAUSSIAN = "gaussian"
    TRUNCATED_GAUSSIAN = "truncated_gaussian"
    UNIFORM = "uniform"
    DIRAC_DELTA = "dirac_delta"
    MULTINOMIAL = "multinomial"
    MULTIVARIATE_GAUSSIAN = "multivariate_gaussian"
    RX_CIRCUIT = "rx_circuit"
    LAYERED_CIRCUIT = "layered_circuit"


@pytest.fixture(params=list(ModelKind))
def model(
    request: pytest.FixtureRequest,
    horizontal: Continuous,
    vertical: Continuous,
    color: Symbolic,
    circuit: ProbabilisticCircuit,
) -> ProbabilisticModel:
    if request.param == ModelKind.GAUSSIAN:
        return GaussianDistribution(variable=horizontal, location=0.0, scale=1.0)
    if request.param == ModelKind.TRUNCATED_GAUSSIAN:
        return TruncatedGaussianDistribution(
            variable=horizontal,
            interval=SimpleInterval.from_data(-1, 1),
            location=0.0,
            scale=1.0,
        )
    if request.param == ModelKind.UNIFORM:
        return UniformDistribution(
            variable=horizontal, interval=SimpleInterval.from_data(0, 1)
        )
    if request.param == ModelKind.DIRAC_DELTA:
        return DiracDeltaDistribution(
            variable=horizontal, location=0.0, density_cap=1.0
        )
    if request.param == ModelKind.MULTINOMIAL:
        return MultinomialDistribution((color,), np.array([0.25, 0.75]))
    if request.param == ModelKind.MULTIVARIATE_GAUSSIAN:
        return MultivariateGaussianDistribution(
            variables=(horizontal, vertical),
            mean=np.array([0.0, 1.0]),
            covariance=Covariance.from_matrix(np.array([[1.0, 0.5], [0.5, 2.0]])),
        )
    if request.param == ModelKind.RX_CIRCUIT:
        return circuit
    return RustworkxCircuitToLayeredCircuitConverter.convert(circuit)


# %% deep copies


def test_a_deep_copy_of_a_model_is_a_new_model(model: ProbabilisticModel):
    copied = copy.deepcopy(model)
    assert copied is not model
    assert type(copied) is type(model)


def test_a_deep_copy_of_a_model_shares_its_variables(model: ProbabilisticModel):
    copied = copy.deepcopy(model)
    assert all(
        copied_variable is original_variable
        for copied_variable, original_variable in zip(
            copied.variables, model.variables, strict=True
        )
    )


def test_a_deep_copy_of_a_circuit_shares_variables_across_its_leaves(
    circuit: ProbabilisticCircuit, horizontal: Continuous
):
    copied = copy.deepcopy(circuit)
    leaf_variables = [
        variable for leaf_unit in copied.leaves for variable in leaf_unit.variables
    ]
    assert [variable for variable in leaf_variables if variable == horizontal] == [
        horizontal,
        horizontal,
    ]
    assert all(
        variable is horizontal for variable in leaf_variables if variable == horizontal
    )


@pytest.mark.parametrize(
    "model",
    [kind for kind in ModelKind if kind != ModelKind.MULTINOMIAL],
    indirect=True,
)
def test_a_deep_copy_of_a_model_does_not_move_with_the_original(
    model: ProbabilisticModel,
):
    copied = copy.deepcopy(model)
    samples = copied.sample(10)
    log_likelihood_before = copied.log_likelihood(samples)
    model.apply_translation(
        {variable: 100.0 for variable in model.variables if variable.is_numeric}
    )
    np.testing.assert_array_equal(copied.log_likelihood(samples), log_likelihood_before)
