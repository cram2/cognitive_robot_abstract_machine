import copy

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    MultivariateLeaf,
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)

# %% shared fixtures


@pytest.fixture
def first() -> Continuous:
    return Continuous("a")


@pytest.fixture
def second() -> Continuous:
    return Continuous("b")


@pytest.fixture
def other() -> Continuous:
    return Continuous("c")


@pytest.fixture
def correlated(first, second) -> MultivariateGaussianDistribution:
    """
    A Gaussian whose variables are laid out in the reverse of the order a circuit sorts
    them in, so a leaf that confused the two orders would read the wrong columns.
    """
    return MultivariateGaussianDistribution(
        variables=(second, first),
        mean=np.array([10.0, -10.0]),
        covariance=Covariance.from_matrix(np.array([[1.0, 0.6], [0.6, 2.0]])),
    )


@pytest.fixture
def independent_of_it(other) -> GaussianDistribution:
    return GaussianDistribution(variable=other, location=3.0, scale=0.5)


@pytest.fixture
def circuit(correlated, independent_of_it) -> ProbabilisticCircuit:
    """
    The product of the multivariate Gaussian and a univariate Gaussian over a third
    variable.
    """
    circuit = ProbabilisticCircuit()
    product = ProductUnit(probabilistic_circuit=circuit)
    product.add_subcircuit(leaf(correlated, circuit))
    product.add_subcircuit(leaf(independent_of_it, circuit))
    return circuit


def columns_of(
    circuit: ProbabilisticCircuit, points: np.ndarray, *variables: Continuous
) -> np.ndarray:
    """
    :return: The columns of ``points``, laid out by the circuit's variables, that hold
        the given variables.
    """
    return points[:, [circuit.variables.index(variable) for variable in variables]]


# %% building a leaf


class TestBuildingAMultivariateLeaf:
    def test_the_factory_builds_a_multivariate_leaf(self, correlated):
        assert isinstance(leaf(correlated, ProbabilisticCircuit()), MultivariateLeaf)

    def test_a_circuit_is_over_every_variable_of_its_leaves(
        self, circuit, first, second, other
    ):
        assert set(circuit.variables) == {first, second, other}

    def test_a_circuit_holding_it_can_be_copied(self, circuit):
        copied = copy.deepcopy(circuit)
        assert copied.root is not circuit.root
        points = np.array([[0.0, 1.0, 2.0]])
        assert copied.log_likelihood(points) == pytest.approx(
            circuit.log_likelihood(points)
        )


# %% queries that read the leaf's columns


class TestReadingTheColumnsOfTheLeaf:
    def test_the_likelihood_reads_the_leaf_s_variables_in_its_own_order(
        self, circuit, correlated, independent_of_it, first, second, other
    ):
        points = np.array([[-9.0, 11.0, 3.2], [-10.5, 9.0, 2.5]])
        expected = correlated.log_likelihood(
            columns_of(circuit, points, second, first)
        ) + independent_of_it.log_likelihood(columns_of(circuit, points, other))
        assert circuit.log_likelihood(points) == pytest.approx(expected)

    def test_samples_land_in_the_columns_of_their_own_variables(
        self, circuit, correlated, first, second
    ):
        np.random.seed(69)
        samples = circuit.sample(500)
        assert columns_of(circuit, samples, first, second).mean(
            axis=0
        ) == pytest.approx(
            [
                correlated.mean[correlated.index_of(variable)]
                for variable in (first, second)
            ],
            abs=0.3,
        )


# %% fixing variables of the leaf


class TestConditioningTheLeaf:
    def test_a_circuit_of_only_the_leaf_answers_what_the_distribution_cannot(
        self, correlated, first, second
    ):
        """
        Conditioning the distribution on all of its variables needs a circuit, so
        wrapping it into one is what makes the question answerable.
        """
        circuit = ProbabilisticCircuit()
        leaf(correlated, circuit)
        point = {first: -9.5, second: 10.5}
        conditional, log_density = circuit.log_conditional(point)
        assert log_density == pytest.approx(
            correlated.log_likelihood(np.array([[point[second], point[first]]]))[0]
        )
        assert {
            unit.distribution.variable: unit.distribution.location
            for unit in conditional.leaves
        } == point

    def test_fixing_every_variable_of_the_leaf_leaves_a_dirac_on_each(
        self, circuit, first, second, other
    ):
        conditional, _ = circuit.log_conditional({first: -9.5, second: 10.5})
        dirac_variables = {
            unit.distribution.variable
            for unit in conditional.leaves
            if isinstance(unit.distribution, DiracDeltaDistribution)
        }
        assert dirac_variables == {first, second}
        assert set(conditional.variables) == {first, second, other}

    def test_fixing_every_variable_keeps_the_rest_of_the_circuit(
        self, circuit, independent_of_it, first, second, other
    ):
        conditional, _ = circuit.log_conditional({first: -9.5, second: 10.5})
        marginal = conditional.marginal([other])
        points = np.array([[2.0], [3.5]])
        assert marginal.log_likelihood(points) == pytest.approx(
            independent_of_it.log_likelihood(points)
        )

    def test_fixing_every_variable_while_keeping_the_structure_leaves_diracs(
        self, circuit, correlated, first, second
    ):
        point = {first: -9.5, second: 10.5}
        conditional, log_density = copy.deepcopy(circuit).log_conditional_in_place(
            point, preserve_structure=True
        )
        assert log_density == pytest.approx(
            correlated.log_likelihood(np.array([[point[second], point[first]]]))[0]
        )
        assert not any(
            isinstance(unit, MultivariateLeaf) for unit in conditional.leaves
        )

    def test_fixing_some_variables_of_the_leaf_leaves_the_gaussian_conditional(
        self, circuit, correlated, first, second
    ):
        conditional, log_density = circuit.log_conditional({first: -9.5})
        expected, expected_log_density = correlated.log_conditional({first: -9.5})
        assert log_density == pytest.approx(expected_log_density)
        remaining = conditional.marginal([second])
        points = np.array([[9.0], [10.0]])
        assert remaining.log_likelihood(points) == pytest.approx(
            expected.log_likelihood(points)
        )

    def test_fixing_only_other_variables_leaves_the_leaf_alone(
        self, circuit, correlated, first, second, other
    ):
        conditional, _ = circuit.log_conditional({other: 3.0})
        remaining = conditional.marginal([first, second])
        points = np.array([[-10.0, 10.0]])
        assert remaining.log_likelihood(points) == pytest.approx(
            correlated.log_likelihood(np.array([[10.0, -10.0]]))
        )


# %% confining the leaf to an event


class TestTruncatingTheLeaf:
    def test_an_event_of_several_boxes_is_the_mixture_of_the_boxes(
        self, circuit, correlated, first, second, other
    ):
        event = SimpleEvent.from_data(
            {
                first: closed(-12.0, -10.0) | closed(-9.0, -8.0),
                second: closed(9.0, 11.0),
                other: closed(-np.inf, np.inf),
            }
        ).as_composite_set()
        truncated, log_probability = circuit.log_truncated(event)
        assert log_probability == pytest.approx(
            np.log(correlated.probability(event.marginal([first, second])))
        )
        assert truncated.probability(truncated.support) == pytest.approx(1.0)
        assert truncated.probability(
            SimpleEvent.from_data(
                {
                    first: closed(-9.9, -9.1),
                    second: closed(9.0, 11.0),
                    other: closed(-np.inf, np.inf),
                }
            ).as_composite_set()
        ) == pytest.approx(0.0)

    def test_an_event_of_one_box_confines_the_leaf(
        self, circuit, correlated, first, second, other
    ):
        event = SimpleEvent.from_data(
            {
                first: closed(-11.0, -9.0),
                second: closed(9.0, 11.0),
                other: closed(-np.inf, np.inf),
            }
        ).as_composite_set()
        truncated, log_probability = circuit.log_truncated(event)
        assert log_probability == pytest.approx(
            np.log(correlated.probability(event.marginal([first, second])))
        )
        assert truncated.probability(truncated.support) == pytest.approx(1.0)


# %% dropping and moving the leaf's variables


class TestMarginalAndTranslation:
    def test_a_marginal_without_the_leaf_s_variables_drops_the_leaf(
        self, circuit, other
    ):
        marginal = circuit.marginal([other])
        assert list(marginal.variables) == [other]

    def test_translating_the_circuit_moves_the_leaf(
        self, circuit, correlated, first, second, other
    ):
        circuit.apply_translation({first: 1.0, second: 2.0, other: 0.0})
        [moved] = [
            unit for unit in circuit.leaves if isinstance(unit, MultivariateLeaf)
        ]
        assert moved.distribution.mean.tolist() == [12.0, -9.0]
