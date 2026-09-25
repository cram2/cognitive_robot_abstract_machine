import copy
import math

import numpy as np
import pytest
from random_events.interval import closed, open_closed, singleton
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.variable import Continuous
from scipy.stats import norm, truncnorm
from scipy.stats._multivariate import multivariate_normal_frozen

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.distributions.truncated_multivariate_gaussian import (
    TruncatedMultivariateGaussianDistribution,
)
from probabilistic_model.exceptions import (
    EventIsNotABoxError,
    ProbabilisticCircuitRequiredError,
    ShapeMismatchError,
    VariableNotInDistributionError,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel

# %% shared fixtures


@pytest.fixture
def horizontal() -> Continuous:
    return Continuous("horizontal")


@pytest.fixture
def vertical() -> Continuous:
    return Continuous("vertical")


@pytest.fixture
def independent(horizontal, vertical) -> MultivariateGaussianDistribution:
    """
    Two variables that do not co-vary, so the joint is the product of two marginals and
    every answer can be checked against the univariate distribution already in this
    package.
    """
    return MultivariateGaussianDistribution(
        variables=(horizontal, vertical),
        mean=np.array([1.0, -2.0]),
        covariance=Covariance.from_matrix(np.array([[4.0, 0.0], [0.0, 9.0]])),
    )


@pytest.fixture
def correlated(horizontal, vertical) -> MultivariateGaussianDistribution:
    """
    Standard variables correlated by 0.6, which is the case with no closed form for the
    probability of a box.
    """
    return MultivariateGaussianDistribution(
        variables=(horizontal, vertical),
        mean=np.array([0.0, 0.0]),
        covariance=Covariance.from_matrix(np.array([[1.0, 0.6], [0.6, 1.0]])),
    )


def mean_of(distribution: ProbabilisticModel, variable: Continuous) -> float:
    """
    :return: The expectation of one variable.
    """
    return distribution.expectation([variable])[variable]


def variance_of(distribution: ProbabilisticModel, variable: Continuous) -> float:
    """
    :return: The variance of one variable.
    """
    return distribution.variance([variable])[variable]


def marginal_of(
    distribution: MultivariateGaussianDistribution, variable: Continuous
) -> GaussianDistribution:
    """
    :return: The univariate distribution of one variable, to check a joint answer
        against the implementation this package already has.
    """
    return GaussianDistribution(
        variable=variable,
        location=mean_of(distribution, variable),
        scale=math.sqrt(variance_of(distribution, variable)),
    )


def one_variable(
    variable: Continuous, mean: float, variance: float
) -> MultivariateGaussianDistribution:
    """
    :return: A distribution over a single variable, which several tests need and which
        carries no layout worth restating at each of them.
    """
    return MultivariateGaussianDistribution(
        variables=(variable,),
        mean=np.array([mean]),
        covariance=Covariance.from_matrix(np.array([[variance]])),
    )


# %% the covariance matrix


class TestCovariance:
    matrix = np.array([[1.0, 0.6, 0.0], [0.6, 2.0, -0.3], [0.0, -0.3, 3.0]])

    def test_only_the_lower_triangle_of_the_matrix_is_stored(self):
        covariance = Covariance.from_matrix(self.matrix)
        rows, columns = np.tril_indices(len(self.matrix))
        assert covariance.lower_triangle.tolist() == self.matrix[rows, columns].tolist()

    def test_the_matrix_is_mirrored_from_the_lower_triangle(self):
        assert Covariance.from_matrix(self.matrix).matrix.tolist() == (
            self.matrix.tolist()
        )

    def test_its_dimension_is_the_size_of_the_matrix(self):
        assert Covariance.from_matrix(self.matrix).dimension == len(self.matrix)

    def test_an_entry_is_read_in_either_direction(self):
        covariance = Covariance.from_matrix(self.matrix)
        assert covariance.between(2, 1) == self.matrix[2, 1]
        assert covariance.between(1, 2) == self.matrix[1, 2]

    def test_the_variances_are_the_diagonal(self):
        assert Covariance.from_matrix(self.matrix).variances.tolist() == (
            np.diag(self.matrix).tolist()
        )

    def test_a_marginal_keeps_the_rows_and_columns_asked_for(self):
        marginal = Covariance.from_matrix(self.matrix).marginal([2, 0])
        assert marginal.matrix.tolist() == self.matrix[np.ix_([2, 0], [2, 0])].tolist()

    def test_scaling_multiplies_each_entry_once_per_index_it_relates(self):
        factors = np.array([2.0, 1.0, 3.0])
        scaled = Covariance.from_matrix(self.matrix).scaled(factors)
        assert scaled.matrix == pytest.approx(self.matrix * np.outer(factors, factors))

    def test_a_lower_triangle_of_the_wrong_length_is_rejected(self):
        with pytest.raises(ShapeMismatchError) as error:
            Covariance(lower_triangle=np.zeros(4))
        assert error.value.expected_shape == (3,)
        assert error.value.received_shape == (4,)

    def test_a_matrix_that_is_not_square_is_rejected(self):
        with pytest.raises(ShapeMismatchError) as error:
            Covariance.from_matrix(np.zeros((2, 3)))
        assert error.value.expected_shape == (2, 2)
        assert error.value.received_shape == (2, 3)


# %% what the distribution is built from and holds


class TestBuildingADistribution:
    def test_the_mean_and_the_variance_are_read_by_variable(
        self, independent, horizontal, vertical
    ):
        assert mean_of(independent, horizontal) == 1.0
        assert mean_of(independent, vertical) == -2.0
        assert variance_of(independent, horizontal) == 4.0
        assert variance_of(independent, vertical) == 9.0

    def test_a_covariance_is_read_in_either_direction(
        self, correlated, horizontal, vertical
    ):
        assert correlated.covariance_between(horizontal, vertical) == 0.6
        assert correlated.covariance_between(vertical, horizontal) == 0.6

    def test_a_distribution_about_one_variable_needs_no_layout(self, horizontal):
        distribution = MultivariateGaussianDistribution(
            variables=(horizontal,),
            mean=np.array([3.0]),
            covariance=Covariance.from_matrix(np.array([[0.25]])),
        )
        assert mean_of(distribution, horizontal) == 3.0
        assert variance_of(distribution, horizontal) == 0.25
        assert distribution.variables == (horizontal,)

    def test_a_mean_that_is_not_laid_out_by_the_variables_is_rejected(
        self, horizontal, vertical
    ):
        with pytest.raises(ShapeMismatchError) as error:
            MultivariateGaussianDistribution(
                variables=(horizontal, vertical),
                mean=np.array([0.0]),
                covariance=Covariance.from_matrix(np.zeros((2, 2))),
            )
        assert error.value.expected_shape == (2,)
        assert error.value.received_shape == (1,)

    def test_a_covariance_of_another_dimension_is_rejected(self, horizontal, vertical):
        with pytest.raises(ShapeMismatchError) as error:
            MultivariateGaussianDistribution(
                variables=(horizontal, vertical),
                mean=np.zeros(2),
                covariance=Covariance(lower_triangle=np.ones(1)),
            )
        assert error.value.expected_shape == (2, 2)
        assert error.value.received_shape == (1, 1)

    def test_reading_a_variable_the_distribution_is_not_about_is_rejected(
        self, independent
    ):
        absent = Continuous("absent")
        with pytest.raises(VariableNotInDistributionError) as error:
            mean_of(independent, absent)
        assert error.value.variable == absent

    def test_the_variables_keep_the_layout_order(
        self, independent, horizontal, vertical
    ):
        assert independent.variables == (horizontal, vertical)

    def test_the_mean_and_covariance_are_laid_out_by_the_variables(
        self, independent, horizontal, vertical
    ):
        assert independent.mean.tolist() == [1.0, -2.0]
        assert independent.covariance.matrix.tolist() == [[4.0, 0.0], [0.0, 9.0]]

    def test_the_support_is_every_value_the_variables_can_take(self, independent):
        """
        A Gaussian rules nothing out, so its support is the universal event over its own
        variables rather than one rebuilt from the reals.
        """
        assert independent.support == (
            independent.universal_simple_event().as_composite_set()
        )

    def test_every_tractable_query_is_answered_by_one_scipy_distribution(
        self, independent
    ):
        assert isinstance(independent.scipy_distribution, multivariate_normal_frozen)


# %% density


class TestLikelihood:
    def test_the_joint_density_of_independent_variables_is_the_product_of_the_marginals(
        self, independent, horizontal, vertical
    ):
        point = np.array([[0.5, -1.0]])
        expected = (
            marginal_of(independent, horizontal).likelihood(np.array([[0.5]]))[0]
            * marginal_of(independent, vertical).likelihood(np.array([[-1.0]]))[0]
        )
        assert independent.likelihood(point)[0] == pytest.approx(expected)

    def test_every_point_is_scored(self, independent):
        points = np.array([[0.0, 0.0], [1.0, -2.0], [5.0, 5.0]])
        assert independent.log_likelihood(points).shape == (3,)

    def test_the_mean_is_the_most_likely_point(self, correlated):
        at_mean = correlated.log_likelihood(np.array([[0.0, 0.0]]))[0]
        elsewhere = correlated.log_likelihood(np.array([[0.4, -0.7]]))[0]
        assert at_mean > elsewhere


# %% the probability of a box, which has no closed form when the variables co-vary


class TestProbabilityOfABox:
    def test_the_whole_support_is_certain(self, correlated):
        assert correlated.probability(correlated.support) == pytest.approx(1.0)

    def test_a_box_over_independent_variables_is_the_product_of_the_marginals(
        self, independent, horizontal, vertical
    ):
        event = SimpleEvent.from_data(
            {horizontal: closed(0.0, 2.0), vertical: closed(-3.0, 1.0)}
        )
        expected = marginal_of(independent, horizontal).probability_of_simple_event(
            event
        ) * marginal_of(independent, vertical).probability_of_simple_event(event)
        assert independent.probability_of_simple_event(event) == pytest.approx(expected)

    def test_a_quadrant_of_correlated_variables_follows_the_orthant_formula(
        self, correlated, horizontal, vertical
    ):
        """
        The probability that two standard correlated variables are both positive is
        ``1/4 + arcsin(correlation) / 2pi`` — the one box probability a correlated
        Gaussian has in closed form, so it checks the numerical integration against
        something other than itself.
        """
        both_positive = SimpleEvent.from_data(
            {
                horizontal: open_closed(0.0, np.inf),
                vertical: open_closed(0.0, np.inf),
            }
        )
        correlation = correlated.covariance_between(horizontal, vertical)
        assert correlated.probability_of_simple_event(both_positive) == pytest.approx(
            0.25 + math.asin(correlation) / (2 * math.pi)
        )

    def test_correlation_changes_the_probability_of_a_quadrant(
        self, correlated, independent, horizontal, vertical
    ):
        """
        Without this the integration could be ignoring the off-diagonal entirely and
        every other assertion here would still hold.
        """
        both_positive = SimpleEvent.from_data(
            {
                horizontal: open_closed(0.0, np.inf),
                vertical: open_closed(0.0, np.inf),
            }
        )
        assert correlated.probability_of_simple_event(both_positive) > 0.25

    def test_a_variable_confined_to_two_intervals_sums_them(
        self, independent, horizontal, vertical
    ):
        whole_column = closed(-np.inf, np.inf)
        lower = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0), vertical: whole_column}
        )
        upper = SimpleEvent.from_data(
            {horizontal: closed(3.0, 4.0), vertical: whole_column}
        )
        both = SimpleEvent.from_data(
            {
                horizontal: closed(0.0, 1.0) | closed(3.0, 4.0),
                vertical: whole_column,
            }
        )
        assert independent.probability_of_simple_event(both) == pytest.approx(
            independent.probability_of_simple_event(lower)
            + independent.probability_of_simple_event(upper)
        )

    def test_a_box_with_no_width_is_impossible(self, correlated, horizontal, vertical):
        flattened = SimpleEvent.from_data(
            {horizontal: singleton(0.0), vertical: closed(-1.0, 1.0)}
        )
        assert correlated.probability_of_simple_event(flattened) == 0.0


# %% the precision matrix


class TestPrecision:
    def test_the_precision_is_the_inverse_of_the_covariance(self, correlated):
        assert correlated.precision @ correlated.covariance.matrix == pytest.approx(
            np.eye(2)
        )


# %% the most likely point


class TestMode:
    def test_the_mode_is_the_mean(self, independent, horizontal, vertical):
        mode, likelihood = independent.mode()
        assert mode.contains(np.array([1.0, -2.0]))
        assert likelihood == pytest.approx(
            independent.likelihood(np.array([[1.0, -2.0]]))[0]
        )


# %% conditioning on a value


class TestConditioningOnAValue:
    def test_the_conditional_density_is_the_joint_divided_by_the_marginal(
        self, correlated, horizontal, vertical
    ):
        """
        The definition of conditioning, checked against the distribution's own joint
        density and its own marginal rather than against a second copy of the Gaussian
        conditioning formula.
        """
        given = 0.8
        conditioned, _ = correlated.conditional({vertical: given})
        somewhere = 0.3

        joint = correlated.likelihood(np.array([[somewhere, given]]))[0]
        marginal = correlated.marginal([vertical]).likelihood(np.array([[given]]))[0]
        assert conditioned.likelihood(np.array([[somewhere]]))[0] == pytest.approx(
            joint / marginal
        )

    def test_conditioning_leaves_only_the_free_variables(
        self, correlated, horizontal, vertical
    ):
        conditioned, _ = correlated.conditional({vertical: 0.0})
        assert conditioned.variables == (horizontal,)

    def test_the_probability_returned_is_the_marginal_density_of_the_value(
        self, correlated, vertical
    ):
        given = 0.8
        _, probability = correlated.conditional({vertical: given})
        assert probability == pytest.approx(
            correlated.marginal([vertical]).likelihood(np.array([[given]]))[0]
        )

    def test_conditioning_an_uncorrelated_variable_leaves_it_alone(
        self, independent, horizontal, vertical
    ):
        conditioned, _ = independent.conditional({vertical: 100.0})
        assert mean_of(conditioned, horizontal) == pytest.approx(
            mean_of(independent, horizontal)
        )
        assert variance_of(conditioned, horizontal) == pytest.approx(
            variance_of(independent, horizontal)
        )

    def test_conditioning_always_narrows_a_correlated_variable(
        self, correlated, horizontal, vertical
    ):
        conditioned, _ = correlated.conditional({vertical: 0.0})
        assert variance_of(conditioned, horizontal) < variance_of(
            correlated, horizontal
        )

    def test_the_conditioned_covariance_is_exactly_symmetric(self, horizontal):
        """
        Badly scaled entries make the conditioning arithmetic round differently on the
        two sides of the diagonal, which must not leave the covariance asymmetric.
        """
        first, second, given = horizontal, Continuous("second"), Continuous("given")
        distribution = MultivariateGaussianDistribution(
            variables=(first, second, given),
            mean=np.zeros(3),
            covariance=Covariance.from_matrix(
                np.array(
                    [
                        [1e8, 1e-2, 0.9e4],
                        [1e-2, 1e-8, 1e-5],
                        [0.9e4, 1e-5, 3.0],
                    ]
                )
            ),
        )
        conditioned, _ = distribution.conditional({given: 1.0})
        assert conditioned.covariance_between(
            first, second
        ) == conditioned.covariance_between(second, first)

    def test_conditioning_on_every_variable_needs_a_circuit(
        self, independent, horizontal, vertical
    ):
        """
        Nothing is left free, so what remains is a product of Dirac impulses, which only
        a probabilistic circuit represents.
        """
        with pytest.raises(ProbabilisticCircuitRequiredError) as error:
            independent.conditional({horizontal: 1.5, vertical: -0.5})
        assert error.value.model is independent

    def test_conditioning_on_an_unknown_variable_is_rejected(self, independent):
        absent = Continuous("absent")
        with pytest.raises(VariableNotInDistributionError) as error:
            independent.conditional({absent: 0.0})
        assert error.value.variable == absent


# %% multiplying by the Gaussian likelihood of an observation


class TestProductWithAGaussianLikelihood:
    def test_the_density_is_the_product_of_the_two_densities_up_to_normalization(
        self, correlated, horizontal, vertical
    ):
        """
        The answer divided by the two densities it multiplies is the same constant
        everywhere.
        """
        likelihood = MultivariateGaussianDistribution(
            variables=(horizontal, vertical),
            mean=np.array([1.0, -0.5]),
            covariance=Covariance.from_matrix(np.array([[0.5, 0.1], [0.1, 2.0]])),
        )
        product = correlated.product_with_gaussian_likelihood(likelihood)
        points = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]])
        log_normalization = (
            correlated.log_likelihood(points)
            + likelihood.log_likelihood(points)
            - product.log_likelihood(points)
        )
        assert log_normalization == pytest.approx(
            np.full(len(points), log_normalization[0])
        )

    def test_a_density_over_fewer_variables_multiplies_only_those(
        self, correlated, horizontal, vertical
    ):
        """
        A density over one of the variables is constant along the others, so the product
        divided by the two densities is again the same constant everywhere.
        """
        likelihood = one_variable(vertical, 2.0, 0.5)
        product = correlated.product_with_gaussian_likelihood(likelihood)
        points = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]])
        log_normalization = (
            correlated.log_likelihood(points)
            + likelihood.log_likelihood(points[:, [1]])
            - product.log_likelihood(points)
        )
        assert product.variables == correlated.variables
        assert log_normalization == pytest.approx(
            np.full(len(points), log_normalization[0])
        )

    def test_the_mean_moves_toward_the_other_mean(self, independent, horizontal):
        product = independent.product_with_gaussian_likelihood(
            one_variable(horizontal, 5.0, 1.0)
        )
        assert mean_of(independent, horizontal) < mean_of(product, horizontal) < 5.0

    def test_the_product_is_more_certain_than_either_factor(
        self, independent, horizontal
    ):
        product = independent.product_with_gaussian_likelihood(
            one_variable(horizontal, 5.0, 4.0)
        )
        assert variance_of(product, horizontal) < variance_of(independent, horizontal)

    def test_two_equally_certain_densities_meet_halfway(self, horizontal):
        """
        With both densities equally uncertain, neither outweighs the other, so the
        product mean is their midpoint exactly.
        """
        product = one_variable(horizontal, 0.0, 2.0).product_with_gaussian_likelihood(
            one_variable(horizontal, 10.0, 2.0)
        )
        assert mean_of(product, horizontal) == pytest.approx(5.0)

    def test_repeated_products_accumulate_into_the_covariance(self, horizontal):
        """
        Checked against the information form — precisions add — rather than against a
        stored number, so the recursion is verified against an independent formulation
        of the same law instead of a second copy of itself.
        """
        starting_variance, likelihood_variance, products = 1.0, 4.0, 100
        distribution = one_variable(horizontal, 0.0, starting_variance)
        for _ in range(products):
            distribution = distribution.product_with_gaussian_likelihood(
                one_variable(horizontal, 1.0, likelihood_variance)
            )

        expected_precision = 1 / starting_variance + products / likelihood_variance
        assert variance_of(distribution, horizontal) == pytest.approx(
            1 / expected_precision
        )

    def test_the_product_does_not_change_the_distribution_it_multiplied(
        self, independent, horizontal
    ):
        before = mean_of(independent, horizontal)
        independent.product_with_gaussian_likelihood(one_variable(horizontal, 5.0, 1.0))
        assert mean_of(independent, horizontal) == before

    def test_a_density_over_a_variable_it_is_not_over_is_rejected(self, independent):
        absent = Continuous("absent")
        with pytest.raises(VariableNotInDistributionError) as error:
            independent.product_with_gaussian_likelihood(one_variable(absent, 0.0, 1.0))
        assert error.value.variable == absent


# %% reading fewer variables than the distribution is about


class TestMarginal:
    def test_a_marginal_keeps_the_mean_and_covariance_of_what_it_kept(
        self, correlated, horizontal
    ):
        marginal = correlated.marginal([horizontal])
        assert marginal.variables == (horizontal,)
        assert mean_of(marginal, horizontal) == mean_of(correlated, horizontal)
        assert variance_of(marginal, horizontal) == variance_of(correlated, horizontal)

    def test_a_marginal_over_none_of_its_variables_is_nothing(self, correlated):
        assert correlated.marginal([Continuous("absent")]) is None

    def test_a_marginal_is_laid_out_in_the_distribution_s_own_order(
        self, correlated, horizontal, vertical
    ):
        assert correlated.marginal([vertical, horizontal]).variables == (
            horizontal,
            vertical,
        )


# %% moments


class TestMoments:
    def test_the_expectation_is_the_mean(self, independent, horizontal, vertical):
        expectation = independent.expectation()
        assert expectation[horizontal] == pytest.approx(1.0)
        assert expectation[vertical] == pytest.approx(-2.0)

    def test_the_variance_is_the_diagonal_of_the_covariance(
        self, independent, horizontal, vertical
    ):
        variance = independent.variance()
        assert variance[horizontal] == pytest.approx(4.0)
        assert variance[vertical] == pytest.approx(9.0)

    def test_a_higher_moment_matches_the_variable_s_own_marginal(
        self, independent, horizontal
    ):
        order = VariableMap({horizontal: 4})
        center = VariableMap({horizontal: 0.0})
        assert independent.moment(order, center)[horizontal] == pytest.approx(
            marginal_of(independent, horizontal).moment(order, center)[horizontal]
        )


# %% translation and scaling


class TestTranslationAndScaling:
    def test_translating_moves_the_mean_and_leaves_the_covariance(
        self, independent, horizontal, vertical
    ):
        independent.apply_translation({horizontal: 3.0})
        assert mean_of(independent, horizontal) == 4.0
        assert mean_of(independent, vertical) == -2.0
        assert variance_of(independent, horizontal) == 4.0

    def test_scaling_multiplies_the_mean_and_squares_into_the_variance(
        self, independent, horizontal, vertical
    ):
        independent.apply_scaling({horizontal: 2.0})
        assert mean_of(independent, horizontal) == 2.0
        assert variance_of(independent, horizontal) == 16.0
        assert variance_of(independent, vertical) == 9.0

    def test_scaling_carries_into_a_covariance_once_per_variable(
        self, correlated, horizontal, vertical
    ):
        correlated.apply_scaling({horizontal: 2.0})
        assert correlated.covariance_between(horizontal, vertical) == pytest.approx(1.2)

    def test_a_translation_of_a_variable_it_is_not_over_is_ignored(
        self, independent, horizontal
    ):
        independent.apply_translation({horizontal: 3.0, Continuous("absent"): 5.0})
        assert independent.mean.tolist() == [4.0, -2.0]

    def test_a_scaling_of_a_variable_it_is_not_over_is_ignored(
        self, independent, horizontal
    ):
        independent.apply_scaling({horizontal: 2.0, Continuous("absent"): 5.0})
        assert independent.covariance.matrix.tolist() == [[16.0, 0.0], [0.0, 9.0]]

    def test_a_variable_left_out_of_a_scaling_keeps_its_size(
        self, independent, vertical
    ):
        independent.apply_scaling({})
        assert mean_of(independent, vertical) == -2.0
        assert variance_of(independent, vertical) == 9.0


# %% sampling


class TestSampling:
    def test_every_sample_carries_one_number_per_variable(self, correlated):
        assert correlated.sample(7).shape == (7, 2)

    def test_samples_of_one_variable_are_still_a_column(self, horizontal):
        assert one_variable(horizontal, 0.0, 1.0).sample(5).shape == (5, 1)

    def test_samples_fall_where_the_distribution_says_they_should(self, independent):
        np.random.seed(69)
        samples = independent.sample(20000)
        assert samples.mean(axis=0) == pytest.approx(np.array([1.0, -2.0]), abs=0.1)


# %% copying


class TestCopying:
    def test_a_copy_moves_without_moving_the_original(self, independent, horizontal):
        copied = copy.copy(independent)
        copied.apply_translation({horizontal: 10.0})
        assert mean_of(independent, horizontal) == 1.0
        assert mean_of(copied, horizontal) == 11.0

    def test_a_deep_copy_is_the_same_distribution(self, correlated):
        copied = copy.deepcopy(correlated)
        assert copied.variables == correlated.variables
        assert copied.mean.tolist() == correlated.mean.tolist()
        assert (
            copied.covariance.matrix.tolist() == correlated.covariance.matrix.tolist()
        )

    def test_a_deep_copy_moves_without_moving_the_original(
        self, independent, horizontal
    ):
        copied = copy.deepcopy(independent)
        copied.apply_scaling({horizontal: 2.0})
        assert independent.covariance.matrix.tolist() == [[4.0, 0.0], [0.0, 9.0]]

    def test_a_deep_copy_of_a_truncated_distribution_keeps_its_box(
        self, correlated, horizontal, vertical
    ):
        truncated, _ = correlated.truncated(
            box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        )
        copied = copy.deepcopy(truncated)
        assert copied.box == truncated.box
        assert copied.normalizing_constant == truncated.normalizing_constant


# %% confining a distribution to an event


def box_over(horizontal, vertical, lower: float, upper: float) -> SimpleEvent:
    """
    :return: The same interval on both variables, which is the shape every truncation
        here is confined to.
    """
    return SimpleEvent.from_data(
        {horizontal: closed(lower, upper), vertical: closed(lower, upper)}
    )


def mode_point_of(truncated: TruncatedMultivariateGaussianDistribution) -> np.ndarray:
    """
    :return: The one point the mode is, read back in the distribution's own order, so a
        mode an optimiser lands next to can be compared to the value it should be.
    """
    mode, _ = truncated.log_mode()
    box = mode.simple_sets[0]
    return np.array(
        [box[variable].simple_sets[0].lower for variable in truncated.variables]
    )


class TestTruncation:
    def test_truncating_answers_with_a_distribution_that_is_no_longer_gaussian(
        self, correlated, horizontal, vertical
    ):
        """
        A correlated Gaussian confined to a box is not a Gaussian, so truncation cannot
        answer with one of its own kind.
        """
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, probability = correlated.truncated(box)
        assert isinstance(truncated, TruncatedMultivariateGaussianDistribution)
        assert probability == pytest.approx(correlated.probability(box))

    def test_an_event_of_more_than_one_box_is_not_this_class_to_answer(
        self, correlated, horizontal, vertical
    ):
        """
        A variable confined to two separate intervals leaves a shape that needs a
        circuit rather than one truncated Gaussian.
        """
        two_intervals = SimpleEvent.from_data(
            {
                horizontal: closed(0.0, 1.0) | closed(3.0, 4.0),
                vertical: closed(0.0, 1.0),
            }
        ).as_composite_set()
        with pytest.raises(EventIsNotABoxError) as error:
            correlated.truncated(two_intervals)
        assert error.value.model is correlated

    def test_an_impossible_event_leaves_nothing(self, correlated, horizontal, vertical):
        nothing = SimpleEvent.from_data(
            {horizontal: singleton(0.0), vertical: singleton(0.0)}
        ).as_composite_set()
        truncated, probability = correlated.truncated(nothing)
        assert truncated is None
        assert probability == 0.0

    def test_the_truncated_density_is_the_original_scaled_up_to_one(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, probability = correlated.truncated(box)
        inside = np.array([[0.5, 0.5]])
        assert truncated.likelihood(inside)[0] == pytest.approx(
            correlated.likelihood(inside)[0] / probability
        )

    def test_nothing_outside_the_event_can_happen(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        assert truncated.likelihood(np.array([[5.0, 5.0]]))[0] == 0.0

    def test_an_excluded_end_of_the_box_cannot_happen(
        self, correlated, horizontal, vertical
    ):
        box = SimpleEvent.from_data(
            {horizontal: open_closed(0.0, 1.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        likelihoods = truncated.likelihood(np.array([[0.0, 0.5], [1.0, 0.5]]))
        assert likelihoods[0] == 0.0
        assert likelihoods[1] > 0.0

    def test_the_box_reads_a_point_laid_out_in_the_order_of_the_variables(
        self, horizontal, vertical
    ):
        """
        The variables are laid out in the reverse of the order the box sorts them in, so
        a check that read the point in the box's order would rule it out.
        """
        distribution = MultivariateGaussianDistribution(
            variables=(vertical, horizontal),
            mean=np.zeros(2),
            covariance=Covariance.from_matrix(np.eye(2)),
        )
        box = SimpleEvent.from_data(
            {horizontal: closed(5.0, 6.0), vertical: closed(0.0, 1.0)}
        ).as_composite_set()
        truncated, _ = distribution.truncated(box)
        assert truncated.box_contains(np.array([0.5, 5.5]))
        assert not truncated.box_contains(np.array([5.5, 0.5]))

    def test_the_cumulative_distribution_of_variables_that_do_not_co_vary_is_the_product_of_truncated_normals(
        self, independent, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 2.0).as_composite_set()
        truncated, _ = independent.truncated(box)
        point = np.array([[1.0, 0.5]])
        expected = 1.0
        for index, variable in enumerate(independent.variables):
            deviation = math.sqrt(independent.covariance_between(variable, variable))
            location = independent.mean[index]
            expected *= truncnorm.cdf(
                point[0, index],
                a=(0.0 - location) / deviation,
                b=(2.0 - location) / deviation,
                loc=location,
                scale=deviation,
            )
        assert truncated.cumulative_distribution_function(point)[0] == pytest.approx(
            expected
        )

    def test_the_cumulative_distribution_is_zero_below_and_one_above_the_box(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        values = truncated.cumulative_distribution_function(
            np.array([[-1.0, 0.5], [2.0, 3.0]])
        )
        assert values[0] == 0.0
        assert values[1] == pytest.approx(1.0)

    def test_the_truncated_distribution_is_certain_of_its_own_event(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        assert truncated.probability(truncated.support) == pytest.approx(1.0)

    def test_the_mode_is_the_mean_when_the_mean_survived_the_truncation(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, -1.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        mode, _ = truncated.mode()
        assert mode.contains(np.array([0.0, 0.0]))

    def test_the_mode_of_variables_that_do_not_co_vary_is_the_mean_pulled_into_the_box(
        self, independent, horizontal, vertical
    ):
        """
        Variables that do not co-vary are most likely, within a box, exactly where each
        of them on its own is: at the point of its own interval nearest its own mean.
        """
        box = SimpleEvent.from_data(
            {horizontal: closed(3.0, 4.0), vertical: closed(-2.0, 0.0)}
        ).as_composite_set()
        truncated, _ = independent.truncated(box)
        assert mode_point_of(truncated) == pytest.approx(
            [3.0, mean_of(independent, vertical)]
        )

    def test_the_mode_is_on_the_boundary_once_the_mean_is_cut_away(
        self, correlated, horizontal, vertical
    ):
        """
        A Gaussian falls away from its mean in every direction, so a box that excludes
        the mean still has exactly one most likely point: its own corner nearest to it.
        """
        box = box_over(horizontal, vertical, 3.0, 4.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        _, log_density = truncated.log_mode()
        assert mode_point_of(truncated) == pytest.approx([3.0, 3.0])
        assert log_density == pytest.approx(
            truncated.log_likelihood(np.array([[3.0, 3.0]]))[0]
        )

    def test_a_variable_the_box_leaves_free_follows_the_one_it_confines(
        self, horizontal, vertical
    ):
        """
        Two variables that co-vary are not most likely where each of them on its own
        would be: confining one of them moves where the other is most likely with it.
        """
        strongly_correlated = MultivariateGaussianDistribution(
            variables=(horizontal, vertical),
            mean=np.array([0.0, 0.0]),
            covariance=Covariance.from_matrix(np.array([[1.0, 0.9], [0.9, 1.0]])),
        )
        box = SimpleEvent.from_data(
            {horizontal: closed(1.0, 2.0), vertical: closed(-5.0, 5.0)}
        ).as_composite_set()
        truncated, _ = strongly_correlated.truncated(box)
        assert mode_point_of(truncated) == pytest.approx([1.0, 0.9])

    def test_no_point_of_the_box_is_more_likely_than_the_mode(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 3.0, 4.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        _, log_density = truncated.log_mode()
        grid = np.linspace(3.0, 4.0, 11)
        elsewhere = np.array([[first, second] for first in grid for second in grid])
        assert max(truncated.log_likelihood(elsewhere)) <= log_density

    def test_the_mode_of_an_open_box_is_a_point_the_box_still_allows(
        self, correlated, horizontal, vertical
    ):
        """
        An interval that excludes its own lower end has no nearest point, so the mode is
        the next value there is rather than the end itself, which the box gives no
        density at all.
        """
        box = SimpleEvent.from_data(
            {horizontal: open_closed(3.0, 4.0), vertical: open_closed(3.0, 4.0)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(box)
        _, log_density = truncated.log_mode()
        assert log_density > -np.inf
        assert truncated.log_likelihood(np.array([[3.0, 3.0]]))[0] == -np.inf

    def test_truncating_again_narrows_the_event(self, correlated, horizontal, vertical):
        box = box_over(horizontal, vertical, 0.0, 2.0).as_composite_set()
        smaller = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        narrowed, probability = truncated.truncated(smaller)
        assert narrowed.likelihood(np.array([[1.5, 1.5]]))[0] == 0.0
        assert probability == pytest.approx(
            correlated.probability(smaller) / correlated.probability(box)
        )

    def test_the_burn_in_period_length_carries_over_to_a_further_truncation(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        truncated.burn_in_period_length = 7
        smaller = box_over(horizontal, vertical, 0.0, 0.5).as_composite_set()
        further, _ = truncated.truncated(smaller)
        assert further.burn_in_period_length == truncated.burn_in_period_length

    def test_the_burn_in_period_length_carries_over_to_a_conditional(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        truncated.burn_in_period_length = 7
        conditional, _ = truncated.log_conditional({horizontal: 0.5})
        assert conditional.burn_in_period_length == truncated.burn_in_period_length

    def test_every_sample_falls_inside_the_event(
        self, correlated, horizontal, vertical
    ):
        np.random.seed(69)
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        samples = truncated.sample(200)
        assert samples.shape == (200, 2)
        assert all(truncated.support.contains(sample) for sample in samples)

    def test_samples_of_correlated_variables_follow_the_truncated_distribution(
        self, horizontal, vertical
    ):
        """
        With ``vertical = rho * horizontal + sqrt(1 - rho^2) * z``, confining only
        ``horizontal`` to ``[1, inf)`` moves the mean of ``vertical`` to ``rho`` times
        the truncated normal mean ``pdf(1) / sf(1)``. Drawing each variable from its own
        interval alone would leave ``vertical`` centred at zero.
        """
        np.random.seed(69)
        rho = 0.9
        correlated = MultivariateGaussianDistribution(
            variables=(horizontal, vertical),
            mean=np.zeros(2),
            covariance=Covariance.from_matrix(np.array([[1.0, rho], [rho, 1.0]])),
        )
        confined = SimpleEvent.from_data(
            {horizontal: closed(1.0, np.inf), vertical: closed(-np.inf, np.inf)}
        ).as_composite_set()
        truncated, _ = correlated.truncated(confined)
        samples = truncated.sample(5000)
        assert samples[:, 1].mean() == pytest.approx(
            rho * norm.pdf(1.0) / norm.sf(1.0), abs=0.05
        )

    def test_a_box_the_distribution_almost_never_lands_in_is_still_sampled(
        self, correlated, horizontal, vertical
    ):
        """
        A box far out in the tails is almost never reached by drawing from the
        untruncated distribution, and must still be sampled.
        """
        np.random.seed(69)
        unlikely = box_over(horizontal, vertical, 8.0, 9.0).as_composite_set()
        truncated, _ = correlated.truncated(unlikely)
        samples = truncated.sample(50)
        assert samples.shape == (50, 2)
        assert all(truncated.support.contains(sample) for sample in samples)

    def test_variables_that_do_not_co_vary_are_sampled_in_a_distant_box(
        self, independent, horizontal, vertical
    ):
        np.random.seed(69)
        unlikely = SimpleEvent.from_data(
            {horizontal: closed(20.0, 21.0), vertical: closed(20.0, 21.0)}
        ).as_composite_set()
        truncated, _ = independent.truncated(unlikely)
        samples = truncated.sample(50)
        assert samples.shape == (50, 2)
        assert all(truncated.support.contains(sample) for sample in samples)


# %% fixing a variable of a distribution that has already been confined


class TestConditioningATruncatedDistribution:
    def test_it_answers_with_the_slice_the_box_makes_at_that_value(
        self, correlated, horizontal, vertical
    ):
        """
        Fixing one variable of a truncated Gaussian leaves the Gaussian conditional,
        itself confined to what the box still allows the rest.
        """
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        conditional, _ = truncated.conditional({vertical: 0.5})
        assert isinstance(conditional, TruncatedMultivariateGaussianDistribution)
        assert conditional.variables == (horizontal,)
        assert conditional.likelihood(np.array([[1.5]]))[0] == 0.0
        assert conditional.probability(conditional.support) == pytest.approx(1.0)

    def test_the_slice_is_the_untruncated_conditional_confined_the_same_way(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        conditional, _ = truncated.conditional({vertical: 0.5})

        untruncated_conditional, _ = correlated.conditional({vertical: 0.5})
        interval = SimpleEvent.from_data(
            {horizontal: closed(0.0, 1.0)}
        ).as_composite_set()
        expected, _ = untruncated_conditional.truncated(interval)
        inside = np.array([[0.25]])
        assert conditional.likelihood(inside)[0] == pytest.approx(
            expected.likelihood(inside)[0]
        )

    def test_the_density_returned_is_what_splits_the_joint_into_the_two_halves(
        self, correlated, horizontal, vertical
    ):
        """
        A conditional density times the density of what was fixed is the joint density,
        which is what the number answered alongside the conditional has to be.
        """
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        conditional, log_density = truncated.log_conditional({vertical: 0.5})
        assert truncated.likelihood(np.array([[0.25, 0.5]]))[0] == pytest.approx(
            math.exp(log_density) * conditional.likelihood(np.array([[0.25]]))[0]
        )

    def test_a_value_the_box_rules_out_leaves_nothing(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        conditional, log_density = truncated.log_conditional({vertical: 5.0})
        assert conditional is None
        assert log_density == -np.inf

    def test_fixing_every_variable_needs_a_circuit(
        self, correlated, horizontal, vertical
    ):
        box = box_over(horizontal, vertical, 0.0, 1.0).as_composite_set()
        truncated, _ = correlated.truncated(box)
        with pytest.raises(ProbabilisticCircuitRequiredError):
            truncated.log_conditional({horizontal: 0.25, vertical: 0.5})
