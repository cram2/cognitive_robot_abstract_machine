"""
A uniform prior answers a statement with nothing but what its conditions allow.
"""

import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a, and_, distribution_of, or_
from krrood.parametrization.exceptions import UnboundedParameterError
from krrood.parametrization.model_registries import UniformPriorRegistry
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from ._fixtures import Coin

# %% fixtures

NUMBER_OF_DRAWS = 200
"""
How many coins are drawn per test; enough that a region holding half the probability is
drawn from with near certainty.
"""


def two_separate_regions():
    """
    :return: A statement allowing ``a`` between 0 and 1 or between 3 and 4, with ``b``
        between 0 and 2 and ``c`` between 0 and 1 in both.
    """
    coin = a(Coin)(a=..., b=..., c=...)
    coin.where(
        or_(
            and_(coin.a >= 0.0, coin.a <= 1.0, coin.b >= 0.0, coin.b <= 2.0),
            and_(coin.a >= 3.0, coin.a <= 4.0, coin.b >= 0.0, coin.b <= 2.0),
        ),
        coin.c >= 0.0,
        coin.c <= 1.0,
    )
    return coin


def draw(statement, number_of_draws: int = NUMBER_OF_DRAWS) -> list:
    """
    :return: Coins answering ``statement``, drawn from a uniform prior.
    """
    return list(
        statement.evaluate(
            backend=ProbabilisticBackend(
                UniformPriorRegistry(), number_of_samples=number_of_draws
            )
        )
    )


# %% drawing


def test_draws_stay_within_the_allowed_regions():
    coins = draw(two_separate_regions())

    assert len(coins) == NUMBER_OF_DRAWS
    for coin in coins:
        assert 0.0 <= coin.a <= 1.0 or 3.0 <= coin.a <= 4.0
        assert 0.0 <= coin.b <= 2.0
        assert 0.0 <= coin.c <= 1.0


def test_draws_come_from_every_allowed_region():
    coins = draw(two_separate_regions())

    assert any(coin.a <= 1.0 for coin in coins)
    assert any(coin.a >= 3.0 for coin in coins)


def test_equally_large_regions_are_equally_likely():
    distribution = distribution_of(two_separate_regions()).first(
        backend=ProbabilisticBackend(UniformPriorRegistry())
    )
    [variable_a] = [
        variable for variable in distribution.variables if variable.name == "Coin.a"
    ]

    first_region = distribution.probability(
        SimpleEvent.from_data({variable_a: closed(0.0, 1.0)}).as_composite_set()
    )

    assert first_region == pytest.approx(0.5)


def test_the_prior_is_a_tensorized_circuit():
    distribution = distribution_of(two_separate_regions()).first(
        backend=ProbabilisticBackend(UniformPriorRegistry())
    )

    assert isinstance(distribution, LayeredProbabilisticCircuit)


def test_a_parameter_without_conditions_is_refused():
    coin = a(Coin)(a=..., b=..., c=...)
    coin.where(coin.a >= 0.0, coin.a <= 1.0, coin.b >= 0.0, coin.b <= 1.0)

    with pytest.raises(UnboundedParameterError):
        draw(coin, number_of_draws=1)


def test_a_parameter_bounded_from_one_side_only_is_refused():
    coin = a(Coin)(a=..., b=..., c=...)
    coin.where(
        coin.a >= 0.0, coin.a <= 1.0, coin.b >= 0.0, coin.b <= 1.0, coin.c >= 0.0
    )

    with pytest.raises(UnboundedParameterError):
        draw(coin, number_of_draws=1)
