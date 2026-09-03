import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import (
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
)
from krrood.entity_query_language.factories import average, variable
from krrood.parametrization.model_registries import DictRegistry

from ._fixtures import Coin, build_three_independent_variables_circuit


def test_average_resolves_to_the_expectation_under_probabilistic_backend():
    circuit, var_a, var_b, var_c = build_three_independent_variables_circuit()
    x = variable(Coin)

    backend = ProbabilisticBackend(model_registry=DictRegistry({Coin: circuit}))

    # midpoints of Uniform([0, 1]) and Uniform([0, 2]) -- computed in closed form via
    # ProbabilisticModel.moment, not by sampling and averaging rows
    assert average(x.a).first(backend=backend) == pytest.approx(0.5)
    assert average(x.b).first(backend=backend) == pytest.approx(1.0)


def test_average_still_works_natively():
    heights = [1, 2, 3, 4, 5]
    heights_var = variable(int, domain=heights)

    assert average(heights_var).first() == pytest.approx(sum(heights) / len(heights))


def test_average_with_grouping_is_not_reinterpreted_probabilistically():
    circuit, var_a, _, _ = build_three_independent_variables_circuit()
    x = variable(Coin)

    backend = ProbabilisticBackend(model_registry=DictRegistry({Coin: circuit}))

    # ProbabilisticBackend only shortcuts a bare average(...) selection -- one grouped
    # (or filtered/ordered) needs enumerated data to group over, which a bare
    # variable(...) has none of, so it falls through to the ordinary generative-backend
    # error rather than silently ignoring the grouping.
    with pytest.raises(GenerativeBackendQueryIsNotUnderspecifiedVariable):
        average(x.a).grouped_by(x).first(backend=backend)
