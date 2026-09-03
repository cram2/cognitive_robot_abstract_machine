import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a, distribution_of, variable
from krrood.parametrization.model_registries import DictRegistry

from ._fixtures import Coin, build_three_independent_variables_circuit


def test_distribution_conditions_and_truncates():
    circuit, var_a, var_b, var_c = build_three_independent_variables_circuit()
    backend = ProbabilisticBackend(model_registry=DictRegistry({Coin: circuit}))

    x = a(Coin)(a=..., b=..., c=1.5)
    x.where(x.variable.a > 0.2)
    result = distribution_of(x).first(backend=backend)

    # a=..., b=... are free; c=1.5 conditions the circuit (a point mass at 1.5, not
    # dropped from the model's variables); a > 0.2 truncates
    assert {v.name for v in result.variables} == {"Coin.a", "Coin.b", "Coin.c"}

    a_range = SimpleEvent.from_data({var_a: closed(0.2, 1)}).as_composite_set()
    assert result.probability(a_range) == pytest.approx(1.0)

    # b is independent of a/c, so truncating/conditioning on them leaves it unchanged
    b_range = SimpleEvent.from_data({var_b: closed(0, 1)}).as_composite_set()
    assert result.probability(b_range) == pytest.approx(0.5)

    c_point = SimpleEvent.from_data({var_c: closed(1.5, 1.5)}).as_composite_set()
    assert result.probability(c_point) == pytest.approx(1.0)


def test_distribution_narrowed_to_selected_variables():
    circuit, var_a, var_b, var_c = build_three_independent_variables_circuit()
    backend = ProbabilisticBackend(model_registry=DictRegistry({Coin: circuit}))

    x = a(Coin)(a=..., b=..., c=1.5)
    x.where(x.variable.a > 0.2)
    result = distribution_of(x, marginalize_for=(x.variable.a,)).first(backend=backend)

    assert {v.name for v in result.variables} == {"Coin.a"}
    a_range = SimpleEvent.from_data({var_a: closed(0.2, 1)}).as_composite_set()
    assert result.probability(a_range) == pytest.approx(1.0)


def test_distribution_no_conditions_or_where_is_the_full_joint():
    circuit, var_a, var_b, var_c = build_three_independent_variables_circuit()
    backend = ProbabilisticBackend(model_registry=DictRegistry({Coin: circuit}))

    x = a(Coin)(a=..., b=..., c=...)
    result = distribution_of(x).first(backend=backend)

    assert {v.name for v in result.variables} == {"Coin.a", "Coin.b", "Coin.c"}
    a_range = SimpleEvent.from_data({var_a: closed(0, 0.5)}).as_composite_set()
    assert result.probability(a_range) == pytest.approx(0.5)
