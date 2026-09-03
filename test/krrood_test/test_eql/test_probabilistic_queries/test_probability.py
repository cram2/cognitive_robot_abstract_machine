import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent

from krrood.entity_query_language.backends import EntityQueryLanguageBackend, ProbabilisticBackend
from krrood.entity_query_language.factories import and_, probability_of, variable
from krrood.parametrization.exceptions import JointQueryAcrossClassesNotSupported
from krrood.parametrization.model_registries import DictRegistry

from ._fixtures import Coin, OtherClass, build_three_independent_variables_circuit

# A small, hand-picked domain (not sampled) so the expected fraction is exact, not an
# approximation -- 4 out of 6 coins have a < 0.5.
_COIN_DOMAIN = [
    Coin(a=0.1, b=1.0, c=1.0),
    Coin(a=0.2, b=1.0, c=1.0),
    Coin(a=0.3, b=0.5, c=1.0),
    Coin(a=0.4, b=0.5, c=1.0),
    Coin(a=0.6, b=1.0, c=1.0),
    Coin(a=0.9, b=1.0, c=1.0),
]


def test_probability_matches_direct_computation():
    circuit, var_a, var_b, var_c = build_three_independent_variables_circuit()
    x = variable(Coin)

    backend = ProbabilisticBackend(model_registry=DictRegistry({Coin: circuit}))
    result = probability_of(x.a < 0.5).first(backend=backend)

    expected = circuit.probability(
        SimpleEvent.from_data({var_a: closed(0, 0.5)}).as_composite_set()
    )
    assert result == pytest.approx(expected)
    # a is independent of b/c and uniform on [0, 1], so P(a < 0.5) == 0.5 regardless
    assert result == pytest.approx(0.5)


def test_probability_of_conjunction():
    circuit, var_a, var_b, var_c = build_three_independent_variables_circuit()
    x = variable(Coin)

    backend = ProbabilisticBackend(model_registry=DictRegistry({Coin: circuit}))
    result = probability_of(and_(x.a < 0.5, x.b < 1)).first(backend=backend)

    # independent uniforms: P(a < 0.5) * P(b < 1) == 0.5 * 0.5
    assert result == pytest.approx(0.25)


def test_probability_rejects_cross_class():
    x = variable(Coin)
    y = variable(OtherClass)

    with pytest.raises(JointQueryAcrossClassesNotSupported):
        probability_of(and_(x.a < 0.5, y.d < 0.5)).first(
            backend=ProbabilisticBackend(model_registry=DictRegistry({}))
        )


def test_probability_evaluates_natively_by_counting_matching_rows():
    """
    Unlike distribution_of, probability_of also has a native evaluation strategy: a
    probability is definitionally the fraction of a domain's rows the condition holds
    for, so it's counted directly -- no ProbabilisticBackend/fitted model needed, just
    an enumerable domain.
    """
    x = variable(Coin, domain=_COIN_DOMAIN)
    result = probability_of(x.a < 0.5).first()  # no backend -> defaults to native

    assert result == pytest.approx(4 / 6)


def test_probability_native_evaluation_explicit_backend_and_conjunction():
    x = variable(Coin, domain=_COIN_DOMAIN)
    result = probability_of(and_(x.a < 0.5, x.b < 1)).first(
        backend=EntityQueryLanguageBackend()
    )

    # 2 of 6 coins have both a < 0.5 and b < 1 (a=0.3,b=0.5 and a=0.4,b=0.5)
    assert result == pytest.approx(2 / 6)


def test_probability_of_true_rejected_natively_too():
    """
    A content-free condition names no class either way -- there's neither a model to
    resolve (ProbabilisticBackend) nor a domain to count over (native).
    """
    with pytest.raises(JointQueryAcrossClassesNotSupported):
        probability_of(True).first()
