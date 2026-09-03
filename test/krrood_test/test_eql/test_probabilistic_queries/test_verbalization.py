from krrood.entity_query_language.factories import a, and_, distribution_of, probability_of, variable
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

from ._fixtures import Coin


def test_verbalize_distribution_of_match():
    match = a(Coin)(a=..., b=..., c=1.5)
    match.where(match.variable.a > 0.2)

    text = verbalize_expression(distribution_of(match))

    # a and b are underspecified -- the distribution's free variables by default, so
    # they're not named explicitly ("the distribution over a Coin" already means "over
    # every attribute not given"), and definitely not "predicted", the match's
    # generative-sampling framing, which doesn't fit a distribution query at all
    assert text.startswith("The distribution over a Coin")
    assert "predict" not in text
    assert "given that its c is 1.5" in text
    assert "where its a is greater than 0.2" in text


def test_verbalize_distribution_of_select_all():
    match = a(Coin)(a=..., b=..., c=...)

    text = verbalize_expression(distribution_of(match))

    assert text == "The distribution over a Coin"


def test_verbalize_distribution_of_narrowed_to_variables():
    match = a(Coin)(a=..., b=..., c=1.5)

    text = verbalize_expression(distribution_of(match, marginalize_for=(match.variable.a,)))

    # marginalization narrows which variable(s) the joint is over -- named directly in
    # the subject, not a trailing "restricted to" qualifier (that reads like a
    # truncation, a where-style condition, not a choice of variables)
    assert text.startswith("The distribution over the a of a Coin")


def test_verbalize_probability_of_condition():
    x = variable(Coin)

    text = verbalize_expression(probability_of(x.a < 0.5))

    assert text == "the probability that the a of a Coin is less than 0.5"


def test_verbalize_probability_of_conjunction():
    x = variable(Coin)

    text = verbalize_expression(probability_of(and_(x.a < 0.5, x.b < 1)))

    assert text.startswith("the probability that")
    assert "the a of a Coin is less than 0.5" in text
    assert "the b of the Coin is less than 1" in text
