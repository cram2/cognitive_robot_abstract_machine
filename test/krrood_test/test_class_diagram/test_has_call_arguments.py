"""Tests of the things that stand for a callee and the arguments it is called
with."""

from __future__ import annotations

from krrood.class_diagrams.call_arguments import CallArguments
from krrood.entity_query_language.ground_term import CallTerm
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs


def scale(value: float, factor: float = 2.0) -> float:
    return value * factor


# %% bound call arguments


def test_call_term_binds_positional_and_defaulted_arguments():
    term = CallTerm(scale, CallArguments((3.0,)))
    assert term._bound_call_arguments_() == CallArguments((3.0, 2.0))


def test_factory_and_kwargs_binds_its_keyword_arguments():
    pattern = HasFactoryAndKwargs(scale, _kwargs_={"value": 3.0})
    assert pattern._callee_ is scale
    assert pattern._bound_call_arguments_() == CallArguments((3.0, 2.0))
