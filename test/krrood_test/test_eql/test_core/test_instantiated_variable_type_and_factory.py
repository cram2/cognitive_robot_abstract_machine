from __future__ import annotations

import pytest

from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.exceptions import (
    ResultTypeCannotBeDetermined,
    ResultTypeIsNotAType,
)
from krrood.entity_query_language.factories import an, inference, variable_from
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs
from ...dataset.instantiated_variable_operations import (
    Halved,
    Interval,
    IsEven,
    UnannotatedHalved,
    doubled,
    repeated,
    unannotated_doubled,
    unconstrained_doubled,
)


# %% inference of a type


def test_inferred_type_is_both_result_type_and_factory():
    interval = inference(Interval)(lower=variable_from([1]), upper=variable_from([3]))
    assert interval._type_ is Interval
    assert interval._factory_ is Interval


def test_inference_builds_with_a_factory_method():
    interval = inference(Interval.around)(
        center=variable_from([5]), radius=variable_from([2])
    )
    assert interval._type_ is Interval
    assert interval._factory_ == Interval.around
    assert interval.tolist() == [Interval.around(center=5, radius=2)]


def test_declared_factory_of_an_inferred_type_is_its_factory():
    interval = inference(Interval)(lower=variable_from([1]), upper=variable_from([3]))
    assert interval._declared_factory_ is Interval


def test_inferred_variable_is_a_factory_with_keyword_arguments():
    lower = variable_from([1])
    interval = inference(Interval)(lower=lower, upper=variable_from([3]))
    assert isinstance(interval, HasFactoryAndKwargs)
    (stated_lower,) = interval._stated_values_(keyword="lower")
    assert stated_lower is lower


def test_a_result_type_that_is_not_a_type_is_rejected():
    with pytest.raises(ResultTypeIsNotAType):
        InstantiatedVariable(_factory_=Interval.around, _type_=Interval.around)


def test_match_on_a_factory_method_has_the_type_the_method_returns():
    assert an(Interval.around)._type_ is Interval


def test_match_on_a_factory_method_returning_self_has_the_class_it_is_bound_to():
    assert an(Interval.point)._type_ is Interval


def test_inference_with_a_factory_method_returning_self_has_the_class_it_is_bound_to():
    assert inference(Interval.point)(value=variable_from([4]))._type_ is Interval


# %% symbolic functions


def test_symbolic_function_factory_is_the_function_and_type_its_return_type():
    expression = doubled(variable_from([1]))
    assert expression._factory_ is doubled.__wrapped__
    assert expression._type_ is int


def test_symbolic_function_declared_factory_is_the_decorated_function():
    assert doubled(variable_from([1]))._declared_factory_ is doubled


def test_symbolic_function_returning_a_generic_has_the_generic_origin_as_type():
    assert repeated(variable_from([1]))._type_ is list


def test_unannotated_symbolic_function_is_rejected():
    with pytest.raises(ResultTypeCannotBeDetermined):
        unannotated_doubled(variable_from([1]))


def test_symbolic_function_returning_any_has_object_as_type():
    assert unconstrained_doubled(variable_from([1]))._type_ is object


def test_symbolic_function_is_named_after_its_function():
    assert doubled(variable_from([1]))._name_ == doubled.__name__


def test_symbolic_function_still_binds_its_computed_value():
    assert doubled(variable_from([1, 2])).tolist() == [doubled(1), doubled(2)]


# %% symbolic callable classes


def test_predicate_type_and_factory_are_the_predicate_class():
    expression = IsEven(number=variable_from([2]))
    assert expression._type_ is IsEven
    assert expression._factory_ is IsEven


def test_symbolic_function_class_type_is_the_return_type_of_its_call():
    expression = Halved(number=variable_from([4]))
    assert expression._type_ is int
    assert expression._factory_ is Halved


def test_symbolic_function_class_with_unannotated_call_is_rejected():
    with pytest.raises(ResultTypeCannotBeDetermined):
        UnannotatedHalved(number=variable_from([4]))


def test_symbolic_function_class_still_binds_its_computed_value():
    assert Halved(number=variable_from([4, 7])).tolist() == [Halved(number=4)(), Halved(number=7)()]
