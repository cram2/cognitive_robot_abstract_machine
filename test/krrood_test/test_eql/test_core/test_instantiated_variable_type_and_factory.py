from __future__ import annotations

import pytest

from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.exceptions import ResultTypeIsNotAType
from krrood.entity_query_language.factories import inference, variable_from
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs
from ...dataset.instantiated_variable_operations import (
    Halved,
    Interval,
    IsEven,
    doubled,
    repeated,
    unannotated_doubled,
)


# %% inference of a type


def test_inferred_type_is_both_result_type_and_factory():
    interval = inference(Interval)(lower=variable_from([1]), upper=variable_from([3]))
    assert interval._type_ is Interval
    assert interval._factory_ is Interval


def test_inference_builds_with_a_factory_method():
    interval = inference(Interval, factory=Interval.around)(
        center=variable_from([5]), radius=variable_from([2])
    )
    assert interval._type_ is Interval
    assert interval._factory_ == Interval.around
    assert interval.tolist() == [Interval.around(5, 2)]


def test_inferred_variable_is_a_factory_with_keyword_arguments():
    lower = variable_from([1])
    interval = inference(Interval)(lower=lower, upper=variable_from([3]))
    assert isinstance(interval, HasFactoryAndKwargs)
    assert list(interval._stated_values_("lower")) == [lower]


def test_a_result_type_that_is_not_a_type_is_rejected():
    with pytest.raises(ResultTypeIsNotAType):
        InstantiatedVariable(_type_=Interval.around)


# %% symbolic functions


def test_symbolic_function_factory_is_the_function_and_type_its_return_type():
    expression = doubled(variable_from([1]))
    assert expression._factory_ is doubled.__wrapped__
    assert expression._type_ is int


def test_symbolic_function_returning_a_generic_has_the_generic_origin_as_type():
    assert repeated(variable_from([1]))._type_ is list


def test_unannotated_symbolic_function_has_object_as_type():
    assert unannotated_doubled(variable_from([1]))._type_ is object


def test_symbolic_function_is_named_after_its_function():
    assert doubled(variable_from([1]))._name_ == doubled.__name__


def test_symbolic_function_still_binds_its_computed_value():
    assert doubled(variable_from([1, 2])).tolist() == [doubled(1), doubled(2)]


# %% symbolic callable classes


def test_predicate_type_and_factory_are_the_predicate_class():
    expression = IsEven(variable_from([2]))
    assert expression._type_ is IsEven
    assert expression._factory_ is IsEven


def test_symbolic_function_class_type_is_the_return_type_of_its_call():
    expression = Halved(variable_from([4]))
    assert expression._type_ is int
    assert expression._factory_ is Halved


def test_symbolic_function_class_still_binds_its_computed_value():
    assert Halved(variable_from([4, 7])).tolist() == [Halved(4)(), Halved(7)()]
