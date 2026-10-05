import inspect

import pytest

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.operators.causal import cause
from krrood.patterns.exceptions import KeywordNamesNoFactoryParameter
from krrood.patterns import factory_and_kwargs
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs

from ..dataset.derived_attributes import Rectangle
from ..dataset.example_classes import KRROODPosition


def record_keywords(**keywords: float) -> dict[str, float]:
    """
    A factory that accepts arbitrary keywords.

    :return: The keywords it was called with.
    """
    return keywords


# %% keywords naming no parameter of the factory


def test_keyword_naming_no_parameter_is_refused():
    factory_and_kwargs = HasFactoryAndKwargs(
        KRROODPosition, _kwargs_={"x": 1.0, "y": 2.0, "z": 3.0, "w": 4.0}
    )
    with pytest.raises(KeywordNamesNoFactoryParameter) as error:
        factory_and_kwargs.construct_instance()
    assert error.value.factory is KRROODPosition
    assert error.value.keyword == "w"


def test_keyword_naming_no_parameter_is_refused_as_a_call_with_an_unexpected_keyword():
    """
    Python refuses an unexpected keyword argument with a :class:`TypeError`, so a caller
    handling that refusal handles this one too.
    """
    factory_and_kwargs = HasFactoryAndKwargs(KRROODPosition, _kwargs_={"w": 4.0})
    with pytest.raises(TypeError):
        factory_and_kwargs.construct_instance()


def test_factory_accepting_arbitrary_keywords_receives_every_keyword():
    factory_and_kwargs = HasFactoryAndKwargs(record_keywords, _kwargs_={"w": 4.0})
    assert factory_and_kwargs.construct_instance() == {"w": 4.0}


def test_misspelled_match_keyword_is_refused_at_construction():
    position = a(KRROODPosition)(x=1.0, y=2.0, zz=3.0)
    with pytest.raises(KeywordNamesNoFactoryParameter) as error:
        position.construct_instance()
    assert error.value.keyword == "zz"


def test_match_keyword_marked_as_a_causal_role_is_left_out_of_construction():
    position = a(KRROODPosition)(x=1.0, y=2.0, z=3.0, distance_to_origin=cause)
    assert position.construct_instance() == KRROODPosition(1.0, 2.0, 3.0)


def test_match_keyword_naming_a_property_of_the_matched_type_is_left_out_of_construction():
    rectangle = a(Rectangle)(width=3.0, height=4.0, area=12.0)
    assert rectangle.construct_instance() == Rectangle(3.0, 4.0)


def test_match_keyword_naming_a_method_of_the_matched_type_is_left_out_of_construction():
    rectangle = a(Rectangle)(width=3.0, height=4.0, perimeter=...)
    assert rectangle.construct_instance() == Rectangle(3.0, 4.0)


def test_match_keyword_naming_a_constructor_parameter_that_is_also_a_class_attribute_is_constructed_with():
    rectangle = a(Rectangle)(width=3.0, height=4.0, layer=2)
    assert rectangle.construct_instance() == Rectangle(3.0, 4.0, layer=2)


# %% deciding which keywords are constructed with


def test_repeated_construction_reads_the_factory_signature_once(monkeypatch):
    def make_rectangle(width: float, height: float) -> Rectangle:
        return Rectangle(width, height)

    signature_reads = []
    read_signature = inspect.signature

    def counting_signature(callable_):
        signature_reads.append(callable_)
        return read_signature(callable_)

    monkeypatch.setattr(factory_and_kwargs.inspect, "signature", counting_signature)
    rectangle = a(make_rectangle)(width=3.0, height=4.0, area=12.0)
    rectangle.construct_instance()
    rectangle.construct_instance()
    assert signature_reads == [make_rectangle]
