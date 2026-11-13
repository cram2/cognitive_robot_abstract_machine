"""
Tests for the value-emitter family of :mod:`krrood.code_generation.source_emission`.
"""

from __future__ import annotations

from enum import Enum, IntEnum

import pytest

from krrood.code_generation.ast_helpers import ast_unparse_expression
from krrood.code_generation.exceptions import ObjectNotSerializableError
from krrood.code_generation.source_emission import (
    BoolSourceEmitter,
    EnumSourceEmitter,
    FloatSourceEmitter,
    IntSourceEmitter,
    SourceEmissionContext,
    StrSourceEmitter,
    ValueSourceEmitter,
)
from test.krrood_test.dataset.nested_dataclass_records import (
    Animal,
    Owner,
    Species,
    rex,
)


class Colour(Enum):
    """
    An enum whose members the emitted expressions reference.
    """

    RED = "red"


class Priority(IntEnum):
    """
    An enum whose members are also integers.
    """

    HIGH = 1


# %% scalars and enum members


@pytest.mark.parametrize(
    "value", [Colour.RED, Priority.HIGH, True, 3, 2.5, "text", None]
)
def test_emitted_value_evaluates_back_to_the_value(value):
    """
    The expression evaluates to the value it was built from, keeping an enum member a
    member even when it is also an integer.
    """
    context = SourceEmissionContext()

    source = ast_unparse_expression(
        ValueSourceEmitter.emit_source(subject=value, context=context)
    )

    rebuilt = eval(source, {Colour.__name__: Colour, Priority.__name__: Priority})
    assert rebuilt == value
    assert type(rebuilt) is type(value)


def test_emitting_an_enum_member_references_its_enum_type():
    """
    An enum member's type has to be imported by the generated module.
    """
    context = SourceEmissionContext()

    ValueSourceEmitter.emit_source(subject=Colour.RED, context=context)

    assert context.referenced_types == {Colour}


def test_supported_scalar_types_are_the_types_the_scalar_emitters_bind():
    """
    The supported scalar types follow the emitters of the family.
    """
    assert set(ValueSourceEmitter.supported_scalar_types) == {
        emitter.emitted_type
        for emitter in (
            BoolSourceEmitter,
            IntSourceEmitter,
            FloatSourceEmitter,
            StrSourceEmitter,
            EnumSourceEmitter,
        )
    }


# %% dataclass instances


def test_emitted_dataclass_rebuilds_an_equal_instance_and_references_its_types():
    """
    A nested dataclass is emitted as constructor calls of every type it holds.
    """
    context = SourceEmissionContext()

    source = ast_unparse_expression(
        ValueSourceEmitter.emit_source(subject=rex(), context=context)
    )

    record_types = {Animal, Species, Owner}
    assert (
        eval(
            source, {record_type.__name__: record_type for record_type in record_types}
        )
        == rex()
    )
    assert context.referenced_types == record_types


def test_emitting_a_value_no_emitter_applies_to_raises():
    """
    A value of a type outside the family has no emitted form.
    """
    with pytest.raises(ObjectNotSerializableError) as error:
        ValueSourceEmitter.emit_source(subject=[1], context=SourceEmissionContext())
    assert error.value.value == [1]


def test_value_emitter_belongs_to_the_value_family():
    """
    An emitter several levels below the family base still belongs to it.
    """
    assert IntSourceEmitter.family is ValueSourceEmitter
