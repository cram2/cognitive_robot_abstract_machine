"""
Tests of the check for field accessors written by hand.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from krrood.development_tools.agents.field_accessor_definition import (
    HandWrittenAccessor,
    HandWrittenFieldAccessorChecker,
)
from krrood.patterns import dataclass_field_descriptor

from .dataset_files import FieldAccessorSnippet
from .dataset_lines import HandWrittenAccessorsLine

# %% hand written field accessors


def test_a_hand_written_data_descriptor_and_a_dataclass_property_setter_are_reported() -> (
    None
):
    """
    A computed dataclass property and a plain class property may have a setter.
    """
    violations = HandWrittenFieldAccessorChecker().violations(
        path=FieldAccessorSnippet.HAND_WRITTEN_ACCESSORS
    )

    assert [
        (violation.first_line, violation.description) for violation in violations
    ] == [
        (
            HandWrittenAccessorsLine.DATA_DESCRIPTOR_SETTER,
            HandWrittenAccessor.DATA_DESCRIPTOR,
        ),
        (
            HandWrittenAccessorsLine.DATACLASS_PROPERTY_SETTER,
            HandWrittenAccessor.DATACLASS_PROPERTY_SETTER,
        ),
    ]


def test_the_field_descriptor_base_may_define_its_own_setter() -> None:
    """
    The base class is the one place that implements ``__set__``.
    """
    base_module = Path(inspect.getsourcefile(dataclass_field_descriptor))

    assert list(HandWrittenFieldAccessorChecker().violations(path=base_module)) == []
