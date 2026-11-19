"""
Tests of properties read on the class.
"""

from __future__ import annotations

from ..dataset.abstract_classproperties import (
    AbstractDescription,
    ImplementedDescription,
)

# %% abstract classproperties


def test_a_classproperty_wrapping_an_abstract_method_is_abstract() -> None:
    """
    An abstract base class records an abstract classproperty, as it does a property.
    """
    assert AbstractDescription.__abstractmethods__ == frozenset({"describe"})


def test_a_class_implementing_an_abstract_classproperty_is_not_abstract() -> None:
    """
    Implementing the classproperty removes it from the abstract methods.
    """
    assert ImplementedDescription.__abstractmethods__ == frozenset()
