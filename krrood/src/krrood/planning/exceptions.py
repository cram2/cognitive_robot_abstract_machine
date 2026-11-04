"""Errors raised while building states."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.entity_query_language.ground_term import GroundTerm
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.exceptions import DataclassException

# %% facts


@dataclass
class ConditionIsNotAFact(DataclassException):
    """A condition that does not fix the value of a ground term was given as a
    fact."""

    condition: Any
    """
    The condition that was given.
    """

    def error_message(self) -> str:
        return f"{self.condition} does not fix the value of a ground term."

    def suggest_correction(self) -> str:
        return (
            "State a fact as 'expression == value' for an attribute, index or call, as a "
            "predicate for it to hold, or as not_(predicate) for it not to hold."
        )


@dataclass
class ComparisonIsNotAnEquality(ConditionIsNotAFact):
    """A fact compares a ground term by other than equality, which constrains
    its value without fixing it."""

    condition: Comparator
    """
    The comparison that was given.
    """

    def error_message(self) -> str:
        return (
            f"{self.condition} compares with '{self.condition._name_}', which does not "
            "fix the value of the ground term; only '==' does."
        )

    def suggest_correction(self) -> str:
        return (
            "Compare the ground term with '==' to give it a value. To state that the "
            "comparison holds, write it as the body of a predicate and state the "
            "predicate: that assigns the predicate itself, not the values it compares."
        )


@dataclass
class FactValueIsNotConstant(DataclassException):
    """A fact compares a ground term with an expression instead of with a
    value."""

    condition: Any
    """
    The fact whose value is an expression.
    """

    def error_message(self) -> str:
        return f"The value {self.condition} gives its ground term is not a constant."

    def suggest_correction(self) -> str:
        return "Compare the ground term with a plain value."


@dataclass
class MissingGroundTerm(DataclassException):
    """A fact gives a value to a ground term that does not exist, such as an
    attribute the instance does not have."""

    term: GroundTerm
    """
    The ground term that does not exist.
    """

    def error_message(self) -> str:
        return f"{self.term} has no value, so a fact cannot assign it."

    def suggest_correction(self) -> str:
        return "Assign only attributes the instance has and keys the container stores."


@dataclass
class AssignmentBypassesSetter(DataclassException):
    """A fact gives a value to a ground term that live objects set through code
    of their own class, such as a property setter, which a state never runs."""

    term: GroundTerm
    """
    The ground term whose setting runs code.
    """

    def error_message(self) -> str:
        return (
            f"Setting {self.term} on a live object runs code of its class, which may "
            "change more than this ground term; a state would only store the value."
        )

    def suggest_correction(self) -> str:
        return (
            "Assign the plain fields the setter stores, and state what else it changes "
            "as derived ground terms (EQL rules or symbolic functions) or as further facts."
        )
