"""
Errors raised while building states and applying actions to them.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Any

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.ground_term import GroundTerm
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.planning.action_schema import ActionSchema

# %% facts


@dataclass(kw_only=True)
class ConditionIsNotAFact(DataclassException):
    """
    A condition that does not fix the value of a ground term was given as a fact.
    """

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


@dataclass(kw_only=True)
class ComparisonIsNotAnEquality(ConditionIsNotAFact):
    """
    A fact compares a ground term by other than equality, which constrains its value
    without fixing it.
    """

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


@dataclass(kw_only=True)
class MissingGroundTerm(DataclassException):
    """
    A fact gives a value to a ground term that does not exist, such as an attribute the
    instance does not have.
    """

    term: GroundTerm
    """
    The ground term that does not exist.
    """

    def error_message(self) -> str:
        return f"{self.term} has no value, so a fact cannot assign it."

    def suggest_correction(self) -> str:
        return "Assign only attributes the instance has and keys the container stores."


@dataclass(kw_only=True)
class AssignmentBypassesSetter(DataclassException):
    """
    A fact gives a value to a ground term that live objects set through code of their
    own class, such as a property setter, which a state never runs.
    """

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


@dataclass(kw_only=True)
class AssignedValueHasWrongType(DataclassException):
    """
    A fact or effect gives a ground term a value of another type than the class diagram
    declares for it.
    """

    term: GroundTerm
    """
    The ground term that was given the value.
    """

    value: Any
    """
    The value it was given.
    """

    def error_message(self) -> str:
        return (
            f"{self.term} was given {self.value!r} of type {type(self.value).__name__}, "
            "which its declared type does not admit."
        )

    def suggest_correction(self) -> str:
        return "Give the ground term a value of the type its field declares."


# %% effects


@dataclass(kw_only=True)
class ValueIsNotACollection(DataclassException):
    """
    An effect inserts an element into a ground term whose value is not a collection of
    elements.
    """

    collection: SymbolicExpression
    """
    The expression the effect inserts into.
    """

    value: Any
    """
    The value of the expression, which is not a collection.
    """

    def error_message(self) -> str:
        return (
            f"{self.collection} has the value {self.value!r}, which is not a "
            "collection an element can be inserted into."
        )

    def suggest_correction(self) -> str:
        return (
            "Insert only into lists, tuples and sets; assign any other value with an "
            "assignment instead."
        )


# %% actions


@dataclass(kw_only=True)
class ActionIsNotApplicable(DataclassException):
    """
    A ground action was applied in a state that does not satisfy its precondition.
    """

    action: ActionSchema
    """
    The action that was applied.
    """

    failed_conditions: list[SymbolicExpression]
    """
    The parts of the precondition that do not hold in the state.
    """

    def error_message(self) -> str:
        failed = ", ".join(str(condition) for condition in self.failed_conditions)
        return (
            f"The precondition of {self.action} does not hold in the state: {failed}."
        )

    def suggest_correction(self) -> str:
        return (
            "Apply only actions whose precondition the state satisfies, such as the "
            "results of applicable_actions evaluated in that state."
        )
