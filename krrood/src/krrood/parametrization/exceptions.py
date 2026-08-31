from dataclasses import dataclass

from typing_extensions import Any, List, Type

import random_events.variable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import ConditionType
from krrood.exceptions import DataclassException, InputError


@dataclass
class WhereExpressionIsFirstOrder(DataclassException):
    """
    Raised when a quantified `Where` expression is asked to be translated into a random
    event, since a product algebra is propositional and constrains a fixed set of
    variables instead of the objects a quantifier ranges over.
    """

    where_expression: ConditionType
    """
    The quantified expression that has no random event representation.
    """

    def error_message(self) -> str:
        return (
            f"The where expression {self.where_expression} quantifies over a variable, "
            f"which no fixed set of random event variables represents."
        )

    def suggest_correction(self) -> str:
        return (
            "State the condition over the attributes that are parameterized instead of "
            "quantifying, or evaluate the quantified condition on the query results."
        )


@dataclass
class WhereExpressionHasNoRandomEventRepresentation(DataclassException):
    """
    Raised when a part of a `Where` expression constrains something that no random event
    variable stands for, for example a comparison between two variables.
    """

    where_expression: ConditionType
    """
    The expression that has no random event representation.
    """

    def error_message(self) -> str:
        return (
            f"The where expression {self.where_expression} is neither a logical operator "
            f"nor a comparison between a variable and a literal, so it constrains no "
            f"random event variable."
        )

    def suggest_correction(self) -> str:
        return (
            "Compare a variable against a literal value, and combine such comparisons "
            "with and_, or_ and not_ only."
        )


@dataclass
class EmptyVariableDomain(InputError):
    variable: Variable

    def error_message(self) -> str:
        return f"The domain of the variable {self.variable} is empty. Domains must be non-empty for the variable to be valid."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidEllipsis(InputError):
    type_: Type

    def error_message(self) -> str:
        return f"Ellipsis is not allowed for type {self.type_}. Ellipsis are only allowed for the leaf objects (random events compatible types, see `random_events.variable.Variable.compatible_types`)."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class DoRequiresCausalCircuitModel(DataclassException):
    """
    Raised when a match has a ``cause`` intervention but the model registry resolved a
    model that is not a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.

    ``cause`` needs a registered causal graph to know what to cut when intervening; a
    plain (non-causal) probabilistic model has no such graph.
    """

    resolved_model: Any

    def error_message(self) -> str:
        return (
            f"A cause intervention was used, but the model registry returned "
            f"{type(self.resolved_model).__name__}, not a CausalCircuit. cause needs a "
            f"registered causal graph to know what to cut when intervening."
        )

    def suggest_correction(self) -> str:
        return "Use a model registry that returns a CausalCircuit for this class (see CausalCircuitRegistry)."


@dataclass
class MultipleEffectVariablesNotSupported(DataclassException):
    """
    Raised when a query declares more than one effect variable via
    ``causes_effect(...)``.

    :meth:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit.backdoor_adjustment`
    takes exactly one effect variable -- there is no multi-effect form of the
    interventional computation to route a query with several through. Multiple
    ``cause`` fields are fine: each candidate is searched independently and the one
    that best explains the effect becomes the primary cause.
    """

    variables: List[random_events.variable.Variable]
    """
    The declared effect variables.
    """

    def error_message(self) -> str:
        return (
            f"Found {len(self.variables)} effect variables "
            f"({[v.name for v in self.variables]}), but a causal search supports "
            f"exactly one."
        )

    def suggest_correction(self) -> str:
        return "Declare exactly one effect per query."
