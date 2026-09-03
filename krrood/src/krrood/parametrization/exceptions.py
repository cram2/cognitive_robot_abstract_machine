from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, List, Set, TYPE_CHECKING, Type

import random_events.variable
from krrood.entity_query_language.core.variable import Variable
from krrood.exceptions import DataclassException, InputError

if TYPE_CHECKING:
    from krrood.entity_query_language.factories import ConditionType


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


@dataclass
class JointQueryAcrossClassesNotSupported(DataclassException):
    """
    Raised when a probabilistic query (``probability_of(...)``, ``average(...)``)
    doesn't reference attributes of exactly one EQL class -- either none at all, e.g.
    ``probability_of(True)`` (a content-free condition names no class, so there is no
    model to resolve it against), or more than one, e.g. ``average(x.A)`` combined
    with attributes of a second class in the same call, for
    ``x = variable(ClassOne)`` and ``y = variable(ClassTwo)``.

    ``distribution_of(...)`` never raises this: it wraps a single ``Match``, which is
    always for one class by construction.

    Every :class:`~krrood.parametrization.model_registries.ModelRegistry` resolves a
    single model per class, so there is no established way to ground a query
    referencing zero, or more than one, classes' models yet.
    """

    owner_classes: Set[Type]
    """
    The distinct owner classes found among the query's referenced attributes -- empty
    when it referenced none at all.
    """

    def error_message(self) -> str:
        if not self.owner_classes:
            return (
                "The query referenced no class-bound attributes at all, so there is "
                "no model to resolve it against."
            )
        names = ", ".join(sorted(cls.__name__ for cls in self.owner_classes))
        return (
            f"The query referenced attributes owned by {len(self.owner_classes)} "
            f"different classes ({names}), but only a single-class query is "
            f"supported."
        )

    def suggest_correction(self) -> str:
        return (
            "Reference attributes reached from a single variable(...) root, or split "
            "into separate queries."
        )


@dataclass
class RelationalCircuitRegistryRequiresMatch(DataclassException):
    """
    Raised when a :class:`~krrood.parametrization.model_registries.RelationalCircuitRegistry`
    is asked to resolve a model for parameters that aren't a
    :class:`~krrood.parametrization.parameterizer.UnderspecifiedParameters` (i.e. not a
    ``Match``, directly or wrapped by ``distribution_of(...)``) -- ``probability_of(...)``
    and a bare ``average(...)`` build the lighter
    :class:`~krrood.parametrization.parameterizer.ConditionParameters`/
    :class:`~krrood.parametrization.parameterizer.SelectedAttributesParameters`
    instead, which carry no match statement to ground.
    """

    parameters: Any
    """
    The parameters that were given instead of an ``UnderspecifiedParameters``.
    """

    def error_message(self) -> str:
        return (
            f"RelationalCircuitRegistry needs a Match's full statement to ground its "
            f"circuit, but got {type(self.parameters).__name__}, which carries none."
        )

    def suggest_correction(self) -> str:
        return (
            "Use RelationalCircuitRegistry only with distribution_of(...) (or a bare "
            "Match), not probability_of(...)/average(...)."
        )
