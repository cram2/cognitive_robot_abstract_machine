from dataclasses import dataclass

from typing_extensions import Type

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.operations import Where
from krrood.exceptions import DataclassException, InputError


@dataclass
class WhereExpressionNotInDisjunctiveNormalForm(DataclassException):
    """
    Raised when a `Where` expression is not in disjunctive normal form.
    Check `is_disjunctive_normal_form` for more information and to see if the expression is in disjunctive normal form.
    """

    where_expression: ConditionType

    def __post_init__(self):
        self.message = f"The where expression {self.where_expression} is not in disjunctive normal form."


@dataclass
class EmptyVariableDomain(InputError):
    variable: Variable

    def __post_init__(self):
        self.message = f"The domain of the variable {self.variable} is empty. Domains must be non-empty for the variable to be valid."


@dataclass
class InvalidEllipsis(InputError):
    type_: Type

    def __post_init__(self):
        self.message = f"Ellipsis is not allowed for type {self.type_}. Ellipsis are only allowed for the leaf objects (random events compatible types, see `random_events.variable.Variable.compatible_types`)."
