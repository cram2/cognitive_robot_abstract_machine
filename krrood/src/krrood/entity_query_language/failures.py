"""
This module defines some custom exception types used by the entity_query_language package.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Type, Any, List, Tuple, Optional

from ..utils import DataclassException

if TYPE_CHECKING:
    from .query.query import (
        Query,
    )
    from .query.operations import GroupedBy
    from .query.quantifiers import ResultQuantifier
    from .operators.aggregators import Aggregator
    from .query.builders import GroupedByBuilder
    from .core.base_expressions import SymbolicExpression, Selectable
    from .core.variable import Variable
    from .query.match import Match


@dataclass
class QuantificationNotSatisfiedError(DataclassException, ABC):
    """
    Represents a custom exception where the quantification constraints are not satisfied.

    This exception is used to indicate errors related to the quantification
    of the query results.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    expression: ResultQuantifier
    """
    The result quantifier expression where the error occurred.
    """
    expected_number: int
    """
    Expected number of solutions (i.e, quantification constraint value).
    """


@dataclass
class GreaterThanExpectedNumberOfSolutions(QuantificationNotSatisfiedError):
    """
    Represents an error when the number of solutions exceeds the
    expected threshold.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    def __post_init__(self):
        self.message = f"More than {self.expected_number} solutions found for the expression {self.expression}."
        super().__post_init__()


@dataclass
class LessThanExpectedNumberOfSolutions(QuantificationNotSatisfiedError):
    """
    Represents an error that occurs when the number of solutions found
    is lower than the expected number.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    found_number: int
    """
    The number of solutions found.
    """

    def __post_init__(self):
        self.message = (
            f"Found {self.found_number} solutions which is less than the expected {self.expected_number} "
            f"solutions for the expression {self.expression}."
        )
        super().__post_init__()


@dataclass
class MultipleSolutionFound(GreaterThanExpectedNumberOfSolutions):
    """
    Raised when a query unexpectedly yields more than one solution where a single
    result was expected.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    expected_number: int = 1


@dataclass
class NoSolutionFound(LessThanExpectedNumberOfSolutions):
    """
    Raised when a query does not yield any solution.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    expected_number: int = 1
    found_number: int = 0


@dataclass
class LogicalError(DataclassException):
    """
    Raised when there is an error in the logical structure/evaluation of the query.
    """


@dataclass
class VariableCannotBeEvaluated(DataclassException):
    """
    Raised when a variable cannot be evaluated due to missing or invalid information in the variable.
    """

    variable: Variable

    def __post_init__(self):
        self.message = (
            f"Variable {self.variable} cannot be evaluated because of missing or invalid information."
            f"The variable couldn't be identified as one of (already bound, has a domain, or is inferred,"
            f"Check that the variable is correctly defined and that all required information is provided."
        )
        super().__post_init__()


@dataclass
class UsageError(DataclassException):
    """
    Raised when there is an incorrect usage of the entity query language API.
    """

    ...


@dataclass
class TryingToModifyAnAlreadyBuiltQuery(UsageError):
    """
    Raised when trying to build an already built `Query`.

    Check how to write queries correctly in :doc:`/krrood/doc/eql/writing_queries`.
    """

    query: Query
    """
    The query that has already been built.
    """

    def __post_init__(self):
        self.message = f"{self.query} was already built."
        super().__post_init__()


@dataclass
class UnsupportedExpressionTypeForDistinct(UsageError):
    """
    Raised when an expression type is not supported for distinct operation.

    For further details, see the section on `distinct` and its usage in aggregations in :doc:`/krrood/doc/eql/result_processors`.
    """

    unsupported_expression_type: Type[SymbolicExpression]

    def __post_init__(self):
        self.message = f"Distinct operation is not supported for expression type {self.unsupported_expression_type}"
        super().__post_init__()


@dataclass
class NoConditionsProvided(UsageError):
    """
    Raised when no conditions are provided to the where/having statement of a query.

    For further details, see the section on writing queries and `where` clauses in :doc:`/krrood/doc/eql/writing_queries`.
    """

    query: Query
    """
    The query that has no conditions in its where/having statement.
    """

    def __post_init__(self):
        self.message = f"No conditions were provided to the where/having statement of the query {self.query}"
        super().__post_init__()


@dataclass
class NestedAggregationError(UsageError):
    """
    Raised when an aggregation is nested within another aggregation.

    For further details, see the "Features and Constraints" section regarding nested aggregations in :doc:`/krrood/doc/eql/result_processors`.
    """

    parent_aggregator: Aggregator
    """
    The parent aggregator.
    """

    def __post_init__(self):
        self.message = (
            f"Aggregator {self.parent_aggregator} has a child aggregator {self.parent_aggregator._child_}."
            f"Aggregations cannot be nested within another aggregation unless the inner aggregation is explicitly "
            f"grouped, E.g. eql.max(eql.count(...).grouped_by(...)) ), or wrapped in an entity query, "
            f"E.g. eql.max(entity(eql.count(...)))"
        )
        super().__post_init__()


@dataclass
class AggregationUsageError(UsageError):
    """
    Raised when there is an incorrect usage of aggregation in the entity query language API.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    query: Optional[Query] = field(default=None, kw_only=True)
    """
    The query that contains the aggregation.
    """


@dataclass
class UnsupportedAggregationOfAGroupedByVariable(AggregationUsageError):
    """
    Raised when there is an aggregation over a grouped_by variable that is not Count.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    grouped_by: GroupedBy
    """
    The grouped_by operation that contains the grouped_by variable that is being aggregated over.
    """

    def __post_init__(self):
        self.message = (
            f"Aggregation over grouped_by variable that is not Count "
            f"{self.grouped_by.aggregators_of_grouped_by_variables} in the grouped_by operation"
            f" {self.grouped_by}"
        )
        super().__post_init__()


@dataclass
class NonAggregatedSelectedVariablesError(AggregationUsageError):
    """
    Raised when a non-aggregated and not grouped_by variable(s) is selected along with an aggregated variable.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    grouped_by_builder: GroupedByBuilder
    """
    The builder class for the GroupedDataSource operation.
    """
    non_aggregated_variables: List[Selectable]
    """
    The non-aggregated selected variables.
    """
    aggregated_variables: List[Selectable]
    """
    The aggregated variables.
    """

    def __post_init__(self):
        self.message = (
            f"The variables {self.non_aggregated_variables} are neither aggregated nor grouped by, they cannot be selected"
            f" along with the aggregated variables {self.aggregated_variables}. You can only select variables that are"
            f" either aggregated or are in the grouped by variables {self.grouped_by_builder.variables_to_group_by}."
        )
        super().__post_init__()


@dataclass
class NonAggregatorInHavingConditionsError(AggregationUsageError):
    """
    Raised when a non-aggregator is used in a having condition.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    non_aggregators: Tuple[Selectable, ...]

    def __post_init__(self):
        self.message = f"The having condition of the query {self.query} contains non-aggregators {self.non_aggregators}."
        super().__post_init__()


@dataclass
class AggregatorInWhereConditionsError(AggregationUsageError):
    """
    Raised when an aggregator is used in a where condition.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    aggregators: Tuple[Aggregator, ...]
    """
    The aggregators in the where condition.
    """

    def __post_init__(self):
        self.message = (
            f"The where condition of the query {self.query} contains aggregators {self.aggregators}."
            f"If you want filter using aggregators, use `QueryObjectquery.having()` instead. Or wrap the aggregator"
            f"in a subquery e.g. `an(entity(...).where(entity(eql.count(...)) > n))`"
        )
        super().__post_init__()


@dataclass
class NoKwargsInMatchVar(UsageError):
    """
    Raised when a match_variable is used without any keyword arguments.

    For further details, see the notes on using `match_variable` vs `variable` in :doc:`/krrood/doc/eql/match`.
    """

    match_variable: Match

    def __post_init__(self):
        self.message = (
            f"The match variable {self.match_variable} was used without any keyword arguments."
            f"If you don't want to specify keyword arguments use variable() instead"
        )
        super().__post_init__()


@dataclass
class WrongSelectableType(UsageError):
    """
    Raised when a wrong variable type is given to the select() statement.

    For further details, see the sections on `entity()`, `set_of()`, and `variable()` in :doc:`/krrood/doc/eql/writing_queries`.
    """

    wrong_variable_type: Type
    expected_types: List[Type]

    def __post_init__(self):
        self.message = f"Select expects one of {self.expected_types}, instead {self.wrong_variable_type} was given."
        super().__post_init__()


@dataclass
class LiteralConditionError(UsageError):
    """
    Raised when a literal (i.e. a non-variable) condition is given to the query.
    Example:
        >>> a = True
        >>> body = let(Body, None)
        >>> query = an(entity(body, a))
    This could also happen when you are using a predicate or a symbolic_function and all the given arguments are literals.
    Example:
        >>> predicate = HasType(Body("Body1"), Body)
        >>> query = an(entity(let(Body, None), predicate))
    So make sure that at least one of the arguments to the predicate or symbolic function are variables.

    For further details, see the warning about literal conditions in :doc:`/krrood/doc/eql/writing_queries`.
    """

    query: Query
    """
    The query that contains the literal condition.
    """
    literal_conditions: List[Any]
    """
    The literal conditions that are given to the query.
    """

    def __post_init__(self):
        self.message = (
            f"The following Literal {self.literal_conditions} was given to the query {self.query}."
            f"Literal conditions are not allowed in queries, as they are always"
            f"either True or False, independent on any other values/bindings in the query"
        )
        super().__post_init__()


@dataclass
class CannotProcessResultOfGivenChildType(UsageError):
    """
    Raised when the entity query language API cannot process the results of a given child type during evaluation.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    unsupported_child_type: Type
    """
    The unsupported child type.
    """

    def __post_init__(self):
        self.message = (
            f"The child type {self.unsupported_child_type} cannot have its results processed"
            f" during evaluation because it doesn't implement the `_process_result_` method."
        )
        super().__post_init__()


@dataclass
class NonPositiveLimitValue(UsageError):
    """
    Raised when a limit value for the query results is not positive.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    wrong_limit_value: int

    def __post_init__(self):
        self.message = (
            f"Quantifier limit value must be a positive integer (i.e., greater than 0),"
            f" instead got {self.wrong_limit_value}"
        )
        super().__post_init__()


@dataclass
class UnsupportedOperation(UsageError):
    """
    Raised when an operation is not supported by the entity query language API.

    For further details, see :doc:`/krrood/doc/eql/logical_operators` and :doc:`/krrood/doc/eql/comparators`.
    """

    ...


@dataclass
class UnSupportedOperand(UnsupportedOperation):
    """
    Raised when an operand is not supported by the operation.

    For further details, see :doc:`/krrood/doc/eql/logical_operators` and :doc:`/krrood/doc/eql/comparators`.
    """

    operation: Type[SymbolicExpression]
    """
    The operation used.
    """
    unsupported_operand: Any
    """
    The operand that is not supported by the operation.
    """

    def __post_init__(self):
        self.message = f"{self.unsupported_operand} cannot be used as an operand for {self.operation} operations."
        super().__post_init__()


@dataclass
class UnsupportedNegation(UnsupportedOperation):
    """
    Raised when negating quantifiers.

    For further details, see the section on negation in :doc:`/krrood/doc/eql/logical_operators`.
    """

    operation_type: Type[SymbolicExpression]
    """
    The type of the operation that is being negated.
    """

    def __post_init__(self):
        self.message = (
            f"Symbolic NOT operations on {self.operation_type} types"
            f" operands are not allowed, you can negate the conditions instead,"
            f" as negating them is most likely not what you want"
            f" because it is ambiguous and can be very expensive to compute."
            f"To Negate Conditions do:"
            f" `not_(condition)` instead of `not_(an(entity(..., condition)))`."
        )
        super().__post_init__()


@dataclass
class QuantificationSpecificationError(UsageError):
    """
    Raised when the quantification constraints specified on the query results are invalid or inconsistent.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """


@dataclass
class QuantificationConsistencyError(QuantificationSpecificationError):
    """
    Raised when the quantification constraints specified on the query results are inconsistent.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    ...


@dataclass
class NegativeQuantificationError(QuantificationConsistencyError):
    """
    Raised when the quantification constraints specified on the query results have a negative value.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    message: str = f"ResultQuantificationConstraint must be a non-negative integer."


@dataclass
class InvalidChildType(UsageError):
    """
    Raised when an invalid entity type is given to the quantification operation.

    For further details, see :doc:`/krrood/doc/eql/writing_queries`.
    """

    invalid_child_type: Type
    """
    The invalid child type.
    """
    correct_child_types: List[Type]
    """
    The list of valid child types.
    """

    def __post_init__(self):
        self.message = f"The child type {self.invalid_child_type} is not valid. It must be a subclass of {self.correct_child_types}"
        super().__post_init__()


@dataclass
class NoExpressionFoundForGivenID(DataclassException):
    """
    Raised when no expression is found for the given expression ID.
    """

    symbolic_expression: SymbolicExpression
    """
    The current symbolic expression being evaluated.
    """
    expression_id: uuid.UUID
    """
    The ID of the expression that was not found.
    """

    def __post_init__(self):
        self.message = f"No expression found for ID: {self.expression_id} during evaluation of {self.symbolic_expression}."
        super().__post_init__()


@dataclass
class ClassDiagramError(DataclassException):
    """
    An error related to the class diagram.

    For further details, see :doc:`/krrood/doc/eql/domain_mapping`.
    """


@dataclass
class NoneWrappedFieldError(ClassDiagramError):
    """
    Raised when a field of a class is not wrapped by a WrappedField.

    For further details, see :doc:`/krrood/doc/eql/domain_mapping`.
    """

    clazz: Type
    attr_name: str

    def __post_init__(self):
        self.message = f"Field '{self.attr_name}' of class '{self.clazz.__name__}' is not wrapped by a WrappedField."
        super().__post_init__()


@dataclass
class NoChildToReplace(DataclassException):
    """
    Raised when trying to replace a child of an expression that has no children.
    """

    expression: SymbolicExpression
    """
    The expression that has no children.
    """
    old_child: SymbolicExpression
    """
    The child that was attempted to be replaced.
    """
    new_child: SymbolicExpression
    """
    The new child that was attempted to be set.
    """

    def __post_init__(self):
        self.message = f"Expression '{self.expression}' has no child '{self.old_child}' to replace with '{self.new_child}'."
        super().__post_init__()
