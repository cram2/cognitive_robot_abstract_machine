"""
This module defines some custom exception types used by the entity_query_language
package.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Type, Any, List, Tuple, Optional

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.entity_query_language.backends import QueryBackend
    from krrood.entity_query_language.query.query import (
        Query,
    )
    from krrood.entity_query_language.query.operations import GroupedBy
    from krrood.entity_query_language.operators.aggregators import Aggregator
    from krrood.entity_query_language.query.builders import GroupedByBuilder
    from krrood.entity_query_language.core.base_expressions import (
        SymbolicExpression,
        Selectable,
    )
    from krrood.entity_query_language.core.mapped_variable import (
        Attribute,
        MappedVariable,
    )
    from krrood.entity_query_language.core.variable import Variable
    from krrood.entity_query_language.query.match import (
        Match,
        AbstractMatchExpression,
        AttributeMatch,
    )


@dataclass
class QuantificationNotSatisfiedError(DataclassException, ABC):
    """
    Represents a custom exception where the quantification constraints are not
    satisfied.

    This exception is used to indicate errors related to the quantification of the query
    results.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    expression: SymbolicExpression
    """
    The query expression whose result count violated the quantification constraint.
    """
    expected_number: int
    """
    Expected number of solutions (i.e, quantification constraint value).
    """


@dataclass
class GreaterThanExpectedNumberOfSolutions(QuantificationNotSatisfiedError):
    """
    Represents an error when the number of solutions exceeds the expected threshold.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    def error_message(self) -> str:
        return f"More than {self.expected_number} solutions found for the expression {self.expression}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class LessThanExpectedNumberOfSolutions(QuantificationNotSatisfiedError):
    """
    Represents an error that occurs when the number of solutions found is lower than the
    expected number.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    found_number: int
    """
    The number of solutions found.
    """

    def error_message(self) -> str:
        return (
            f"Found {self.found_number} solutions which is less than the expected {self.expected_number} "
            f"solutions for the expression {self.expression}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MultipleSolutionFound(GreaterThanExpectedNumberOfSolutions):
    """
    Raised when a query unexpectedly yields more than one solution where a single result
    was expected.

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
    Raised when a variable cannot be evaluated due to missing or invalid information in
    the variable.
    """

    variable: Variable

    def error_message(self) -> str:
        return (
            f"Variable {self.variable} cannot be evaluated because of missing or invalid information."
            f"The variable couldn't be identified as one of (already bound, has a domain, or is inferred,"
            f"Check that the variable is correctly defined and that all required information is provided."
        )

    def suggest_correction(self) -> str:
        return ""


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

    Check how to write queries correctly in
    :doc:`/krrood/doc/eql/writing_queries`.
    """

    query: Query
    """
    The query that has already been built.
    """

    def error_message(self) -> str:
        return f"{self.query} was already built."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class SymbolicDunderAccessError(AttributeError, UsageError):
    """
    Raised when a dunder attribute is accessed symbolically on a query variable.

    Subclasses :class:`AttributeError` so that ``copy``/``pickle`` and other machinery
    that probes optional dunder hooks via ``getattr(obj, "__hook__", default)`` still
    treats the access as a missing attribute instead of propagating an error.
    """

    attribute_name: str
    """
    The dunder attribute name that was accessed symbolically.
    """

    def error_message(self) -> str:
        return (
            f"The dunder attribute {self.attribute_name!r} cannot be accessed symbolically on a "
            f"query variable. Dunder (double-underscore) names are never treated as symbolic "
            f"attribute access: mapping them would let copy/pickle machinery recurse into endless "
            f"variable creation and blur the language semantics."
        )

    def suggest_correction(self) -> str:
        return (
            f"Perform the access inside a @symbolic_function that receives the concrete object, e.g. "
            f"`@symbolic_function` def get_value(obj): return obj.{self.attribute_name}, then call "
            f"get_value(variable) inside the query."
        )


@dataclass
class UnsupportedExpressionTypeForDistinct(UsageError):
    """
    Raised when an expression type is not supported for distinct operation.

    For further details, see the section on `distinct` and its usage in aggregations in
    :doc:`/krrood/doc/eql/result_processors`.
    """

    unsupported_expression_type: Type[SymbolicExpression]

    def error_message(self) -> str:
        return f"Distinct operation is not supported for expression type {self.unsupported_expression_type}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoConditionsProvided(UsageError):
    """
    Raised when no conditions are provided to the where/having statement of a query.

    For further details, see the section on writing queries and `where` clauses in
    :doc:`/krrood/doc/eql/writing_queries`.
    """

    query: Query
    """
    The query that has no conditions in its where/having statement.
    """

    def error_message(self) -> str:
        return f"No conditions were provided to the where/having statement of the query {self.query}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class AmbiguousQueryAttribute(UsageError):
    """
    Raised when a condition takes an attribute from a query that selects several
    variables, leaving the attribute without a single subject.

    For further details, see the section on writing queries and `where` clauses in
    :doc:`/krrood/doc/eql/writing_queries`.
    """

    query: Query
    """
    The query the attribute was taken from.
    """

    attribute: SymbolicExpression
    """
    The attribute chain rooted at that query.
    """

    def error_message(self) -> str:
        return (
            f"{self.attribute._name_} takes an attribute from the query {self.query}, which "
            f"selects {len(self.query._selected_variables_)} variables, so the attribute has no "
            f"single subject."
        )

    def suggest_correction(self) -> str:
        return (
            "Take the attribute from the variable it belongs to, e.g. `body.name` instead of "
            "`query.name`, or index the query by that variable, e.g. `query[body].name`."
        )


@dataclass
class NotNumberLikeFieldError(UsageError):
    """
    Raised when accessing a field expected to be number-like (see
    :meth:`~krrood.entity_query_language.core.mapped_variable.Attribute.number_like_field`),
    but it does not exist or resolves to a non-numeric type.
    """

    attribute: Attribute
    """
    The attribute that was accessed.
    """

    resolved_type: Optional[Type]
    """
    The attribute's resolved type, or None if it does not exist at all.
    """

    def error_message(self) -> str:
        if self.resolved_type is None:
            return f"{self.attribute} does not exist."
        return f"{self.attribute} is {self.resolved_type}, not number-like."

    def suggest_correction(self) -> str:
        return f"give the queried type a number-valued '{self.attribute._attribute_name_}' field."


@dataclass
class MultipleValuesAlongAccessPath(UsageError):
    """
    Raised when a chain is followed from a value outside query evaluation and a step maps
    that value to several, leaving the rest of the chain without one value to follow.
    """

    chain: MappedVariable
    """
    The chain that was being followed.
    """

    step: MappedVariable
    """
    The step along it that reaches more than one value.
    """

    def error_message(self) -> str:
        return (
            f"{self.chain._name_} passes through {self.step._name_}, which reaches one "
            f"value per element rather than a single one, so the rest of the access "
            f"path has no one value to follow."
        )

    def suggest_correction(self) -> str:
        return (
            "Follow a chain whose every step maps one value to one value, or aggregate "
            "the collection instead of flattening it."
        )


@dataclass
class UnselectedQueryVariable(UsageError):
    """
    Raised when a query over several variables is indexed by a variable it does not
    select, so the index names nothing in the rows the query yields.

    For further details, see the section on writing queries and `where` clauses in
    :doc:`/krrood/doc/eql/writing_queries`.
    """

    query: Query
    """
    The query that was indexed.
    """

    key: Any
    """
    What the query was indexed by.
    """

    def error_message(self) -> str:
        return (
            f"The query {self.query} was indexed by {self.key}, which is not one of the "
            f"variables it selects, so its rows hold nothing under that key."
        )

    def suggest_correction(self) -> str:
        selected = ", ".join(
            variable._name_ for variable in self.query._selected_variables_
        )
        return f"Index the query by one of the variables it selects: {selected}."


@dataclass
class ReadOnlyMapping(UsageError):
    """
    Raised when a value is written back through a chain whose step computes or picks its
    value instead of naming where that value is kept.
    """

    mapping: MappedVariable
    """
    The step the value would have been written through.
    """

    def error_message(self) -> str:
        return (
            f"{self.mapping._name_} does not name where its value is kept, so a value "
            f"cannot be written through it."
        )

    def suggest_correction(self) -> str:
        return (
            "Write through a step that names where the value is kept: an attribute, or "
            "an index by the key it is stored under."
        )


@dataclass
class NestedAggregationError(UsageError):
    """
    Raised when an aggregation is nested within another aggregation.

    For further details, see the "Features and Constraints" section
    regarding nested aggregations in
    :doc:`/krrood/doc/eql/result_processors`.
    """

    parent_aggregator: Aggregator
    """
    The parent aggregator.
    """

    def error_message(self) -> str:
        return (
            f"Aggregator {self.parent_aggregator} has a child aggregator {self.parent_aggregator._child_}."
            f"Aggregations cannot be nested within another aggregation unless the inner aggregation is explicitly "
            f"grouped, E.g. eql.max(eql.count(...).grouped_by(...)) ), or wrapped in an entity query, "
            f"E.g. eql.max(entity(eql.count(...)))"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class AggregationUsageError(UsageError):
    """
    Raised when there is an incorrect usage of aggregation in the entity query language
    API.

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
    The grouped_by operation that contains the grouped_by variable that is being
    aggregated over.
    """

    def error_message(self) -> str:
        return (
            f"Aggregation over grouped_by variable that is not Count "
            f"{self.grouped_by.aggregators_of_grouped_by_variables} in the grouped_by operation"
            f" {self.grouped_by}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NonAggregatedSelectedVariablesError(AggregationUsageError):
    """
    Raised when a non-aggregated and not grouped_by variable(s) is selected along with
    an aggregated variable.

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

    def error_message(self) -> str:
        return (
            f"The variables {self.non_aggregated_variables} are neither aggregated nor grouped by, they cannot be selected"
            f" along with the aggregated variables {self.aggregated_variables}. You can only select variables that are"
            f" either aggregated or are in the grouped by variables {self.grouped_by_builder.variables_to_group_by}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NonAggregatorInHavingConditionsError(AggregationUsageError):
    """
    Raised when a non-aggregator is used in a having condition.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    non_aggregators: Tuple[Selectable, ...]

    def error_message(self) -> str:
        return f"The having condition of the query {self.query} contains non-aggregators {self.non_aggregators}."

    def suggest_correction(self) -> str:
        return ""


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

    def error_message(self) -> str:
        return f"The where condition of the query {self.query} contains aggregators {self.aggregators}."

    def suggest_correction(self) -> str:
        return (
            "if you want to filter using aggregators, use `QueryObjectquery.having()` instead, or wrap the "
            "aggregator in a subquery e.g. `an(entity(...).where(entity(eql.count(...)) > n))`."
        )


@dataclass
class WrongSelectableType(UsageError):
    """
    Raised when a wrong variable type is given to the select() statement.

    For further details, see the sections on `entity()`, `set_of()`, and `variable()` in
    :doc:`/krrood/doc/eql/writing_queries`.
    """

    wrong_variable_type: Type
    expected_types: List[Type]

    def error_message(self) -> str:
        return f"Select expects one of {self.expected_types}, instead {self.wrong_variable_type} was given."

    def suggest_correction(self) -> str:
        return ""


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

    def error_message(self) -> str:
        return (
            f"The following Literal {self.literal_conditions} was given to the query {self.query}."
            f"Literal conditions are not allowed in queries, as they are always"
            f"either True or False, independent on any other values/bindings in the query"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class CannotProcessResultOfGivenChildType(UsageError):
    """
    Raised when the entity query language API cannot process the results of a given
    child type during evaluation.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    unsupported_child_type: Type
    """
    The unsupported child type.
    """

    def error_message(self) -> str:
        return (
            f"The child type {self.unsupported_child_type} cannot have its results processed"
            f" during evaluation because it doesn't implement the `_process_result_` method."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NonPositiveLimitValue(UsageError):
    """
    Raised when a limit value for the query results is not positive.

    For further details, see :doc:`/krrood/doc/eql/result_processors`.
    """

    wrong_limit_value: int

    def error_message(self) -> str:
        return (
            f"Quantifier limit value must be a positive integer (i.e., greater than 0),"
            f" instead got {self.wrong_limit_value}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnsupportedOperation(UsageError):
    """
    Raised when an operation is not supported by the entity query language API.

    For further details, see :doc:`/krrood/doc/eql/logical_operators` and
    :doc:`/krrood/doc/eql/comparators`.
    """

    ...


@dataclass
class UnSupportedOperand(UnsupportedOperation):
    """
    Raised when an operand is not supported by the operation.

    For further details, see :doc:`/krrood/doc/eql/logical_operators` and
    :doc:`/krrood/doc/eql/comparators`.
    """

    operation: Type[SymbolicExpression]
    """
    The operation used.
    """

    unsupported_operand: Any
    """
    The operand that is not supported by the operation.
    """

    def error_message(self) -> str:
        return f"{self.unsupported_operand} cannot be used as an operand for {self.operation} operations."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnsupportedNegation(UnsupportedOperation):
    """
    Raised when negating quantifiers.

    For further details, see the section on negation in
    :doc:`/krrood/doc/eql/logical_operators`.
    """

    operation_type: Type[SymbolicExpression]
    """
    The type of the operation that is being negated.
    """

    def error_message(self) -> str:
        return (
            f"Symbolic NOT operations on {self.operation_type} types"
            f" operands are not allowed, as negating them is most likely not what you want"
            f" because it is ambiguous and can be very expensive to compute."
        )

    def suggest_correction(self) -> str:
        return "negate the conditions instead: `not_(condition)` instead of `not_(an(entity(..., condition)))`."


@dataclass
class QuantificationSpecificationError(UsageError):
    """
    Raised when the quantification constraints specified on the query results are
    invalid or inconsistent.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """


@dataclass
class QuantificationConsistencyError(QuantificationSpecificationError):
    """
    Raised when the quantification constraints specified on the query results are
    inconsistent.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    ...


@dataclass
class InvalidQuantificationRangeError(QuantificationConsistencyError):
    """
    Raised when the upper quantification bound is smaller than the lower bound.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    at_least: Any
    """
    The lower bound of the quantification range.
    """

    at_most: Any
    """
    The upper bound of the quantification range.
    """

    def error_message(self) -> str:
        return f"at_most {self.at_most} cannot be less than at_least {self.at_least}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NegativeQuantificationError(QuantificationConsistencyError):
    """
    Raised when the quantification constraints specified on the query results have a
    negative value.

    For further details, see :doc:`/krrood/doc/eql/result_quantifiers`.
    """

    def error_message(self) -> str:
        return "ResultQuantificationConstraint must be a non-negative integer."

    def suggest_correction(self) -> str:
        return ""


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

    def error_message(self) -> str:
        return f"The child type {self.invalid_child_type} is not valid. It must be a subclass of {self.correct_child_types}"

    def suggest_correction(self) -> str:
        return ""


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

    def error_message(self) -> str:
        return f"No expression found for ID: {self.expression_id} during evaluation of {self.symbolic_expression}."

    def suggest_correction(self) -> str:
        return ""


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

    def error_message(self) -> str:
        return f"Field '{self.attr_name}' of class '{self.clazz.__name__}' is not wrapped by a WrappedField."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class SelfReferentialInsertionError(DataclassException):
    """
    Raised when insert_at would create a self-referential selector node.
    """

    anchor: SymbolicExpression = field(kw_only=True)
    """
    The existing rule-tree node that the new condition would have been spliced onto.
    """

    def error_message(self) -> str:
        return (
            f"The new condition is the same node as the anchor {self.anchor!r} — "
            "this would create a self-referential Refinement/Alternative and corrupt "
            "the anchor's conclusions."
        )

    def suggest_correction(self) -> str:
        return "Provide a condition that is a different node than the anchor."


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

    def error_message(self) -> str:
        return f"Expression '{self.expression}' has no child '{self.old_child}' to replace with '{self.new_child}'."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class GenerativeBackendQueryIsNotUnderspecifiedVariable(DataclassException):
    """
    Exception raised when a query is not a match inside a generative backend.
    """

    expression: Query
    """
    The query that was passed to the generative backend.
    """

    def error_message(self) -> str:
        return f"Query {self.expression} is not an underspecified variable inside a generative backend."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class SelectiveBackendCannotResolveEllipsisMatch(DataclassException):
    """
    Exception raised when a match with an ``...`` (Ellipsis) attribute is evaluated with
    a selective backend.
    """

    match: Match
    """
    The match that has an Ellipsis attribute.
    """

    def error_message(self) -> str:
        return (
            f"{self.match} has an Ellipsis (...) attribute, so it cannot be resolved by a "
            f"selective backend: selecting only finds existing instances, it cannot fill in an "
            f"attribute left unspecified."
        )

    def suggest_correction(self) -> str:
        return "Evaluate with a GenerativeBackend (or ProbabilisticBackend) instead."


@dataclass
class BackendCannotEvaluateCause(DataclassException):
    """
    Raised when a match with a :class:`~krrood.entity_query_language.operators.causal.Cause`
    (``cause``) intervention is evaluated with a backend that has no notion of a
    causal graph to search over, and that backend was configured (via
    ``raise_on_unresolvable_cause=True``) to fail loudly instead of warning and treating
    the intervention as an ordinary unspecified field.
    """

    match: Match
    """
    The match that has a ``Cause`` attribute.
    """

    backend_type: Type[QueryBackend]
    """
    The type of the backend that cannot evaluate the intervention causally.
    """

    def error_message(self) -> str:
        return (
            f"{self.match} contains a cause intervention, which {self.backend_type.__name__} "
            f"cannot evaluate causally: it has no notion of a causal graph to intervene on."
        )

    def suggest_correction(self) -> str:
        return "Evaluate with a ProbabilisticBackend backed by a CausalCircuit-aware model registry."


@dataclass
class CalledMatchMultipleTimes(DataclassException):
    """
    Exception raised when a match expression is called multiple times.
    """

    match: AbstractMatchExpression

    def error_message(self) -> str:
        return (
            f"Match expression '{self.match}' was called multiple times. "
            f"Match acts like a constructor and hence should not be called multiple times. "
            f"Invoking the `__call__` method multiple times has unexpected side effects."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration(
    DataclassException
):
    attribute_match: AttributeMatch

    def error_message(self) -> str:
        return (
            "If you want to use EQL to generate answers, "
            f"assignments in underspecified queries must be concrete objects or a symbolic expression. "
            f"If the assignment is Ellipsis, the type of the field must be an Enum, otherwise EQL can't "
            f"generate it. "
            f"Got {self.attribute_match.name_from_variable_access_path} = {self.attribute_match.assigned_variable._type_}."
        )

    def suggest_correction(self) -> str:
        return (
            "if you're looking for more flexible generations, try ProbabilisticBackend."
        )


@dataclass
class CausesEffectRequiresEqualityComparator(UsageError):
    """
    Raised when a :func:`~krrood.entity_query_language.query.match.Match.causes_effect`
    condition is not an equality comparator (or a conjunction of equality comparators).

    A causal effect must be expressed as ``attribute == value`` (or several such
    comparisons ANDed together), the same restriction Pearl's atomic point-intervention
    ``do(X=x)`` already implies: you can ask what causes an attribute to equal a value,
    not what causes it to satisfy an inequality or an arbitrary relation to another
    attribute.
    """

    condition: SymbolicExpression
    """
    The condition that is not an equality comparator or conjunction thereof.
    """

    def error_message(self) -> str:
        return (
            f"causes_effect(...) requires an equality comparator (attribute == value) "
            f"or a conjunction of equality comparators, got {self.condition}."
        )

    def suggest_correction(self) -> str:
        return (
            "Compare an attribute against a literal value with `==`, e.g. "
            "`match.causes_effect(match.variable.status == SUCCESS)`, combining "
            "several such comparisons with `and_` if needed."
        )


@dataclass
class NoCausesEffectConditionForCause(DataclassException):
    """
    Raised when a :class:`~krrood.entity_query_language.operators.causal.Cause` (``cause``)
    is present in a match but no
    :meth:`~krrood.entity_query_language.query.match.Match.causes_effect` condition
    declares which variable it should optimize for.
    """

    expression: Query
    """
    The query that has a ``Cause`` but no declared effect.
    """

    def error_message(self) -> str:
        return (
            f"{self.expression} has a cause intervention but no causes_effect(...) "
            f"condition, so there is nothing to search for the best intervention region "
            f"against."
        )

    def suggest_correction(self) -> str:
        return (
            "Add a causes_effect(...) condition declaring the effect, e.g. "
            "`match.causes_effect(match.variable.status == SUCCESS)`."
        )


@dataclass
class NoCauseVariablesForRanking(DataclassException):
    """
    Raised when
    :meth:`~krrood.entity_query_language.backends.ProbabilisticBackend.rank_causes` is
    called on a match with no :class:`~krrood.entity_query_language.operators.causal.Cause`
    (``cause``) fields to rank.
    """

    expression: Query
    """
    The query that has no ``Cause`` fields.
    """

    def error_message(self) -> str:
        return f"{self.expression} has no cause fields, so there is nothing to rank."

    def suggest_correction(self) -> str:
        return "Mark at least one field with cause before calling rank_causes()."


@dataclass
class MatchTypeCannotBeDetermined(DataclassException):
    """
    Raised when a match fails at inferring its type.
    """

    match: Match
    """
    The match that failed to infer its type.
    """

    def error_message(self) -> str:
        return (
            f"Match type cannot be determined for {self.match}. "
            f"Tried to infer the type from {self.match.factory}."
            f"The factory given to the match must ether be a classmethod the returns its class or a "
            f"method where the return type is a class which has been concretely imported (not via "
            f"TYPE_CHECKING). If that is not an option for you, set the `target_type` keyword "
            f"argument of `an`/`the`."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ModelingError(DataclassException):
    """
    Exception raised when there's an error in the model (classes, functions, etc.)
    definition.
    """


@dataclass
class WrongPropertyReturnStatementImplementation(ModelingError):
    """
    Exception raised when the implementation of a return statement of a property of a
    class is wrong.
    """

    property_object: property
    """
    The property that is wrongly implemented.
    """

    reason: str
    """
    The reason for the wrong property.
    """

    clazz: Optional[Type] = None
    """
    The class that has the property.
    """

    def error_message(self) -> str:
        clazz = self.clazz if self.clazz is not None else "UNKNOWN_CLASS"
        return (
            f"The implementation of the property {self.property_object} of the class {clazz} is wrong, "
            f"the reason is: {self.reason}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoReturnStatementInProperty(ModelingError):
    """
    Exception raised when the implementation of a property has no return statement.
    """

    property_object: property
    """
    The property that is wrongly implemented.
    """

    clazz: Optional[Type] = None
    """
    The class that has the property.
    """

    def error_message(self) -> str:
        clazz = self.clazz if self.clazz is not None else "UNKNOWN_CLASS"
        return f"The implementation of the property {self.property_object} of the class {clazz} has no return statement"

    def suggest_correction(self) -> str:
        return ""
