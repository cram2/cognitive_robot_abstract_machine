"""
Core symbolic expression system used to build and evaluate entity queries.

This module defines the symbolic types (variables, sources, logical and
comparison operators) and the evaluation mechanics.
"""

from __future__ import annotations

import operator
import typing
import uuid
from abc import abstractmethod, ABC
from collections import UserDict, defaultdict
from copy import copy
from dataclasses import dataclass, field, fields, MISSING, is_dataclass
from functools import lru_cache, cached_property, wraps

from typing_extensions import (
    Iterable,
    Any,
    Optional,
    Type,
    Dict,
    ClassVar,
    Union as TypingUnion,
    TYPE_CHECKING,
    List,
    Tuple,
    Callable,
    Self,
    Set,
    Iterator,
    Generic,
    Collection,
)

from .cache_data import (
    SeenSet,
    ReEnterableLazyIterable,
)
from .enums import PredicateType
from .failures import (
    MultipleSolutionFound,
    NoSolutionFound,
    UnsupportedNegation,
    GreaterThanExpectedNumberOfSolutions,
    LessThanExpectedNumberOfSolutions,
    UnSupportedOperand,
    NonPositiveLimitValue,
    InvalidChildType,
    LiteralConditionError,
    NonAggregatedSelectedVariablesError,
    NoConditionsProvided,
    AggregatorInWhereConditionsError,
    NonAggregatorInHavingConditionsError,
    UnsupportedAggregationOfAGroupedByVariable,
    NestedAggregationError,
    TryingToModifyAnAlreadyBuiltQuery,
    NoExpressionFoundForGivenID,
)
from .failures import VariableCannotBeEvaluated
from .result_quantification_constraint import (
    ResultQuantificationConstraint,
    Exactly,
)
from .symbol_graph import SymbolGraph
from .utils import (
    is_iterable,
    generate_combinations,
    make_list,
    make_set,
    T,
    merge_args_and_kwargs,
    convert_args_and_kwargs_into_a_hashable_key,
    ensure_hashable,
    chain_evaluate_variables,
)
from ..class_diagrams.class_diagram import WrappedClass
from ..class_diagrams.failures import ClassIsUnMappedInClassDiagram
from ..class_diagrams.wrapped_field import WrappedField

if TYPE_CHECKING:
    from .conclusion import Conclusion
    from .entity import ConditionType

Bindings = Dict[int, Any]
"""
A dictionary for variable bindings in EQL operations
"""

GroupKey = Tuple[Any, ...]
"""
A tuple representing values of variables that are used in the grouped_by clause.
"""


@dataclass
class OperationResult:
    """
    A data structure that carries information about the result of an operation in EQL.
    """

    bindings: Bindings
    """
    The bindings resulting from the operation, mapping variable IDs to their values.
    """
    is_false: bool
    """
    Whether the operation resulted in a false value (i.e., The operation condition was not satisfied)
    """
    operand: SymbolicExpression
    """
    The operand that produced the result.
    """

    @cached_property
    def has_value(self) -> bool:
        return self.operand._binding_id_ in self.bindings

    @cached_property
    def is_true(self) -> bool:
        return not self.is_false

    @property
    def value(self) -> Any:
        """
        The value of the operation result, retrieved from the bindings using the operand's ID.

        :raises: KeyError if the operand is not found in the bindings.
        """
        return self.bindings[self.operand._binding_id_]

    def __contains__(self, item):
        return item in self.bindings

    def __getitem__(self, item):
        return self.bindings[item]

    def __setitem__(self, key, value):
        self.bindings[key] = value

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return (
            self.bindings == other.bindings
            and self.is_true == other.is_true
            and self.operand == other.operand
        )


@dataclass(eq=False)
class SymbolicExpression(ABC):
    """
    Base class for all symbolic expressions.

    Symbolic expressions form a rooted directed acyclic graph (rooted DAG) and are evaluated lazily to produce
    bindings for variables, subject to logical constraints.
    """

    _id_: int = field(init=False, repr=False, default_factory=lambda: uuid.uuid4().int)
    """
    Unique identifier of this node.
    """
    _conclusion_: typing.Set[Conclusion] = field(init=False, default_factory=set)
    """
    Set of conclusion expressions attached to this node, these are evaluated when the truth value of this node is true
    during evaluation.
    """
    _symbolic_expression_stack_: ClassVar[List[SymbolicExpression]] = []
    """
    The current stack of symbolic expressions that has been entered using the ``with`` statement.
    """
    _is_false__: bool = field(init=False, repr=False, default=False)
    """
    Internal flag indicating current truth value of evaluation result for this expression.
    """
    _children_: List[SymbolicExpression] = field(
        init=False, repr=False, default_factory=list
    )
    """
    The children expressions of this symbolic expression.
    """
    _parents_: List[SymbolicExpression] = field(
        init=False, repr=False, default_factory=list
    )
    """
    The parents expressions of this symbolic expression.
    """
    _parent__: Optional[SymbolicExpression] = field(
        init=False, repr=False, default=None
    )
    """
    Internal attribute used to track the parent symbolic expression of this expression.
    """
    _eval_parent_: Optional[SymbolicExpression] = field(
        default=None, init=False, repr=False
    )
    """
    The current parent symbolic expression of this expression during evaluation. Since a node can have multiple parents,
    this attribute is used to track the current parent that is being evaluated.
    """

    @lru_cache
    def _get_expression_by_id_(self, id_: int) -> SymbolicExpression:
        try:
            return next(
                expression
                for expression in self._all_expressions_
                if expression._id_ == id_
            )
        except StopIteration:
            raise NoExpressionFoundForGivenID(self, id_)

    @property
    def _is_false_(self) -> bool:
        """
        :return: The current truth value of evaluation result for this expression.
        """
        return self._is_false__

    @_is_false_.setter
    def _is_false_(self, value: bool):
        """
        Set the current truth value of evaluation result for this expression.
        """
        self._is_false__ = value

    def tolist(self):
        """
        Evaluate and return the results as a list.
        """
        return list(self.evaluate())

    def evaluate(
        self,
        limit: Optional[int] = None,
    ) -> Iterator[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]]:
        """
        Evaluate the query and map the results to the correct output data structure.
        This is the exposed evaluation method for users.

        :param limit: The maximum number of results to return. If None, return all results.
        """
        SymbolGraph().remove_dead_instances()
        results = map(
            self._process_result_, (res for res in self._evaluate_() if res.is_true)
        )
        if limit is None:
            yield from results
        elif not isinstance(limit, int) or limit <= 0:
            raise NonPositiveLimitValue(limit)
        else:
            for res_num, result in enumerate(results, 1):
                yield result
                if res_num == limit:
                    return

    def _replace_child_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        """
        Replace a child expression with a new child expression.

        :param old_child: The old child expression.
        :param new_child: The new child expression.
        """
        _children_ids_ = [v._id_ for v in self._children_]
        child_idx = _children_ids_.index(old_child._id_)
        self._children_[child_idx] = new_child
        new_child._parent_ = self
        old_child._remove_parent_(self)
        self._replace_child_field_(old_child, new_child)

    @abstractmethod
    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        """
        Replace a child field with a new child expression.

        :param old_child: The old child expression.
        :param new_child: The new child expression.
        """
        pass

    def _remove_parent_(self, parent: SymbolicExpression):
        self._parents_.remove(parent)
        if parent is self._parent__:
            self._parent_ = None

    def _update_children_(
        self, *children: SymbolicExpression
    ) -> Tuple[SymbolicExpression, ...]:
        """
        Update multiple children expressions of this symbolic expression.
        """
        children = dict(enumerate(children))
        for k, v in children.items():
            if not isinstance(v, SymbolicExpression):
                children[k] = Literal(v)
        for k, v in children.items():
            # With graph structure, do not copy nodes; just connect an edge.
            v._parent_ = self
        return tuple(children.values())

    def _on_parent_update_(self, parent: SymbolicExpression) -> None:
        """
        This method is called when the parent is updated. Subclasses should implement this method if they have logic
        to be done when a parent is updated.

        :param parent: The new parent expression.
        """
        pass

    def _process_result_(self, result: OperationResult) -> Any:
        """
        Map the result to the correct output data structure for user usage. It defaults to returning the bindings
        as a dictionary mapping variable objects to their values.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        return UnificationDict(
            {self._get_expression_by_id_(id_): v for id_, v in result.bindings.items()}
        )

    def _evaluate_(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ):
        """
        Wrapper for ``SymbolicExpression._evaluate__*`` methods that automatically
        manages the ``_eval_parent_`` attribute during evaluation.

        This wraps evaluation generator methods so that, for the duration
        of the wrapped call, ``self._eval_parent_`` is set to the ``parent`` argument
        passed to the evaluation method and then restored to its previous value
        afterwards. This allows evaluation code to reliably inspect the current
        parent expression without having to manage this state manually.

        :param sources: The current bindings of variables.
        :return: An Iterator method whose body automatically sets and restores ``self._eval_parent_`` around the
        underlying evaluation logic.
        """

        previous_parent = self._eval_parent_
        self._eval_parent_ = parent
        try:
            sources = sources or {}
            if self._binding_id_ in sources:
                yield OperationResult(sources, self._is_false_, self)
            else:
                yield from map(
                    self._evaluate_conclusions_and_update_bindings_,
                    self._evaluate__(sources),
                )
        finally:
            self._eval_parent_ = previous_parent

    def _evaluate_conclusions_and_update_bindings_(
        self, current_result: OperationResult
    ) -> OperationResult:
        """
        Update the bindings of the results by evaluating the conclusions using the received bindings.

        :param current_result: The current result of this expression.
        """
        # Only evaluate the conclusions at the root condition expression (i.e. after all conditions have been evaluated)
        # and when the result truth value is True.
        if not (self._conditions_root_ is self) or current_result.is_false:
            return current_result
        for conclusion in self._conclusion_:
            current_result.bindings = next(
                conclusion._evaluate_(current_result.bindings, parent=self)
            ).bindings
        return current_result

    @cached_property
    def _binding_id_(self) -> int:
        """
        The binding id is the id used in the bindings (the results dictionary of operations). It is sometimes different
        from the id of the symbolic expression itself because some operations do not have results themselves but their
        children do, so they delegate the binding id to one of their children. For example, in the case of quantifiers,
        the quantifier expression itself does not have a binding id, but it delegates it to its child variable that is
         being selected and tracked.
        """
        return self._id_

    @abstractmethod
    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterator[OperationResult]:
        """
        Evaluate the symbolic expression and set the operands indices.
        This method should be implemented by subclasses.
        """
        pass

    def _add_conclusion_(self, conclusion: Conclusion):
        """
        Add a conclusion expression to this symbolic expression.
        """
        self._conclusion_.add(conclusion)

    @property
    def _parent_(self) -> Optional[SymbolicExpression]:
        """
        :return: The parent symbolic expression of this expression.
        """
        if self._eval_parent_ is not None:
            return self._eval_parent_
        elif self._parent__ is not None:
            return self._parent__
        return None

    @_parent_.setter
    def _parent_(self, value: Optional[SymbolicExpression]):
        """
        Set the parent symbolic expression of this expression.

        :param value: The new parent symbolic expression of this expression.
        """
        if value is self:
            return

        if value is None and self._parent__ is not None:
            if self._id_ in [v._id_ for v in self._parent__._children_]:
                self._parent__._children_.remove(self)
            self._parents_.remove(self._parent__)

        self._parent__ = value

        if value is not None and value._id_ not in [v._id_ for v in self._parents_]:
            self._parents_.append(value)
            self._on_parent_update_(value)

        if value is not None and self._id_ not in [v._id_ for v in value._children_]:
            value._children_.append(self)

    @property
    def _conditions_root_(self) -> Optional[SymbolicExpression]:
        """
        :return: The root of the symbolic expression graph that contains conditions, or None if no conditions found.
        """
        for expression in self._all_expressions_:
            if isinstance(expression, Filter):
                return expression.condition
        return self._root_

    @property
    def _root_(self) -> SymbolicExpression:
        """
        :return: The root of the symbolic expression tree.
        """
        expression = self
        while expression._parent_ is not None:
            expression = expression._parent_
        return expression

    @property
    @abstractmethod
    def _name_(self) -> str:
        """
        :return: The name of this symbolic expression.
        """
        pass

    @property
    def _all_expressions_(self) -> Iterator[SymbolicExpression]:
        """
        :return: All nodes in the symbolic expression tree.
        """
        yield self._root_
        yield from self._root_._descendants_

    @property
    def _descendants_(self) -> Iterator[SymbolicExpression]:
        """
        :return: All descendants of this symbolic expression.
        """
        yield from self._children_
        for child in self._children_:
            yield from child._descendants_

    @classmethod
    def _current_parent_in_context_stack_(cls) -> Optional[SymbolicExpression]:
        """
        :return: The current parent symbolic expression in the enclosing context of the ``with`` statement. Used when
        making rule trees.
        """
        if cls._symbolic_expression_stack_:
            return cls._symbolic_expression_stack_[-1]
        return None

    @cached_property
    def _unique_variables_(self) -> Set[Variable]:
        """
        :return: Set of unique variables in this symbolic expression.
        """
        return make_set(self._all_variable_instances_)

    @cached_property
    @abstractmethod
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        ...

    def __and__(self, other):
        return AND(self, other)

    def __or__(self, other):
        return optimize_or(self, other)

    def _invert_(self):
        """
        Invert the symbolic expression.
        """
        return Not(self)

    def __enter__(self) -> Self:
        """
        Enter a context where this symbolic expression is the current parent symbolic expression. This updates the
        current parent symbolic expression, the context stack and returns this expression.
        """
        expression = self
        if (expression is self._root_) or (expression._parent_ is self._root_):
            expression = expression._conditions_root_
        SymbolicExpression._symbolic_expression_stack_.append(expression)
        return self

    def __exit__(self, *args):
        """
        Exit the context and remove this symbolic expression from the context stack.
        """
        SymbolicExpression._symbolic_expression_stack_.pop()

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return self._name_


@dataclass(eq=False, repr=False)
class DerivedExpression(SymbolicExpression, ABC):
    """
    A symbolic expression that has its results derived from another symbolic expression, and thus it's value is the
    value of the child expression.
    """

    @property
    @abstractmethod
    def _original_expression_(self) -> SymbolicExpression:
        """
        The original expression from which this expression is derived.
        """
        ...

    @property
    def _binding_id_(self) -> int:
        return self._original_expression_._binding_id_

    @property
    def _is_false_(self) -> bool:
        return self._original_expression_._is_false_

    def _process_result_(self, result: OperationResult) -> Any:
        return self._original_expression_._process_result_(result)


@dataclass(eq=False, repr=False)
class UnaryExpression(SymbolicExpression, ABC):
    """
    A unary expression is a symbolic expression that takes a single argument (i.e., has a single child expression).
    The results of the child expression are the inputs to this expression.
    """

    _child_: SymbolicExpression
    """
    The child expression of this symbolic expression.
    """

    def __post_init__(self):
        self._child_ = self._update_children_(self._child_)[0]

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        if self._child_ is old_child:
            self._child_ = new_child

    @cached_property
    def _all_variable_instances_(self) -> List[Selectable]:
        return self._child_._all_variable_instances_

    @property
    def _name_(self) -> str:
        return self.__class__.__name__


@dataclass(eq=False, repr=False)
class MultiArityExpression(SymbolicExpression, ABC):
    """
    A multi-arity expression is a symbolic expression that takes multiple arguments (i.e., has multiple child
    expressions).
    """

    _operation_children_: Tuple[SymbolicExpression, ...] = field(default_factory=tuple)
    """
    The children expressions of this symbolic expression.
    """

    def __post_init__(self):
        self.update_children(*self._operation_children_)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        try:
            old_child_index = self._operation_children_.index(old_child)
            self._operation_children_ = (
                self._operation_children_[:old_child_index]
                + (new_child,)
                + self._operation_children_[old_child_index + 1 :]
            )
        except ValueError:
            pass

    def update_children(self, *children: SymbolicExpression) -> None:
        self._operation_children_ = self._update_children_(*children)

    @cached_property
    def _all_variable_instances_(self) -> List[Selectable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        variables = []
        for child in self._operation_children_:
            variables.extend(child._all_variable_instances_)
        return variables


@dataclass(eq=False, repr=False)
class BinaryExpression(SymbolicExpression, ABC):
    """
    A base class for binary operators that can be used to combine symbolic expressions.
    """

    left: SymbolicExpression
    """
    The left operand of the binary operator.
    """
    right: SymbolicExpression
    """
    The right operand of the binary operator.
    """

    def __post_init__(self):
        self.left, self.right = self._update_children_(self.left, self.right)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        if self.left is old_child:
            self.left = new_child
        elif self.right is old_child:
            self.right = new_child

    @cached_property
    def _all_variable_instances_(self) -> List[Selectable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        return self.left._all_variable_instances_ + self.right._all_variable_instances_


@dataclass(eq=False, repr=False)
class Product(MultiArityExpression):
    """
    A symbolic operation that evaluates its children in nested sequence, passing bindings from one to the next such that
    each binding has a value from each child expression. It represents a cartesian product of all child expressions.
    """

    @property
    def _name_(self) -> str:
        return f"Product()"

    def _evaluate__(self, sources: Bindings) -> Iterator[OperationResult]:
        """
        Evaluate the symbolic expressions by generating combinations of values from their evaluation generators.
        """
        yield from (
            OperationResult(bindings, False, self)
            for bindings in chain_evaluate_variables(
                self._operation_children_, sources, parent=self
            )
        )


@dataclass(eq=False, repr=False)
class Filter(DerivedExpression, ABC):
    """
    Data source that evaluates the truth value for each data point according to a condition expression and filters out
    the data points that do not satisfy the condition.
    The truth value of this expression is derived from the truth value of the condition expression.
    """

    @property
    def _original_expression_(self) -> SymbolicExpression:
        return self.condition

    @property
    @abstractmethod
    def condition(self) -> SymbolicExpression:
        """
        The conditions expression which generate the valid bindings that satisfy the constraints.
        """
        ...

    @property
    def _name_(self):
        return self.__class__.__name__

    def evaluate_conclusions_and_update_bindings(
        self, condition_result: OperationResult
    ) -> OperationResult:
        """
        Update the bindings of the results by evaluating the conclusions using the received bindings from the condition
         expression as sources.

        :param condition_result: The result of the condition expression.
        """
        if condition_result.is_false:
            return condition_result
        for conclusion in self.condition._conclusion_:
            condition_result.bindings = next(
                conclusion._evaluate_(condition_result.bindings, parent=self)
            ).bindings
        return condition_result


@dataclass(eq=False, repr=False)
class Where(Filter, UnaryExpression):
    """
    A symbolic expression that represents the `where()` statement of `QueryObjectDescriptor`. It is used to filter
    ungrouped data. Is constructed through the `Where()` method of the `QueryObjectDescriptor`.
    """

    @property
    def condition(self) -> SymbolicExpression:
        return self._child_

    def _evaluate__(self, sources: Bindings) -> Iterator[OperationResult]:
        yield from (
            result
            for result in self._child_._evaluate_(sources, parent=self)
            if result.is_true
        )


@dataclass(eq=False, repr=False)
class Having(Filter, BinaryExpression):
    """
    A symbolic having expression that can be used to filter the grouped results of a query.
    Is constructed through the `QueryObjectDescriptor` using the `having()` method.
    """

    left: GroupedBy
    """
    The grouped by expression that is used to group the results of the query. This is a child of the Having expression.
    As the results need to be grouped before filtering.
    """
    right: SymbolicExpression
    """
    The condition expression that is used to filter the grouped results of the query.
    """

    @property
    def condition(self) -> SymbolicExpression:
        return self.right

    @property
    def grouped_by(self) -> GroupedBy:
        return self.left

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from (
            OperationResult(
                annotated_result.bindings,
                self._is_false_,
                self,
            )
            for grouping_result in self.grouped_by._evaluate_(sources, parent=self)
            for annotated_result in self.condition._evaluate_(
                grouping_result.bindings, parent=self
            )
            if annotated_result.is_true
        )


@dataclass(eq=False, repr=False)
class OrderedBy(BinaryExpression, DerivedExpression):
    """
    Represents an ordered by clause in a query. This orders the results of query according to the values of the
    specified variable.
    """

    right: Selectable
    """
    The variable to order by.
    """
    descending: bool = False
    """
    Whether to order the results in descending order.
    """
    key: Optional[Callable] = None
    """
    A function to extract the key from the variable value.
    """

    @property
    def _original_expression_(self) -> SymbolicExpression:
        """
        The original expression that this expression was derived from.
        """
        return self.left

    @property
    def variable(self) -> Selectable:
        """
        The variable to order by.
        """
        return self.right

    def _evaluate__(self, sources: Bindings) -> Iterator[OperationResult]:
        results = list(self.left._evaluate_(sources, parent=self))
        yield from sorted(
            results,
            key=self.apply_key,
            reverse=self.descending,
        )

    def apply_key(self, result: OperationResult) -> Any:
        """
        Apply the key function to the variable to extract the reference value to order the results by.
        """
        var = self.variable
        var_id = var._binding_id_
        if var_id not in result:
            result[var_id] = next(var._evaluate_(result.bindings, self)).value
        variable_value = result.bindings[var_id]
        if self.key:
            return self.key(variable_value)
        else:
            return variable_value

    @property
    def _name_(self) -> str:
        return f"OrderedBy({self.variable._name_})"


@dataclass(eq=False, repr=False)
class Selectable(SymbolicExpression, Generic[T], ABC):
    _var_: Selectable[T] = field(init=False, default=None)
    """
    A variable that is used if the child class to this class want to provide a variable to be tracked other than 
    itself, this is specially useful for child classes that holds a variable instead of being a variable and want
     to delegate the variable behaviour to the variable it has instead.
    For example, this is the case for the ResultQuantifiers & QueryDescriptors that operate on a single selected
    variable.
    """

    _type_: Type[T] = field(init=False, default=None)
    """
    The type of the variable.
    """

    @cached_property
    def _binding_id_(self) -> int:
        return (
            self._var_._binding_id_
            if self._var_ is not None and self._var_ is not self
            else self._id_
        )

    @cached_property
    def _type__(self):
        return (
            self._var_._type_
            if self._var_ is not None and self._var_ is not self
            else None
        )

    def _process_result_(self, result: OperationResult) -> T:
        """
        Map the result to the correct output data structure for user usage.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        return result.value

    @property
    def _is_iterable_(self):
        """
        Whether the selectable is iterable.

        :return: True if the selectable is iterable, False otherwise.
        """
        if self._var_ and self._var_ is not self:
            return self._var_._is_iterable_
        return False


@dataclass
class DomainMappingCacheItem:
    """
    A cache item for domain mapping creation. To prevent recreating same mapping multiple times, mapping instances are
    stored in a dictionary with a hashable key. This class is used to generate the key for the dictionary that stores
    the mapping instances.
    """

    type: Type[DomainMapping]
    """
    The type of the domain mapping.
    """
    child: CanBehaveLikeAVariable
    """
    The child of the domain mapping (i.e. the original variable on which the domain mapping is applied).
    """
    args: Tuple[Any, ...] = field(default_factory=tuple)
    """
    Positional arguments to pass to the domain mapping constructor.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Keyword arguments to pass to the domain mapping constructor.
    """

    def __post_init__(self):
        self.args = (self.child,) + self.args

    @cached_property
    def all_kwargs(self):
        return merge_args_and_kwargs(
            self.type, self.args, self.kwargs, ignore_first=True
        )

    @cached_property
    def hashable_key(self):
        return (self.type,) + convert_args_and_kwargs_into_a_hashable_key(
            self.all_kwargs
        )

    def __hash__(self):
        return hash(self.hashable_key)

    def __eq__(self, other):
        return (
            isinstance(other, DomainMappingCacheItem)
            and self.hashable_key == other.hashable_key
        )


@dataclass(eq=False, repr=False)
class CanBehaveLikeAVariable(Selectable[T], ABC):
    """
    This class adds the monitoring/tracking behavior on variables that tracks attribute access, calling,
    and comparison operations.
    """

    _known_mappings_: Dict[DomainMappingCacheItem, DomainMapping] = field(
        init=False, default_factory=dict
    )
    """
    A storage of created domain mappings to prevent recreating same mapping multiple times.
    """

    def _update_truth_value_(self, current_value: Any) -> None:
        """
        Updates the truth value of the variable based on the current value.

        :param current_value: The current value of the variable.
        """
        if isinstance(self._parent_, (LogicalOperator, Filter)):
            is_true = (
                len(current_value) > 0
                if is_iterable(current_value)
                else bool(current_value)
            )
            self._is_false_ = not is_true

    def _get_domain_mapping_(
        self, type_: Type[DomainMapping], *args, **kwargs
    ) -> DomainMapping:
        """
        Retrieves or creates a domain mapping instance based on the provided arguments.

        :param type_: The type of the domain mapping to retrieve or create.
        :param args: Positional arguments to pass to the domain mapping constructor.
        :param kwargs: Keyword arguments to pass to the domain mapping constructor.
        :return: The retrieved or created domain mapping instance.
        """
        cache_item = DomainMappingCacheItem(type_, self, args, kwargs)
        if cache_item in self._known_mappings_:
            return self._known_mappings_[cache_item]
        else:
            instance = type_(**cache_item.all_kwargs)
            self._known_mappings_[cache_item] = instance
            return instance

    def _get_domain_mapping_key_(self, type_: Type[DomainMapping], *args, **kwargs):
        """
        Generates a hashable key for the given type and arguments.

        :param type_: The type of the domain mapping.
        :param args: Positional arguments to pass to the domain mapping constructor.
        :param kwargs: Keyword arguments to pass to the domain mapping constructor.
        :return: The generated hashable key.
        """
        args = (self,) + args
        all_kwargs = merge_args_and_kwargs(type_, args, kwargs, ignore_first=True)
        return convert_args_and_kwargs_into_a_hashable_key(all_kwargs)

    def __getattr__(self, name: str) -> CanBehaveLikeAVariable[T]:
        # Prevent debugger/private attribute lookups from being interpreted as symbolic attributes
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )
        return self._get_domain_mapping_(Attribute, name, self._type__)

    def __getitem__(self, key) -> CanBehaveLikeAVariable[T]:
        return self._get_domain_mapping_(Index, key)

    def __call__(self, *args, **kwargs) -> CanBehaveLikeAVariable[T]:
        return self._get_domain_mapping_(Call, args, kwargs)

    def __eq__(self, other) -> Comparator:
        return Comparator(self, other, operator.eq)

    def __ne__(self, other) -> Comparator:
        return Comparator(self, other, operator.ne)

    def __lt__(self, other) -> Comparator:
        return Comparator(self, other, operator.lt)

    def __le__(self, other) -> Comparator:
        return Comparator(self, other, operator.le)

    def __gt__(self, other) -> Comparator:
        return Comparator(self, other, operator.gt)

    def __ge__(self, other) -> Comparator:
        return Comparator(self, other, operator.ge)

    def __hash__(self):
        return super().__hash__()


@dataclass(eq=False, repr=False)
class Aggregator(UnaryExpression, CanBehaveLikeAVariable[T], ABC):
    """
    Base class for aggregators. Aggregators are unary selectable expressions that take a single expression
     as a child.
    They aggregate the results of the child expression and evaluate to either a single value or a set of aggregated
     values for each group when `grouped_by()` is used.
    """

    _default_value_: Optional[T] = field(kw_only=True, default=None)
    """
    The default value to be returned if the child results are empty.
    """
    _distinct_: bool = field(kw_only=True, default=False)
    """
    Whether to consider only distinct values from the child results when applying the aggregation function.
    """

    def __post_init__(self):
        if isinstance(self._child_, Aggregator):
            raise NestedAggregationError(self)
        super().__post_init__()
        self._var_ = self

    def evaluate(self, limit: Optional[int] = None) -> Iterator[T]:
        """
        Wrap the aggregator in an entity and evaluate it (i.e., make a query with this aggregator as the selected
        expression and evaluate it.).

        :param limit: The maximum number of results to return. If None, all results are returned.
        :return: An iterator over the aggregator results.
        """
        return Entity(_selected_variables_=(self,)).evaluate()

    def grouped_by(self, *variables: Variable) -> Entity[T]:
        """
        Group the results by the given variables.
        """
        return Entity(_selected_variables_=(self,)).grouped_by(*variables)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from (
            OperationResult(
                sources
                | self._apply_aggregation_function_and_get_bindings_(child_result),
                False,
                self,
            )
            for child_result in self._child_._evaluate_(sources, parent=self)
        )

    @abstractmethod
    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Bindings:
        """
        Apply the aggregation function to the results of the child.

        :param child_result: The result of the child.
        :return: Bindings containing the aggregated result.
        """
        ...


@dataclass(eq=False, repr=False)
class Count(Aggregator[T]):
    """
    Count the number of child results.
    """

    _child_: Optional[SymbolicExpression] = None
    """
    The child expression to be counted. If not given, the count of all results (by group if `grouped_by()` is specified)
     is returned.
    """

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Bindings:
        if self._distinct_:
            return {self._binding_id_: len(set(child_result.value))}
        else:
            return {self._binding_id_: len(child_result.value)}


@dataclass(eq=False, repr=False)
class EntityAggregator(Aggregator[T], ABC):
    """
    Entity aggregators are aggregators where the child (the entity to be aggregated) is a selectable expression. Also,
     If given, make use of the key function to extract the value to be aggregated from the child result.
    """

    _child_: Selectable[T]
    """
    The child entity to be aggregated.
    """
    _key_function_: Optional[Callable[[Any], Any]] = field(kw_only=True, default=None)
    """
    An optional function that extracts the value to be used in the aggregation.
    """

    def __post_init__(self):
        if not isinstance(self._child_, Selectable):
            raise InvalidChildType(type(self._child_), [Selectable])
        self._var_ = self
        super().__post_init__()

    def get_aggregation_result_from_child_result(self, result: OperationResult) -> Any:
        """
        :param result: The current operation result from the child.
        :return: The aggregated result or the default value if the child result is empty.
        """
        if not result.has_value or len(result.value) == 0:
            return self._default_value_
        results = result.value
        if self._distinct_:
            results = set(results)
        return self.aggregation_function(results)

    @abstractmethod
    def aggregation_function(self, result: Collection) -> Any:
        """
        :param result: The child result to be aggregated.
        :return: The aggregated result.
        """
        ...


Number = int | float
"""
A type representing a number, which can be either an integer or a float.
"""


@dataclass(eq=False, repr=False)
class Sum(EntityAggregator[Number]):
    """
    Calculate the sum of the child results.
    """

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Dict[int, Optional[Number]]:
        return {
            self._binding_id_: self.get_aggregation_result_from_child_result(
                child_result
            )
        }

    def aggregation_function(self, result: Collection[Number]) -> Number:
        return sum(result)


@dataclass(eq=False, repr=False)
class Average(Sum):
    """
    Calculate the average of the child results.
    """

    def aggregation_function(self, result: Collection[Number]) -> Number:
        sum_value = super().aggregation_function(result)
        return sum_value / len(result)


@dataclass(eq=False, repr=False)
class Extreme(EntityAggregator[T], ABC):
    """
    Find and return the extreme value among the child results. If given, make use of the key function to extract
    the value to be compared.
    """

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Bindings:
        extreme_val = self.get_aggregation_result_from_child_result(child_result)
        bindings = child_result.bindings.copy()
        bindings[self._binding_id_] = extreme_val
        return bindings


@dataclass(eq=False, repr=False)
class Max(Extreme[T]):
    """
    Find and return the maximum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def aggregation_function(self, values: Iterable) -> Any:
        return max(values, key=self._key_function_)


@dataclass(eq=False, repr=False)
class Min(Extreme[T]):
    """
    Find and return the minimum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def aggregation_function(self, values: Iterable) -> Any:
        return min(values, key=self._key_function_)


@dataclass(eq=False)
class ResultQuantifier(UnaryExpression, DerivedExpression, ABC):
    """
    Base for quantifiers that return concrete results from entity/set queries
    (e.g., An, The).
    """

    _quantification_constraint_: Optional[ResultQuantificationConstraint] = None
    """
    The quantification constraint that must be satisfied by the result quantifier if present.
    """

    @property
    def _original_expression_(self) -> SymbolicExpression:
        return self._child_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[T]:

        result_count = 0
        values = self._child_._evaluate_(sources, parent=self)
        for value in values:
            result_count += 1
            self._assert_satisfaction_of_quantification_constraints_(
                result_count, done=False
            )
            yield OperationResult(value.bindings, False, self)
        self._assert_satisfaction_of_quantification_constraints_(
            result_count, done=True
        )

    def _assert_satisfaction_of_quantification_constraints_(
        self, result_count: int, done: bool
    ):
        """
        Assert the satisfaction of quantification constraints.

        :param result_count: The current count of results
        :param done: Whether all results have been processed
        :raises QuantificationNotSatisfiedError: If the quantification constraints are not satisfied.
        """
        if self._quantification_constraint_:
            self._quantification_constraint_.assert_satisfaction(
                result_count, self, done
            )

    def _invert_(self):
        raise UnsupportedNegation(self.__class__)

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        if self._quantification_constraint_:
            name += f"({self._quantification_constraint_})"
        return name


class UnificationDict(UserDict):
    """
    A dictionary which maps all expressions that are on a single variable to the original variable id.
    """

    def __getitem__(self, key: Selectable[T]) -> T:
        key = self._id_expression_map_[key._binding_id_]
        return super().__getitem__(key)

    @cached_property
    def _id_expression_map_(self) -> Dict[int, Selectable[T]]:
        return {key._binding_id_: key for key in self.data.keys()}


@dataclass(eq=False, repr=False)
class An(ResultQuantifier):
    """Quantifier that yields all matching results one by one."""

    ...


@dataclass(eq=False, repr=False)
class The(ResultQuantifier):
    """
    Quantifier that expects exactly one result; raises MultipleSolutionFound if more, and NoSolutionFound if none.
    """

    _quantification_constraint_: ResultQuantificationConstraint = field(
        init=False, default_factory=lambda: Exactly(1)
    )

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]]:
        """
        Evaluates the query object descriptor with the given bindings and yields the results.

        :raises MultipleSolutionFound: If more than one result is found.
        :raises NoSolutionFound: If no result is found.
        """
        try:
            yield from super()._evaluate__(sources)
        except LessThanExpectedNumberOfSolutions:
            raise NoSolutionFound(self)
        except GreaterThanExpectedNumberOfSolutions:
            raise MultipleSolutionFound(self)


GroupBindings = Dict[GroupKey, OperationResult]
"""
A dictionary for grouped bindings which maps a group key to its corresponding bindings.
"""


@dataclass(eq=False, repr=False)
class GroupedBy(UnaryExpression):
    """
    Represents a group-by operation in the entity query language. This operation groups the results of a query by
    specific variables. This is useful for aggregating results separately for each group.
    """

    _child_: Product
    """
    The child of the grouped-by operation.
    """
    aggregators: Tuple[Aggregator, ...]
    """
    The aggregators to apply to the grouped results.
    """
    variables_to_group_by: Tuple[Selectable, ...] = ()
    """
    The variables to group the results by their values.
    """

    def _evaluate__(self, sources: Bindings = None) -> Iterator[OperationResult]:
        """
        Generate results grouped by the specified variables in the grouped_by clause.

        :param sources: The current bindings.
        :return: An iterator of OperationResult objects, each representing a group of child results.
        """

        if len(self.aggregators_of_grouped_by_variables_that_are_not_count) > 0:
            raise UnsupportedAggregationOfAGroupedByVariable(self)

        groups, group_key_count = self.get_groups_and_group_key_count(sources)

        for agg in self.aggregators_of_grouped_by_variables:
            for group_key, group in groups.items():
                group[agg._binding_id_] = group_key_count[group_key]

        yield from groups.values()

    def get_groups_and_group_key_count(
        self, sources: Bindings
    ) -> Tuple[GroupBindings, Dict[GroupKey, int]]:
        """
        Create a dictionary of groups and a dictionary of group keys to their corresponding counts starting from the
        initial bindings, then applying the constraints in the where expression then grouping by the variables in the
        grouped_by clause.

        :param sources: The initial bindings.
        :return: A tuple containing the dictionary of groups and the dictionary of group keys to their corresponding counts.
        """

        groups = defaultdict(lambda: OperationResult({}, False, self))
        group_key_count = defaultdict(lambda: 0)

        for res in self._child_._evaluate_(sources, parent=self):

            group_key = tuple(
                ensure_hashable(res[var._binding_id_])
                for var in self.variables_to_group_by
            )

            if self.count_occurrences_of_each_group_key:
                group_key_count[group_key] += 1

            self.update_group_from_bindings(groups[group_key], res.bindings)

        if len(groups) == 0:
            for var in self.aggregated_variables:
                groups[()][var._binding_id_] = []

        return groups, group_key_count

    @cached_property
    def aggregated_variables(self) -> Tuple[SymbolicExpression, ...]:
        """
        :return: A tuple of the aggregated variables in the selected variables of the query descriptor.
        """
        return tuple(var._child_ for var in self.aggregators if var._child_ is not None)

    def update_group_from_bindings(self, group: OperationResult, results: Bindings):
        """
        Updates the group with the given results.

        :param group: The group to be updated.
        :param results: The results to be added to the group.
        """
        for id_, val in results.items():
            if id_ in self.ids_of_variables_to_group_by:
                group[id_] = val
            elif self.is_already_grouped(id_):
                group[id_] = val if is_iterable(val) else [val]
            else:
                if id_ not in group:
                    group[id_] = []
                group[id_].append(val)

    @lru_cache
    def is_already_grouped(self, var_id: int) -> bool:
        expression = self._get_expression_by_id_(var_id)
        return (
            len(self.variables_to_group_by) == 1
            and isinstance(expression, DomainMapping)
            and expression._child_._binding_id_ in self.ids_of_variables_to_group_by
        )

    @cached_property
    def count_occurrences_of_each_group_key(self) -> bool:
        """
        :return: True if there are any aggregators of type Count in the selected variables of the query descriptor that
         are counting values of variables that are in the grouped_by clause, False otherwise.
        """
        return len(self.aggregators_of_grouped_by_variables) > 0

    @cached_property
    def aggregators_of_grouped_by_variables_that_are_not_count(
        self,
    ) -> Tuple[Aggregator, ...]:
        """
        :return: Aggregators in the selected variables of the query descriptor that are aggregating over
         expressions having variables that are in the grouped_by clause and are not Count.
        """
        return tuple(
            var
            for var in self.aggregators_of_grouped_by_variables
            if not isinstance(var, Count)
        )

    @cached_property
    def aggregators_of_grouped_by_variables(self):
        """
        :return: A list of the aggregators in the selected variables of the query descriptor that are aggregating over
         expressions having variables that are in the grouped_by clause.
        """
        return [
            var
            for var in self.aggregators
            if (var._child_ is None)
            or (var._child_._binding_id_ in self.ids_of_variables_to_group_by)
        ]

    @cached_property
    def ids_of_variables_to_group_by(self) -> Tuple[int, ...]:
        """
        :return: A tuple of the binding IDs of the variables to group by.
        """
        return tuple(var._binding_id_ for var in self.variables_to_group_by)

    @property
    def _name_(self) -> str:
        return f"{self.__class__.__name__}({', '.join([var._name_ for var in self.variables_to_group_by])})"


ResultMapping = Callable[[Iterable[Bindings]], Iterator[Bindings]]
"""
A function that maps the results of a query object descriptor to a new set of results.
"""


@dataclass
class ExpressionBuilder(ABC):
    """
    Base class for builder classes of symbolic expressions. This class collects meta-data about expressions to finally
    build the expression.
    """

    query_descriptor: QueryObjectDescriptor
    """
    The query object descriptor that the expression is being built for.
    """

    @abstractmethod
    @cached_property
    def expression(self) -> SymbolicExpression:
        """
        The expression that is built from the metadata.
        """

    def __hash__(self) -> int:
        return hash((self.__class__, self.query_descriptor))


@dataclass(eq=False)
class TruthAnnotatorBuilder(ExpressionBuilder, ABC):
    """
    Metadata for constraint specifiers.
    """

    conditions: Tuple[ConditionType, ...]
    """
    The conditions that must be satisfied.
    """

    def __post_init__(self):
        self.assert_correct_conditions()

    def assert_correct_conditions(self):
        """
        :raises NoConditionsProvidedToWhereStatementOfDescriptor: If no conditions are provided.
        :raises LiteralConditionError: If any of the conditions is a literal expression.
        """
        # If there are no conditions raise error.
        if len(self.conditions) == 0:
            raise NoConditionsProvided(self.query_descriptor)

        # If there's a constant condition raise error.
        literal_expressions = [
            exp for exp in self.conditions if not isinstance(exp, SymbolicExpression)
        ]
        if literal_expressions:
            raise LiteralConditionError(self.query_descriptor, literal_expressions)

    @cached_property
    def aggregators_and_non_aggregators_in_conditions(
        self,
    ) -> Tuple[Tuple[Aggregator, ...], Tuple[Selectable, ...]]:
        """
        :return: A tuple containing the aggregators and non-aggregators in the conditions.
        """
        aggregators, non_aggregators = [], []
        for cond in self.conditions:
            if isinstance(cond, Aggregator):
                aggregators.append(cond)
            elif isinstance(cond, Selectable) and not isinstance(cond, Literal):
                non_aggregators.append(cond)
            for var in cond._children_:
                if isinstance(var, Aggregator):
                    aggregators.append(var)
                elif isinstance(var, DomainMapping) and any(
                    isinstance(v, Aggregator) for v in var._descendants_
                ):
                    aggregators.append(var)
                elif isinstance(var, Selectable) and not isinstance(var, Literal):
                    non_aggregators.append(var)
        return tuple(aggregators), tuple(non_aggregators)

    @cached_property
    def conditions_expression(self) -> SymbolicExpression:
        """
        :return: The expression representing the conditions of the constraint specifier.
        """
        return chained_logic(AND, *self.conditions)


@dataclass(eq=False)
class WhereBuilder(TruthAnnotatorBuilder):
    """
    Metadata for the `Where` constraint specifier.
    """

    def assert_correct_conditions(self):
        """
        Assert that the where conditions are correct.

        :raises AggregatorInWhereConditionsError: If the where conditions contain any aggregators.
        """
        super().assert_correct_conditions()
        aggregators, non_aggregators = (
            self.aggregators_and_non_aggregators_in_conditions
        )
        if aggregators:
            raise AggregatorInWhereConditionsError(
                aggregators, descriptor=self.query_descriptor
            )

    @cached_property
    def expression(self) -> Where:
        return Where(self.conditions_expression)


@dataclass(eq=False)
class HavingBuilder(TruthAnnotatorBuilder):
    """
    Metadata for the `Having` constraint specifier.
    """

    grouped_by: GroupedBy = field(kw_only=True, default=None)
    """
    The GroupedBy expression associated with the having constraint specifier, as the having conditions are applied on
     the aggregations of grouped results.
    """

    def assert_correct_conditions(self):
        """
        Assert that the having conditions are correct.

        :raises NonAggregatorInHavingConditionsError: If the having conditions contain any non-aggregator expressions.
        """
        super().assert_correct_conditions()
        aggregators, non_aggregators = (
            self.aggregators_and_non_aggregators_in_conditions
        )
        if non_aggregators:
            raise NonAggregatorInHavingConditionsError(
                non_aggregators, descriptor=self.query_descriptor
            )

    @cached_property
    def expression(self) -> Having:
        return Having(self.grouped_by, self.conditions_expression)


@dataclass(eq=False)
class GroupedByBuilder(ExpressionBuilder):
    """
    Metadata for the GroupedBy operation.
    """

    variables_to_group_by: Tuple[Selectable, ...] = ()
    """
    The variables to group the results by their values.
    """

    def __post_init__(self):
        self.assert_correct_selected_variables()

    @cached_property
    def expression(self) -> GroupedBy:
        aggregated_variables, non_aggregated_variables = (
            self.query_descriptor._aggregated_and_non_aggregated_variables_in_selection_
        )
        group_by_entity_selected_variables = non_aggregated_variables + [
            var._child_ for var in aggregated_variables if var._child_ is not None
        ]
        where = self.query_descriptor._where_expression_
        children = []
        if where:
            children.append(where)
        children.extend(group_by_entity_selected_variables)
        return GroupedBy(
            _child_=Product(tuple(children)),
            aggregators=tuple(self.aggregators),
            variables_to_group_by=tuple(self.variables_to_group_by),
        )

    @lru_cache
    def assert_correct_selected_variables(self):
        """
        Assert that the selected variables are correct.

        :raises UsageError: If the selected variables are not valid.
        """
        aggregators, non_aggregated_variables = (
            self.query_descriptor._aggregated_and_non_aggregated_variables_in_selection_
        )
        if aggregators and not all(
            self.variable_is_in_or_derived_from_a_grouped_by_variable(v)
            for v in non_aggregated_variables
        ):
            raise NonAggregatedSelectedVariablesError(
                self,
                non_aggregated_variables,
                aggregators,
                descriptor=self.query_descriptor,
            )

    @lru_cache
    def variable_is_in_or_derived_from_a_grouped_by_variable(
        self, variable: SymbolicExpression
    ) -> bool:
        """
        Check if the variable is in or derived from a grouped by variable.

        :param variable: The variable to check.
        """
        if variable._binding_id_ in self.ids_of_variables_to_group_by:
            return True
        elif variable._binding_id_ in self.ids_of_aggregated_variables:
            return False
        elif isinstance(variable, DomainMapping) and any(
            self.variable_is_in_or_derived_from_a_grouped_by_variable(d)
            for d in variable._descendants_
        ):
            return True
        elif (
            isinstance(variable, Variable)
            and isinstance(variable._domain_source_, Selectable)
            and self.variable_is_in_or_derived_from_a_grouped_by_variable(
                variable._domain_source_
            )
        ):
            return True
        else:
            return False

    @cached_property
    def ids_of_aggregated_variables(self) -> Tuple[int, ...]:
        """
        :return: A tuple of ids of aggregated variables.
        """
        return tuple(
            v._child_._binding_id_ for v in self.aggregators if v._child_ is not None
        )

    @cached_property
    def ids_of_variables_to_group_by(self) -> Tuple[int, ...]:
        """
        :return: A tuple of the binding IDs of the variables to group by.
        """
        return tuple(var._binding_id_ for var in self.variables_to_group_by)

    @cached_property
    def aggregators(self) -> Tuple[Aggregator, ...]:
        """
        :return: A tuple of aggregators in the selected variables of the query descriptor.
        """
        return tuple(
            var
            for var in self.query_descriptor._selected_variables_
            if isinstance(var, Aggregator)
        )


@dataclass(eq=False)
class QuantifierBuilder(ExpressionBuilder):
    type: Type[ResultQuantifier] = An
    """
    The type of the quantifier to be built.
    """
    quantification_constraint: Optional[ResultQuantificationConstraint] = None
    """
    The quantification constraint that must be satisfied by the result quantifier if present.
    """

    @cached_property
    def expression(self) -> ResultQuantifier:
        """
        Builds a result quantifier of the specified type with the given child and quantification constraint.
        """
        if self.type is An:
            return self.type(
                self.query_descriptor._expression_,
                _quantification_constraint_=self.quantification_constraint,
            )
        else:
            return self.type(self.query_descriptor._expression_)


@dataclass(eq=False)
class OrderedByBuilder(ExpressionBuilder):
    variable: Selectable
    """
    The variable to order by.
    """
    descending: bool = False
    """
    Whether to order the results in descending order.
    """
    key: Optional[Callable] = None
    """
    A function to extract the key from the variable value.
    """

    @cached_property
    def expression(self) -> SymbolicExpression:
        return OrderedBy(
            self.query_descriptor, self.variable, self.descending, self.key
        )


@dataclass(eq=False, repr=False)
class QueryObjectDescriptor(UnaryExpression, ABC):
    """
    Describes the queried object(s), could be a query over a single variable or a set of variables,
    also describes the condition(s)/properties of the queried object(s).
    """

    _child_: Product = field(default_factory=Product)
    """
    The child of the query object descriptor is the root of the conditions in the query/sub-query graph.
    """
    _selected_variables_: Tuple[Selectable, ...] = field(
        default_factory=tuple, kw_only=True
    )
    """
    The variables that are selected by the query object descriptor.
    """
    _distinct_on: Tuple[Selectable, ...] = field(default=(), init=False)
    """
    Parameters for distinct results of the query object descriptor.
    """
    _results_mapping: List[ResultMapping] = field(init=False, default_factory=list)
    """
    Mapping functions that map the results of the query object descriptor to a new set of results.
    """
    _seen_results: Optional[SeenSet] = field(init=False, default=None)
    """
    A set of seen results, used when distinct is called in the query object descriptor.
    """
    _where_builder_: Optional[WhereBuilder] = field(init=False, default=None)
    """
    The builder for the `Where` expression of the query object descriptor.
    """
    _grouped_by_builder_: Optional[GroupedByBuilder] = field(init=False, default=None)
    """
    The builder for the `GroupedBy` expression of the query object descriptor.
    """
    _having_builder: Optional[HavingBuilder] = field(init=False, default=None)
    """
    The builder for the `Having` expression of the query object descriptor.
    """
    _ordered_by_builder_: Optional[OrderedByBuilder] = field(default=None, init=False)
    """
    The builder for the `OrderedBy` expression if present.
    """
    _quantifier_builder_: Optional[QuantifierBuilder] = field(default=None, init=False)
    """
    The builder for the `ResultQuantifier` expression of the query object descriptor. The default quantifier is `An`
     which yields all results.
    """
    _built_: bool = field(default=False, init=False)
    """
    Whether the query object descriptor has built the query (wired the query operations) or not. If built already, it
    cannot be modified further and an error will be raised if a user tries to modify the query object descriptor.
    """

    def __post_init__(self):
        super().__post_init__()
        for var in self._selected_variables_:
            if isinstance(var, QueryObjectDescriptor):
                var.build()
        self._quantifier_builder_ = QuantifierBuilder(self)

    @staticmethod
    def modifies_query_structure(method):
        """
        A decorator to mark methods that modify the structure of the query. If the query is already
        built, an error will be raised when trying to call any of these methods.
        """

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self._built_:
                raise TryingToModifyAnAlreadyBuiltQuery(self)
            return method(self, *args, **kwargs)

        return wrapper

    def tolist(self) -> List:
        """
        Map the results of the query object descriptor to a list of the selected variable values.

        :return: A list of the selected variable values.
        """
        return list(self.evaluate())

    def evaluate(self, limit: Optional[int] = None) -> Iterator:
        """
        Wrap the query object descriptor in a ResultQuantifier expression and evaluate it,
         returning an iterator over the results.
        """
        return self._quantifier_builder_.expression.evaluate(limit=limit)

    @modifies_query_structure
    def where(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query object descriptor.
        """
        if self._where_builder_ is None:
            self._where_builder_ = WhereBuilder(
                conditions=conditions, query_descriptor=self
            )
        else:
            self._where_builder_.conditions += conditions
        return self

    @modifies_query_structure
    def having(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query object descriptor.
        """
        if self._having_builder is None:
            self._having_builder = HavingBuilder(
                conditions=conditions, query_descriptor=self
            )
        else:
            self._having_builder.conditions += conditions
        return self

    def ordered_by(
        self,
        variable: TypingUnion[Selectable[T], Any],
        descending: bool = False,
        key: Optional[Callable] = None,
    ) -> Self:
        """
        Order the results by the given variable, using the given key function in descending or ascending order.

        :param variable: The variable to order by.
        :param descending: Whether to order the results in descending order.
        :param key: A function to extract the key from the variable value.
        """
        self._ordered_by_builder_ = OrderedByBuilder(
            self, variable, descending=descending, key=key
        )
        # build ordered by expression; this is fine outside the build() as ordered by is the last operation.
        _ = self._ordered_by_builder_.expression
        return self

    def distinct(
        self,
        *on: TypingUnion[Selectable, Any],
    ) -> TypingUnion[Self, T]:
        """
        Apply distinctness constraint to the query object descriptor results.

        :param on: The variables to be used for distinctness.
        :return: This query object descriptor.
        """
        self._distinct_on = on if on else self._selected_variables_
        self._seen_results = SeenSet(keys=self._distinct_on_ids_)
        self._results_mapping.append(self._get_distinct_results_)
        return self

    @modifies_query_structure
    def grouped_by(
        self, *variables_to_group_by: TypingUnion[Selectable, Any]
    ) -> TypingUnion[Self, T]:
        """
        Specify the variables to group the results by.

        :param variables_to_group_by: The variables to group the results by.
        :return: This query object descriptor.
        """
        self._grouped_by_builder_ = GroupedByBuilder(self, variables_to_group_by)
        return self

    def _quantify_(
        self,
        quantifier_type: Type[ResultQuantifier] = An,
        quantification_constraint: Optional[ResultQuantificationConstraint] = None,
    ) -> Self:
        """
        Specify the quantifier type and constraint for the query results, also build the query.

        :param quantifier_type: The type of the quantifier to be used.
        :param quantification_constraint: The constraint to apply to the quantifier.
        :return: This query object descriptor.
        """
        self._quantifier_builder_ = QuantifierBuilder(
            self, quantifier_type, quantification_constraint
        )
        return self

    def __enter__(self):
        """
        Make sure the query is built before entering the context manager for rule trees.
        """
        self.build()
        super().__enter__()

    def _on_parent_update_(self, parent: SymbolicExpression) -> None:
        """
        A parent update means this query object descriptor is complete and should be built.
        """
        self.build()

    def build(self) -> Self:
        """
        Build the query object descriptor by wiring the nodes together in the correct order of evaluation.

        :return: This query object descriptor.
        """
        if self._built_:
            return self

        self._built_ = True

        if self._group_ and self._grouped_by_builder_ is None:
            self._grouped_by_builder_ = GroupedByBuilder(self)

        children = []
        if self._having_builder is not None:
            self._having_builder.grouped_by = self._grouped_by_builder_.expression
            children.append(self._having_builder.expression)
        elif self._grouped_by_builder_ is not None:
            children.append(self._grouped_by_builder_.expression)
        elif self._where_builder_ is not None:
            children.append(self._where_builder_.expression)

        children.extend(self._selected_not_inferred_variables_)

        self._child_.update_children(*children)

        return self

    @property
    def _expression_(self) -> SymbolicExpression:
        """
        The expression representing the query (without quantification), built by wiring the operations together.
        """
        self.build()
        if self._ordered_by_builder_ is not None:
            return self._ordered_by_builder_.expression
        return self

    @property
    def _ordered_by_expression_(self) -> Optional[OrderedByBuilder]:
        return (
            self._ordered_by_builder_.expression if self._ordered_by_builder_ else None
        )

    @cached_property
    def _selected_not_inferred_variables_(self) -> Tuple[SymbolicExpression, ...]:
        return tuple(
            var
            for var in self._selected_variables_
            if not (isinstance(var, Variable) and var._is_inferred_)
        )

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the query descriptor by constraining values, updating conclusions,
        and selecting variables.
        """

        if all(var._binding_id_ in sources for var in self._selected_variables_):
            yield OperationResult(sources, False, self)
            return

        yield from self._generate_results_(sources)

        if self._seen_results is not None:
            self._seen_results.clear()

    @property
    def _where_expression_(self) -> Optional[Where]:
        """
        The built `Where` expression.
        """
        return self._where_builder_.expression if self._where_builder_ else None

    @property
    def _having_expression_(self) -> Optional[Having]:
        """
        The built `Having` expression.
        """
        return self._having_builder.expression if self._having_builder else None

    @property
    def _grouped_by_expression_(self) -> Optional[GroupedBy]:
        """
        The built `GroupedDataSource` expression.
        """
        return (
            self._grouped_by_builder_.expression if self._grouped_by_builder_ else None
        )

    @property
    def _quantifier_expression_(self) -> Optional[ResultQuantifier]:
        return (
            self._quantifier_builder_.expression if self._quantifier_builder_ else None
        )

    def _generate_results_(self, sources: Dict[int, Any]) -> Iterator[OperationResult]:
        """
        Internal generator to process constrained values and selected variables.
        """
        # First evaluate the constraint (if any) and handle conclusions
        yield from (
            OperationResult(result, False, self)
            for result in self._apply_results_mapping_(
                self._get_constrained_values_(sources)
            )
        )

    @cached_property
    def _distinct_on_ids_(self) -> Tuple[int, ...]:
        """
        Get the IDs of variables used for distinctness.
        """
        return tuple(k._binding_id_ for k in self._distinct_on)

    def _get_distinct_results_(
        self, results_gen: Iterable[Dict[int, Any]]
    ) -> Iterator[Dict[int, Any]]:
        """
        Apply distinctness constraint to the query object descriptor results.

        :param results_gen: Generator of result dictionaries.
        :return: Generator of distinct result dictionaries.
        """
        for res in results_gen:
            self._update_res_with_distinct_on_variables_(res)
            if self._seen_results.check(res):
                continue
            self._seen_results.add(res)
            yield res

    def _update_res_with_distinct_on_variables_(self, res: Dict[int, Any]):
        """
        Update the result dictionary with values from distinct-on variables if not already present.

        :param res: The result dictionary to update.
        """
        for i, id_ in enumerate(self._distinct_on_ids_):
            if id_ in res:
                continue
            var_value = self._distinct_on[i]._evaluate_(copy(res), parent=self)
            res[id_] = next(var_value).value

    @cached_property
    def _group_(self) -> bool:
        """
        :return: Whether the results should be grouped or not. Is true when an aggregator is selected.
        """
        return (
            len(self._aggregated_and_non_aggregated_variables_in_selection_[0]) > 0
        ) or (self._grouped_by_builder_ is not None)

    @cached_property
    def _aggregated_and_non_aggregated_variables_in_selection_(
        self,
    ) -> Tuple[List[Selectable], List[Selectable]]:
        """
        :return: The aggregated and non-aggregated variables from the selected variables.
        """
        aggregated_variables = []
        non_aggregated_variables = []
        for variable in self._selected_variables_:
            if isinstance(variable, Aggregator):
                aggregated_variables.append(variable)
            else:
                non_aggregated_variables.append(variable)
        return aggregated_variables, non_aggregated_variables

    @staticmethod
    def _variable_is_inferred_(var: Selectable[T]) -> bool:
        """
        Whether the variable is inferred or not.

        :param var: The variable.
        :return: True if the variable is inferred, otherwise False.
        """
        return isinstance(var, Variable) and var._is_inferred_

    def _any_selected_variable_is_inferred_and_unbound_(self, values: Bindings) -> bool:
        """
        Check if any of the selected variables is inferred and is not bound.

        :param values: The current result with the current bindings.
        :return: True if any of the selected variables is inferred and is not bound, otherwise False.
        """
        return any(
            not self._variable_is_bound_or_its_children_are_bound_(
                var, tuple(values.keys())
            )
            for var in self._selected_variables_
            if self._variable_is_inferred_(var)
        )

    @lru_cache
    def _variable_is_bound_or_its_children_are_bound_(
        self, var: Selectable[T], result: Tuple[int, ...]
    ) -> bool:
        """
        Whether the variable is directly bound or all its children are bound.

        :param var: The variable.
        :param result: The current result containing the current bindings.
        :return: True if the variable is bound, otherwise False.
        """
        if var._binding_id_ in result:
            return True
        unique_vars = [uv for uv in var._unique_variables_ if uv is not var]
        if unique_vars and all(
            self._variable_is_bound_or_its_children_are_bound_(uv, result)
            for uv in unique_vars
        ):
            return True
        return False

    def _get_constrained_values_(self, sources: Bindings) -> Iterator[Bindings]:
        """
        Evaluate the child (i.e., the conditions that constrain the domain of the selected variables).

        :param sources: The current bindings.
        :return: The bindings after applying the constraints of the child.
        """

        for result in self._child_._evaluate_(sources, parent=self):

            if self._any_selected_variable_is_inferred_and_unbound_(result.bindings):
                continue

            yield result.bindings

    def _apply_results_mapping_(
        self, results: Iterator[Bindings]
    ) -> Iterable[Bindings]:
        """
        Process and transform an iterable of results based on predefined mappings and ordering.

        This method applies a sequence of result transformations defined in the instance,
        using a series of mappings to modify the results.

        :param results: An iterable containing dictionaries that represent the initial result set to be transformed.
        :return: An iterable containing dictionaries that represent the transformed data.
        """
        for result_mapping in self._results_mapping:
            results = result_mapping(results)
        return results

    def _invert_(self):
        raise UnsupportedNegation(self.__class__)

    @property
    def _name_(self) -> str:
        return f"({', '.join(var._name_ for var in self._selected_variables_)})"


@dataclass(eq=False, repr=False)
class SetOf(QueryObjectDescriptor):
    """
    A query over a set of variables.
    """

    def _process_result_(self, result: OperationResult) -> UnificationDict:
        """
        Map the result to the correct output data structure for user usage. This returns the selected variables only.
        Return a dictionary with the selected variables as keys and the values as values.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        return UnificationDict(
            {v._var_: result[v._binding_id_] for v in self._selected_variables_}
        )

    def __getitem__(
        self, selected_variable: TypingUnion[CanBehaveLikeAVariable[T], T]
    ) -> TypingUnion[T, SetOfSelectable[T]]:
        """
        Select one of the set of variables, this is useful when you have another query that uses this set of and
        wants to select a specific variable out of the set of variables.

        :param selected_variable: The selected variable from the set of variables.
        """
        self.build()
        return SetOfSelectable(self, selected_variable)


@dataclass(eq=False, repr=False)
class SetOfSelectable(UnaryExpression, CanBehaveLikeAVariable[T]):
    """
    A selected variable from the SetOf operation selected variables.
    """

    _child_: SetOf
    """
    The SetOf operation from which `_selected_var_` was selected.
    """
    _selected_var_: CanBehaveLikeAVariable[T]
    """
    The selected variable in the SetOf.
    """

    def __post_init__(self):
        self._var_ = self

    @property
    def _set_of_(self) -> SetOf:
        """
        The SetOf operation from which `_selected_var_` was selected.
        """
        return self._child_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterator[OperationResult]:
        for v in self._set_of_._evaluate_(sources, self):
            yield OperationResult(
                {**v.bindings, self._binding_id_: v[self._selected_var_._binding_id_]},
                False,
                self,
            )

    @property
    def _name_(self) -> str:
        return f"{self._set_of_._name_}.{self._selected_var_._name_}"

    @cached_property
    def _all_variable_instances_(self) -> List[Selectable]:
        return self._set_of_._all_variable_instances_


@dataclass(eq=False, repr=False)
class Entity(QueryObjectDescriptor, CanBehaveLikeAVariable[T]):
    """
    A query over a single variable.
    """

    def __post_init__(self):
        self._var_ = self.selected_variable
        super().__post_init__()

    @property
    def selected_variable(self):
        return self._selected_variables_[0] if self._selected_variables_ else None


@dataclass(eq=False, repr=False)
class Variable(CanBehaveLikeAVariable[T]):
    """
    A Variable that queries will assign. The Variable produces results of type `T`.
    """

    _type_: Type = field(default=MISSING)
    """
    The result type of the variable. (The value of `T`)
    """

    _name__: str
    """
    The name of the variable.
    """

    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The properties of the variable as keyword arguments.
    """

    _domain_source_: Optional[DomainType] = field(
        default=None, kw_only=True, repr=False
    )
    """
    An optional source for the variable domain. If not given, the global cache of the variable class type will be used
    as the domain, or if kwargs are given the type and the kwargs will be used to inference/infer new values for the
    variable.
    """
    _domain_: ReEnterableLazyIterable = field(
        default_factory=ReEnterableLazyIterable, kw_only=True, repr=False
    )
    """
    The iterable domain of values for this variable.
    """
    _predicate_type_: Optional[PredicateType] = field(default=None, repr=False)
    """
    If this symbol is an instance of the Predicate class.
    """
    _is_inferred_: bool = field(default=False, repr=False)
    """
    Whether this variable should be inferred.
    """
    _child_vars_: Optional[Dict[str, SymbolicExpression]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    A dictionary mapping child variable names to variables, these are from the _kwargs_ dictionary. 
    """

    def __post_init__(self):
        self._child_ = None

        if self._domain_source_:
            self._update_domain_(self._domain_source_)

        self._var_ = self

        self._update_child_vars_from_kwargs_()

    def _update_domain_(self, domain):
        """
        Set the domain and ensure it is a lazy re-enterable iterable.
        """
        if isinstance(domain, CanBehaveLikeAVariable):
            self._update_children_(domain)
        if isinstance(domain, (ReEnterableLazyIterable, CanBehaveLikeAVariable)):
            self._domain_ = domain
            return
        if not is_iterable(domain):
            domain = [domain]
        self._domain_.set_iterable(domain)

    def _update_child_vars_from_kwargs_(self):
        """
        Set the child variables from the kwargs dictionary.
        """
        for k, v in self._kwargs_.items():
            if isinstance(v, SymbolicExpression):
                self._child_vars_[k] = v
            else:
                self._child_vars_[k] = Literal(v, name=k)
        self._update_children_(*self._child_vars_.values())

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        for k, v in self._child_vars_.items():
            if v is old_child:
                self._child_vars_[k] = new_child
                break

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        A variable either is already bound in sources by other constraints (Symbolic Expressions).,
        or will yield from current domain if exists,
        or has no domain and will instantiate new values by constructing the type if the type is given.
        """

        if self._domain_source_ is not None:
            yield from self._iterator_over_domain_values_(sources)
        elif self._is_inferred_ or self._predicate_type_:
            yield from self._instantiate_using_child_vars_and_yield_results_(sources)
        else:
            raise VariableCannotBeEvaluated(self)

    def _iterator_over_domain_values_(
        self, sources: Bindings
    ) -> Iterable[OperationResult]:
        """
        Iterate over the values in the variable's domain, yielding OperationResult instances.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        if isinstance(self._domain_, CanBehaveLikeAVariable):
            yield from self._iterator_over_variable_domain_values_(sources)
        else:
            yield from self._iterator_over_iterable_domain_values_(sources)

    def _iterator_over_variable_domain_values_(self, sources: Bindings):
        """
        Iterate over the values in the variable's domain, where the domain is another variable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for domain in self._domain_._evaluate_(sources, parent=self):
            for v in domain.value:
                bindings = {**sources, **domain.bindings, self._binding_id_: v}
                yield self._build_operation_result_and_update_truth_value_(bindings)

    def _iterator_over_iterable_domain_values_(self, sources: Bindings):
        """
        Iterate over the values in the variable's domain, where the domain is an iterable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for v in self._domain_:
            bindings = {**sources, self._binding_id_: v}
            yield self._build_operation_result_and_update_truth_value_(bindings)

    def _instantiate_using_child_vars_and_yield_results_(
        self, sources: Bindings
    ) -> Iterable[OperationResult]:
        """
        Create new instances of the variable type and using as keyword arguments the child variables values.
        """
        for kwargs in self._generate_combinations_for_child_vars_values_(sources):
            # Build once: unwrapped hashed kwargs for already provided child vars
            bound_kwargs = {
                k: v[self._child_vars_[k]._binding_id_] for k, v in kwargs.items()
            }
            instance = self._type_(**bound_kwargs)
            yield self._process_output_and_update_values_(instance, kwargs)

    def _generate_combinations_for_child_vars_values_(self, sources: Bindings):
        yield from generate_combinations(
            {k: var._evaluate_(sources, self) for k, var in self._child_vars_.items()}
        )

    def _process_output_and_update_values_(
        self, instance: Any, kwargs: Dict[str, OperationResult]
    ) -> OperationResult:
        """
        Process the predicate/variable instance and get the results.

        :param instance: The created instance.
        :param kwargs: The keyword arguments of the predicate/variable, which are a mapping kwarg_name: {var_id: value}.
        :return: The results' dictionary.
        """
        # kwargs is a mapping from name -> {var_id: value};
        # we need a single dict {var_id: value}
        values = {self._binding_id_: instance}
        for d in kwargs.values():
            values.update(d.bindings)
        return self._build_operation_result_and_update_truth_value_(values)

    def _build_operation_result_and_update_truth_value_(
        self, bindings: Bindings
    ) -> OperationResult:
        """
        Build an OperationResult instance and update the truth value based on the bindings.

        :param bindings: The bindings of the result.
        :return: The OperationResult instance with updated truth value.
        """
        self._update_truth_value_(bindings[self._binding_id_])
        return OperationResult(bindings, self._is_false_, self)

    @property
    def _name_(self):
        return self._name__

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        variables = [self]
        for v in self._child_vars_.values():
            variables.extend(v._all_variable_instances_)
        return variables

    @property
    def _is_iterable_(self):
        return is_iterable(next(iter(self._domain_), None))


@dataclass(eq=False, init=False, repr=False)
class Literal(Variable[T]):
    """
    Literals are variables that are not constructed by their type but by their given data.
    """

    def __init__(
        self,
        data: Any,
        name: Optional[str] = None,
        type_: Optional[Type] = None,
        wrap_in_iterator: bool = True,
    ):
        original_data = data
        if wrap_in_iterator:
            data = [data]
        if not type_:
            original_data_lst = make_list(original_data)
            first_value = original_data_lst[0] if len(original_data_lst) > 0 else None
            type_ = type(first_value) if first_value else None
        if name is None:
            if type_:
                name = type_.__name__
            else:
                if isinstance(data, Selectable):
                    name = data._name_
                else:
                    name = type(original_data).__name__
        super().__init__(_name__=name, _type_=type_, _domain_source_=data)


@dataclass(eq=False, repr=False)
class Concatenate(MultiArityExpression, CanBehaveLikeAVariable[T]):
    """
    Concatenation of two or more variables.
    """

    _operation_children_: Tuple[Selectable[T], ...]
    """
    The children of the concatenate operation.
    """

    @property
    def _variables_(self) -> Tuple[Selectable[T], ...]:
        """
        The variables to concatenate.
        """
        return self._operation_children_

    def __post_init__(self):
        self._update_children_(*self._variables_)
        self._var_ = self

    def _evaluate__(self, sources: Bindings) -> Iterable[OperationResult]:

        for var in self._variables_:
            for var_val in var._evaluate_(sources, self):
                self._is_false_ = var_val.is_false
                yield OperationResult(
                    {**sources, **var_val.bindings, self._id_: var_val.value},
                    var_val.is_false,
                    self,
                )

    @property
    def _name_(self):
        return self.__class__.__name__


@dataclass(eq=False, repr=False)
class DomainMapping(UnaryExpression, CanBehaveLikeAVariable[T], ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """

    _child_: CanBehaveLikeAVariable[T]
    """
    The child expression to apply the domain mapping to.
    """

    def __post_init__(self):
        super().__post_init__()
        self._var_ = self

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_

    @cached_property
    def _type_(self):
        return self._child_._type_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Apply the domain mapping to the child's values.
        """

        yield from (
            self._build_operation_result_and_update_truth_value_(
                child_result, mapped_value
            )
            for child_result in self._child_._evaluate_(sources, parent=self)
            for mapped_value in self._apply_mapping_(child_result.value)
        )

    def _build_operation_result_and_update_truth_value_(
        self, child_result: OperationResult, current_value: Any
    ) -> OperationResult:
        """
        Set the current truth value of the operation result, and build the operation result to be yielded.

        :param child_result: The current result from the child operation.
        :param current_value: The current value of this operation that is derived from the child result.
        :return: The operation result.
        """
        self._update_truth_value_(current_value)
        return OperationResult(
            {**child_result.bindings, self._binding_id_: current_value},
            self._is_false_,
            self,
        )

    @abstractmethod
    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        """
        Apply the domain mapping to a symbolic value.
        """
        pass


@dataclass(eq=False, repr=False)
class Attribute(DomainMapping):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.

    For instance, if Body.name is called, then the attribute name is "name" and `_owner_class_` is `Body`
    """

    _attribute_name_: str
    """
    The name of the attribute.
    """

    _owner_class_: Type
    """
    The class that owns this attribute.
    """

    @property
    def _is_iterable_(self):
        if not self._wrapped_field_:
            return False
        return self._wrapped_field_.is_iterable

    @cached_property
    def _type_(self) -> Optional[Type]:
        """
        :return: The type of the accessed attribute.
        """

        if not is_dataclass(self._owner_class_):
            return None

        if self._attribute_name_ not in {f.name for f in fields(self._owner_class_)}:
            return None

        if self._wrapped_owner_class_:
            # try to get the type endpoint from a field
            try:
                return self._wrapped_field_.type_endpoint
            except (KeyError, AttributeError):
                return None
        else:
            wrapped_cls = WrappedClass(self._owner_class_)
            wrapped_cls._class_diagram = SymbolGraph().class_diagram
            wrapped_field = WrappedField(
                wrapped_cls,
                [
                    f
                    for f in fields(self._owner_class_)
                    if f.name == self._attribute_name_
                ][0],
            )
            try:
                return wrapped_field.type_endpoint
            except (AttributeError, RuntimeError):
                return None

    @cached_property
    def _wrapped_field_(self) -> Optional[WrappedField]:
        if self._wrapped_owner_class_ is None:
            return None
        return self._wrapped_owner_class_._wrapped_field_name_map_.get(
            self._attribute_name_, None
        )

    @cached_property
    def _wrapped_owner_class_(self):
        """
        :return: The owner class of the attribute from the symbol graph.
        """
        try:
            return SymbolGraph().class_diagram.get_wrapped_class(self._owner_class_)
        except ClassIsUnMappedInClassDiagram:
            return None

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        yield getattr(value, self._attribute_name_)

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}.{self._attribute_name_}"


@dataclass(eq=False, repr=False)
class Index(DomainMapping):
    """
    A symbolic indexing operation that can be used to access items of symbolic variables via [] operator.
    """

    _key_: Any
    """
    The key to index with.
    """

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        yield value[self._key_]

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}[{self._key_}]"


@dataclass(eq=False, repr=False)
class Call(DomainMapping):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """

    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    """
    The arguments to call the method with.
    """
    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The keyword arguments to call the method with.
    """

    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        if len(self._args_) > 0 or len(self._kwargs_) > 0:
            yield value(*self._args_, **self._kwargs_)
        else:
            yield value()

    @property
    def _name_(self):
        return f"{self._child_._var_._name_}()"


@dataclass(eq=False, repr=False)
class Flatten(DomainMapping):
    """
    Domain mapping that flattens an iterable-of-iterables into a single iterable of items.

    Given a child expression that evaluates to an iterable (e.g., Views.bodies), this mapping yields
    one solution per inner element while preserving the original bindings (e.g., the View instance),
    similar to UNNEST in SQL.
    """

    def _apply_mapping_(self, value: Iterable[Any]) -> Iterable[Any]:
        yield from value

    @cached_property
    def _name_(self):
        return f"Flatten({self._child_._name_})"

    @property
    def _is_iterable_(self):
        """
        :return: False as Flatten does not preserve the original iterable structure.
        """
        return False


def not_contains(container, item) -> bool:
    """
    The inverted contains operation.

    :param container: The container.
    :param item: The item to test if contained in the container.
    :return:
    """
    return not operator.contains(container, item)


@dataclass(eq=False, repr=False)
class Comparator(BinaryExpression):
    """
    A symbolic equality check that can be used to compare symbolic variables using a provided comparison operation.
    """

    left: Selectable
    right: Selectable
    operation: Callable[[Any, Any], bool]
    operation_name_map: ClassVar[Dict[Any, str]] = {
        operator.eq: "==",
        operator.ne: "!=",
        operator.lt: "<",
        operator.le: "<=",
        operator.gt: ">",
        operator.ge: ">=",
    }

    @property
    def _name_(self):
        if self.operation in self.operation_name_map:
            return self.operation_name_map[self.operation]
        return self.operation.__name__

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Compares the left and right symbolic variables using the "operation".
        """

        first_operand, second_operand = self.get_first_second_operands(sources)

        yield from (
            OperationResult(
                second_val.bindings, not self.apply_operation(second_val), self
            )
            for first_val in first_operand._evaluate_(sources, parent=self)
            if first_val.is_true
            for second_val in second_operand._evaluate_(first_val.bindings, parent=self)
            if second_val.is_true
        )

    def apply_operation(self, operand_values: OperationResult) -> bool:
        left_value, right_value = (
            operand_values[self.left._binding_id_],
            operand_values[self.right._binding_id_],
        )
        if (
            self.operation in [operator.eq, operator.ne]
            and is_iterable(left_value)
            and is_iterable(right_value)
        ):
            left_value = make_set(left_value)
            right_value = make_set(right_value)
        res = self.operation(left_value, right_value)
        self._is_false_ = not res
        operand_values[self._id_] = res
        return res

    def get_first_second_operands(
        self, sources: Bindings
    ) -> Tuple[SymbolicExpression, SymbolicExpression]:
        left_has_the = any(isinstance(desc, The) for desc in self.left._descendants_)
        right_has_the = any(isinstance(desc, The) for desc in self.right._descendants_)
        if left_has_the and not right_has_the:
            return self.left, self.right
        elif not left_has_the and right_has_the:
            return self.right, self.left
        if sources and any(
            v._binding_id_ in sources for v in self.right._unique_variables_
        ):
            return self.right, self.left
        else:
            return self.left, self.right


@dataclass(eq=False, repr=False)
class LogicalOperator(SymbolicExpression, ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions using logical constraints on their
    truth values. Examples are conjunction (AND), disjunction (OR), negation (NOT), and conditional quantification
    (ForALL, Exists).
    """

    @property
    def _name_(self):
        return self.__class__.__name__


@dataclass(eq=False, repr=False)
class Not(LogicalOperator, UnaryExpression):
    """
    The logical negation of a symbolic expression. Its truth value is the opposite of its child's truth value. This is
    used when you want bindings that satisfy the negated condition (i.e., that doesn't satisfy the original condition).
    """

    def __post_init__(self):
        if isinstance(self._child_, ResultQuantifier):
            raise UnSupportedOperand(self.__class__, self._child_)
        super().__post_init__()

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        for v in self._child_._evaluate_(sources, parent=self):
            self._is_false_ = v.is_true
            yield OperationResult(v.bindings, self._is_false_, self)


@dataclass(eq=False, repr=False)
class LogicalBinaryOperator(LogicalOperator, BinaryExpression, ABC):
    def __post_init__(self):
        if isinstance(self.left, ResultQuantifier):
            raise UnSupportedOperand(self.__class__, self.left)
        if isinstance(self.right, ResultQuantifier):
            raise UnSupportedOperand(self.__class__, self.right)
        super().__post_init__()


@dataclass(eq=False, repr=False)
class AND(LogicalBinaryOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        left_values = self.left._evaluate_(sources, parent=self)
        for left_value in left_values:
            self._is_false_ = left_value.is_false
            if self._is_false_:
                yield OperationResult(left_value.bindings, self._is_false_, self)
            else:
                yield from self.evaluate_right(left_value)

    def evaluate_right(self, left_value: OperationResult) -> Iterable[OperationResult]:
        right_values = self.right._evaluate_(left_value.bindings, parent=self)
        for right_value in right_values:
            self._is_false_ = right_value.is_false
            yield OperationResult(right_value.bindings, self._is_false_, self)


@dataclass(eq=False, repr=False)
class OR(LogicalBinaryOperator, ABC):
    """
    A symbolic single choice operation that can be used to choose between multiple symbolic expressions.
    """

    left_evaluated: bool = field(default=False, init=False)
    right_evaluated: bool = field(default=False, init=False)

    def evaluate_left(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the left operand, taking into consideration if it should yield when it is False.

        :param sources: The current bindings to use for evaluation.
        :return: The new bindings after evaluating the left operand (and possibly right operand).
        """
        left_values = self.left._evaluate_(sources, parent=self)

        for left_value in left_values:
            self.left_evaluated = True
            left_is_false = left_value.is_false
            if left_is_false:
                yield from self.evaluate_right(left_value.bindings)
            else:
                self._is_false_ = False
                yield OperationResult(left_value.bindings, self._is_false_, self)

    def evaluate_right(self, sources: Bindings) -> Iterable[OperationResult]:
        """
        Evaluate the right operand.

        :param sources: The current bindings to use during evaluation.
        :return: The new bindings after evaluating the right operand.
        """

        self.left_evaluated = False

        right_values = self.right._evaluate_(sources, parent=self)

        for right_value in right_values:
            self._is_false_ = right_value.is_false
            self.right_evaluated = True
            yield OperationResult(right_value.bindings, self._is_false_, self)

        self.right_evaluated = False


@dataclass(eq=False, repr=False)
class Union(OR):
    """
    This operator is a version of the OR operator that always evaluates both the left and the right operand.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from self.evaluate_left(sources)
        yield from self.evaluate_right(sources)


@dataclass(eq=False, repr=False)
class ElseIf(OR):
    """
    A version of the OR operator that evaluates the right operand only when the left operand is False.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle ElseIf logic.
        """
        yield from self.evaluate_left(sources)


@dataclass(eq=False, repr=False)
class QuantifiedConditional(LogicalBinaryOperator, ABC):
    """
    This is the super class of the universal, and existential conditional operators. It is a binary logical operator
    that has a quantified variable and a condition on the values of that variable.
    """

    @property
    def variable(self):
        return self.left

    @variable.setter
    def variable(self, value):
        self.left = value

    @property
    def condition(self):
        return self.right

    @condition.setter
    def condition(self, value):
        self.right = value


@dataclass(eq=False, repr=False)
class ForAll(QuantifiedConditional):
    """
    This operator is the universal conditional operator. It returns bindings that satisfy the condition for all the
    values of the quantified variable. It short circuits by ignoring the bindings that doesn't satisfy the condition.
    """

    @cached_property
    def condition_unique_variable_ids(self) -> List[int]:
        return [
            v._binding_id_
            for v in self.condition._unique_variables_.difference(
                self.left._unique_variables_
            )
        ]

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        solution_set = None

        for var_val in self.variable._evaluate_(sources, parent=self):
            if solution_set is None:
                solution_set = self.get_all_candidate_solutions(var_val.bindings)
            else:
                solution_set = [
                    sol
                    for sol in solution_set
                    if self.evaluate_condition({**sol, **var_val.bindings})
                ]
            if not solution_set:
                solution_set = []
                break

        # Yield the remaining bindings (non-universal) merged with the incoming sources
        yield from [
            OperationResult({**sources, **sol}, False, self) for sol in solution_set
        ]

    def get_all_candidate_solutions(self, sources: Bindings):
        values_that_satisfy_condition = []
        # Evaluate the condition under this particular universal value
        for condition_val in self.condition._evaluate_(sources, parent=self):
            if condition_val.is_false:
                continue
            condition_val_bindings = {
                k: v
                for k, v in condition_val.bindings.items()
                if k in self.condition_unique_variable_ids
            }
            values_that_satisfy_condition.append(condition_val_bindings)
        return values_that_satisfy_condition

    def evaluate_condition(self, sources: Bindings) -> bool:
        for condition_val in self.condition._evaluate_(sources, parent=self):
            return condition_val.is_true
        return False

    def _invert_(self):
        return Exists(self.variable, self.condition._invert_())


@dataclass(eq=False, repr=False)
class Exists(QuantifiedConditional):
    """
    An existential checker that checks if a condition holds for any value of the variable given, the benefit
    of this is that this short circuits the condition and returns True if the condition holds for any value without
    getting all the condition values that hold for one specific value of the variable.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        seen_var_values = []
        for val in self.condition._evaluate_(sources, parent=self):
            var_val = val[self.variable._binding_id_]
            if val.is_true and var_val not in seen_var_values:
                seen_var_values.append(var_val)
                yield OperationResult(val.bindings, False, self)

    def _invert_(self):
        return ForAll(self.variable, self.condition._invert_())


OperatorOptimizer = Callable[[SymbolicExpression, SymbolicExpression], LogicalOperator]


def chained_logic(
    operator: TypingUnion[Type[LogicalOperator], OperatorOptimizer], *conditions
):
    """
    A chian of logic operation over multiple conditions, e.g. cond1 | cond2 | cond3.

    :param operator: The symbolic operator to apply between the conditions.
    :param conditions: The conditions to be chained.
    """
    prev_operation = None
    for condition in conditions:
        if prev_operation is None:
            prev_operation = condition
            continue
        prev_operation = operator(prev_operation, condition)
    return prev_operation


def optimize_or(left: SymbolicExpression, right: SymbolicExpression) -> OR:
    left_vars = {v for v in left._unique_variables_ if not isinstance(v, Literal)}
    right_vars = {v for v in right._unique_variables_ if not isinstance(v, Literal)}
    if left_vars == right_vars:
        return ElseIf(left, right)
    else:
        return Union(left, right)


def _any_of_the_kwargs_is_a_variable(bindings: Dict[str, Any]) -> bool:
    """
    :param bindings: A kwarg like dict mapping strings to objects
    :return: Rather any of the objects is a variable or not.
    """
    return any(isinstance(binding, Selectable) for binding in bindings.values())


DomainType = TypingUnion[Iterable, None]
