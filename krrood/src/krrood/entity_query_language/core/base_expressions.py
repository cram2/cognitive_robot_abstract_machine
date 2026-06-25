"""
This module defines the fundamental symbolic expression classes for the Entity Query Language.

It contains the base classes for all expressions, results, and binding management used during query evaluation.
"""

from __future__ import annotations

import itertools
import uuid
from abc import ABC, abstractmethod
from collections import UserDict
from copy import copy
from dataclasses import dataclass, field
from functools import cached_property
from uuid import UUID

from ordered_set import OrderedSet
from typing_extensions import (
    Dict,
    Any,
    Optional,
    ClassVar,
    List,
    Iterator,
    Union as TypingUnion,
    Tuple,
    Set,
    Self,
    TYPE_CHECKING,
    Generic,
    Type,
)

from krrood.entity_query_language.exceptions import NoExpressionFoundForGivenID
from krrood.entity_query_language.utils import make_list, T, make_set, is_iterable
from krrood.symbol_graph.symbol_graph import SymbolGraph
from krrood.utils import memoize
from krrood.entity_query_language.evaluation_context import (
    EvaluationContext,
    get_evaluation_context,
    set_evaluation_context,
    _evaluation_context_var,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.rules.conclusion import Conclusion
    from krrood.entity_query_language.core.variable import Variable
    from krrood.entity_query_language.query.query import Query

Bindings = Dict[uuid.UUID, Any]
"""
A dictionary for expressions' bindings in EQL that maps the expression's unique identifier to its value.
"""


@dataclass(eq=False)
class SymbolicExpression(ABC):
    """
    Base class for all symbolic expressions.

    Symbolic expressions form a rooted directed acyclic graph and are evaluated lazily to produce
    bindings for variables, subject to logical constraints.
    """

    _id_: uuid.UUID = field(init=False, repr=False, default_factory=uuid.uuid4)
    """
    Unique identifier of this node.
    """
    _conclusions_: Set[Conclusion] = field(init=False, default_factory=set)
    """
    Set of conclusion expressions attached to this node, these are evaluated when the truth value of this node is true
    during evaluation.
    """
    _symbolic_expression_stack_: ClassVar[List[SymbolicExpression]] = []
    """
    The current stack of symbolic expressions that has been entered using the ``with`` statement.
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
    _expression_: SymbolicExpression = field(init=False, repr=False)
    """
    Useful when this expression is a builder that wires multiple components together to create the final expression.
    This defaults to Self.
    """
    _limit_: Optional[int] = field(init=False, repr=False, default=None)
    """
    The maximum number of results to return during evaluation.
    """
    _expression_id_cache_: dict[uuid.UUID, SymbolicExpression] = field(
        init=False, repr=False, default_factory=dict, compare=False
    )
    """
    Cache of expressions by their unique identifier.
    """

    def __post_init__(self):
        self._expression_ = self

    def _get_expression_by_id_(self, id_: uuid.UUID) -> SymbolicExpression:
        """
        Retrieve the expression with the given ID from the collection of all expressions.

        :param id_: The unique identifier of the expression to retrieve.
        :return: The expression with the specified ID, or raises NoExpressionFoundForGivenID if not found.
        """
        # Per-instance cache stored in _expression_id_cache_ so it is collected with the expression object.
        # A class-level @lru_cache would hold strong refs to `self` indefinitely, keeping
        # query trees (and their domain data) alive well beyond the query's lifetime.
        if id_ not in self._expression_id_cache_:
            try:
                self._expression_id_cache_[id_] = next(
                    expression
                    for expression in self._all_expressions_
                    if expression._id_ == id_
                )
            except StopIteration:
                raise NoExpressionFoundForGivenID(self, id_)
        return self._expression_id_cache_[id_]

    def tolist(
        self,
    ) -> list[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]]:
        """
        Evaluate and return the results as a list.
        """
        return make_list(self.evaluate())

    def first(self) -> TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]:
        """
        Evaluate and return the first result of the query object descriptor.

        :return: The first result of the query object descriptor.
        :raises StopIteration: If no results are found.
        """
        return next(self.evaluate())

    def evaluate(
        self,
    ) -> Iterator[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]]:
        """
        Evaluate the query and map the results to the correct output data structure.
        This is the exposed evaluation method for users.
        """
        SymbolGraph().remove_dead_instances()
        results = (
            self._process_result_(res) for res in self._evaluate_() if res.is_true
        )
        yield from itertools.islice(results, self._limit_)

    def _replace_child_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        """
        Replace a child expression with a new child expression.

        :param old_child: The old child expression.
        :param new_child: The new child expression.
        """
        if old_child is new_child:
            return
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
        """
        Remove the parent relationship between this expression and the given parent expression.

        :param parent: The parent expression to remove.
        """
        self._parents_.remove(parent)
        if parent is self._parent__:
            self._parent_ = None

    def _update_children_(
        self, *children: SymbolicExpression
    ) -> Tuple[SymbolicExpression, ...]:
        """
        Update multiple children expressions of this symbolic expression.

        :param children: The new children expressions. Non-``SymbolicExpression``
            values are wrapped in ``Literal`` instances before being attached.
        :return: A tuple of the updated child expressions corresponding to the
            provided ``children`` arguments.
        """
        from krrood.entity_query_language.core.variable import Literal

        children = [
            v if isinstance(v, SymbolicExpression) else Literal(_value_=v)
            for v in children
        ]
        for v in children:
            v._parent_ = self
        return tuple(v._expression_ for v in children)

    def _ensure_children_ids_are_cached_(self, *children: SymbolicExpression) -> None:
        """
        Ensure that the IDs of the provided children expressions are cached within the current expression.

        :param children: The children expressions to cache IDs for.
        """
        for child in children:
            if child._id_ not in self._expression_id_cache_:
                self._expression_id_cache_[child._id_] = child
                child._ensure_children_ids_are_cached_(*child._children_)

    def _process_result_(self, result: OperationResult) -> Any:
        """
        Map the result to the correct output data structure for user usage. It defaults to returning the bindings
        as a dictionary mapping variable objects to their values.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        if self._id_ in result:
            return result[self._id_]
        else:
            return UnificationDict(
                {
                    self._get_expression_by_id_(id_): value
                    for id_, value in result.bindings.items()
                }
            )

    def _evaluate_(
        self,
        sources: Optional[OperationResult] = None,
    ):
        """
        Wrapper for ``SymbolicExpression._evaluate__`` that manages evaluation context lifecycle.

        :param sources: The current OperationResult carrying bindings of variables, or None.
        :return: An iterator of OperationResult instances.
        """
        evaluation_context = get_evaluation_context()
        owns_an_evaluation_context = evaluation_context is None
        if owns_an_evaluation_context:
            from krrood.entity_query_language.evaluation import (
                create_default_evaluation_context,
            )

            evaluation_context = create_default_evaluation_context()
            context_token = set_evaluation_context(evaluation_context)
        try:
            evaluation_context.on_evaluate_enter(expression=self, sources=sources)
            # Normalize sources: always work with an OperationResult
            previous_result = sources
            if sources is not None:
                bindings = copy(sources.bindings)
            else:
                bindings = {}
                sources = OperationResult({})  # empty sentinel for _evaluate__()
            if self._id_ in bindings:
                result = OperationResult(bindings, False, self, previous_result)
                evaluation_context.on_result_yielded(expression=self, result=result)
                yield result
            else:
                for result in map(
                    self._evaluate_conclusions_and_update_bindings_,
                    self._evaluate__(sources),
                ):
                    evaluation_context.on_result_yielded(expression=self, result=result)
                    yield result
        finally:
            evaluation_context.on_evaluate_exit(expression=self)
            if owns_an_evaluation_context:
                _evaluation_context_var.reset(context_token)

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
        for conclusion in self._conclusions_:
            current_result.bindings = next(
                conclusion._evaluate_(current_result)
            ).bindings

        evaluation_context = get_evaluation_context()
        if evaluation_context is not None:
            evaluation_context.on_conclusions_processed(
                expression=self,
                result=current_result,
            )
        return current_result

    @abstractmethod
    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterator[OperationResult]:
        """
        Evaluate the symbolic expression and set the operands bindings in the result according to the evaluation logic
        of this expression.

        :param sources: The current OperationResult carrying bindings of variables.
        :return: An Iterator of OperationResult instances containing the bindings resulting from the evaluation of this
        expression.
        """
        pass

    @property
    def _parent_(self) -> Optional[SymbolicExpression]:
        """
        :return: The parent symbolic expression of this expression.
        """
        return self._parent__

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
            if self._parent__ in self._parents_:
                self._parents_.remove(self._parent__)

        self._parent__ = value

        if value is not None and value._id_ not in [v._id_ for v in self._parents_]:
            self._parents_.append(value)
            value._ensure_children_ids_are_cached_(self)

        if value is not None and self._id_ not in [v._id_ for v in value._children_]:
            value._children_.append(self)

    @property
    def _conditions_root_(self) -> Optional[SymbolicExpression]:
        """
        :return: The root of the symbolic expression graph that contains conditions, or None if no conditions found.
        """
        return next(
            (
                expr.condition
                for expr in self._all_expressions_
                if isinstance(expr, Filter)
            ),
            self._root_,
        )

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
    def _root_query_(self) -> Optional[Query]:
        """
        :return: The root query of the symbolic expression tree, or None if no query found.
        """
        from krrood.entity_query_language.query.query import Query

        root = self._root_
        root_query = None
        for descendant in root._descendants_:
            if isinstance(descendant, Query):
                root_query = descendant
                break
        return root_query

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
        :return: All descendants of this symbolic expression in children first, then depth-first by subtree order.
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
        if not cls._symbolic_expression_stack_:
            return None
        return cls._symbolic_expression_stack_[-1]

    @property
    def _unique_variables_(self) -> Set[Variable]:
        """
        :return: Set of unique variables in this symbolic expression.
        """
        return make_set(self._all_variable_instances_)

    @property
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        from krrood.entity_query_language.core.variable import Variable

        return [c for c in self._children_ if isinstance(c, Variable)]

    @property
    def _leaves_(self) -> Iterator[SymbolicExpression]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        if len(self._children_) == 0:
            yield self
        for child in self._children_:
            yield from child._leaves_

    def _invert_(self):
        """
        Invert the symbolic expression.
        """
        from krrood.entity_query_language.operators.core_logical_operators import Not

        return Not(self)

    def __enter__(self) -> Self:
        """
        Enter a context where this symbolic expression is the current parent symbolic expression. This updates the
        current parent symbolic expression, the context stack and returns this expression.
        """
        SymbolicExpression._symbolic_expression_stack_.append(self)
        return self

    def __exit__(self, *args):
        """
        Exit the context and remove this symbolic expression from the context stack.
        """
        SymbolicExpression._symbolic_expression_stack_.pop()

    def __hash__(self):
        return hash(self._id_)

    def __repr__(self):
        return self._name_


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
        super().__post_init__()
        self._child_ = self._update_children_(self._child_)[0]

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        if self._child_ is old_child:
            self._child_ = new_child

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
        super().__post_init__()
        self.update_children(*self._operation_children_)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        old_child_index = self._operation_children_.index(old_child)
        self._operation_children_ = (
            self._operation_children_[:old_child_index]
            + (new_child,)
            + self._operation_children_[old_child_index + 1 :]
        )

    def update_children(self, *children: SymbolicExpression) -> None:
        self._operation_children_ = self._update_children_(*children)

    @property
    def _name_(self) -> str:
        return self.__class__.__name__


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
        super().__post_init__()
        self.left, self.right = self._update_children_(self.left, self.right)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        if self.left is old_child:
            self.left = new_child
        elif self.right is old_child:
            self.right = new_child


@dataclass(eq=False, repr=False)
class TruthValueOperator(SymbolicExpression, ABC):
    """
    An abstract superclass for operators that work with truth values of operations, thus requiring its children
     expressions to update their truth value when yielding results.
    """

    def _evaluate_child_as_condition_(
        self, child: SymbolicExpression, sources: Optional[OperationResult]
    ) -> Iterator[OperationResult]:
        """
        Evaluate ``child`` and apply truth-value semantics to each result.

        Expressions that carry their own binding (Selectable: Variable, MappedVariable, Comparator, …)
        have their truth value computed from the binding's boolean value.  Expressions that do not
        self-bind (LogicalOperators: AND, OR, NOT, …) already carry the correct ``is_false`` flag and
        are yielded unchanged.

        :param child: The child expression to evaluate in a truth-value context.
        :param sources: The current OperationResult carrying bindings, or None.
        :return: An iterator of OperationResult instances with correct truth values.
        """
        for result in child._evaluate_(sources):
            if result.has_value:
                value = result.value
                is_false = not (len(value) > 0 if is_iterable(value) else bool(value))
                yield OperationResult(
                    result.bindings,
                    is_false,
                    result.operand,
                    result.previous_operation_result,
                )
            else:
                yield result


@dataclass(eq=False, repr=False)
class DerivedExpression(SymbolicExpression, ABC):
    """
    A symbolic expression that has its results derived from another symbolic expression, and thus it's value is the
    value of the child expression. For example, filter expressions just filter the results of their children but they
    do not produce a new value of their own, thus they do not have a binding that belongs to them specifically in the
    result bindings dictionary.
    """

    @property
    @abstractmethod
    def _original_expression_(self) -> SymbolicExpression:
        """
        The original expression from which this expression is derived.
        """
        ...

    def _process_result_(self, result: OperationResult) -> Any:
        return self._original_expression_._process_result_(result)


@dataclass(eq=False, repr=False)
class Filter(DerivedExpression, TruthValueOperator, ABC):
    """
    Data source that evaluates the truth value for each data point according to a condition expression and filters out
    the data points that do not satisfy the condition.
    """

    @property
    def _original_expression_(self) -> SymbolicExpression:
        return self.condition

    @property
    @abstractmethod
    def condition(self) -> SymbolicExpression:
        """
        The conditions expression that generates the valid bindings that satisfy the constraints.
        """
        ...

    @property
    def _name_(self):
        return self.__class__.__name__


@dataclass
class OperationResult:
    """
    A data structure that carries information about the result of an operation in EQL.
    """

    bindings: Bindings
    """
    The bindings resulting from the operation, mapping variable IDs to their values.
    """

    is_false: bool = False
    """
    Whether the operation resulted in a false value (i.e., The operation condition was not satisfied)
    """

    operand: Optional[SymbolicExpression] = None
    """
    The operand that produced the result.
    """

    previous_operation_result: Optional[OperationResult] = None
    """
    The result of the operation that was evaluated before this one.
    """

    satisfied_condition_ids: Optional[OrderedSet[UUID]] = None
    """
    A set of UUIDs of condition expressions in the condition tree that were satisfied (truth value = True)
    during this evaluation. Populated at the conditions root after all conditions have been evaluated.
    Only set when the overall condition result is True.
    """
    evaluated_expression_ids: Optional[OrderedSet[UUID]] = None
    """
    A set of UUIDs of all expressions that were evaluated along the evaluation path that produced
    this result. Populated by the EvaluationTracker observer. Unlike satisfied_condition_ids, this
    includes all evaluated expressions regardless of truth value.
    """

    @property
    def all_bindings(self) -> Bindings:
        """
        :return: All the bindings from all the evaluated operations until this one, including this one.
        Traverses the full previous_operation_result chain (linear traversal with cycle detection).
        """
        combined: Bindings = {}
        seen: set = set()

        def collect(node: Optional[OperationResult]) -> None:
            if node is None or id(node) in seen:
                return
            seen.add(id(node))
            collect(node.previous_operation_result)
            combined.update(node.bindings)  # shallower nodes (closer to self) win

        collect(self)
        return combined

    @property
    def has_value(self) -> bool:
        return self.operand is not None and self.operand._id_ in self.bindings

    @property
    def is_true(self) -> bool:
        return not self.is_false

    @property
    def value(self) -> Any:
        """
        The value of the operation result, retrieved from the bindings using the operand's ID.

        :raises: KeyError if the operand is not found in the bindings.
        """
        if self.operand is None:
            raise ValueError("Cannot get value: operand is None")
        return self.operand._process_result_(self)

    def __contains__(self, item):
        return item in self.bindings

    def __getitem__(self, item):
        return self.bindings[item]

    def __setitem__(self, key, value):
        self.bindings[key] = value

    def update(self, other: Bindings | OperationResult):
        if isinstance(other, OperationResult):
            self.bindings.update(other.bindings)
        else:
            self.bindings.update(other)
        return self

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return (
            self.bindings == other.bindings
            and self.is_true == other.is_true
            and self.operand == other.operand
            and self.previous_operation_result == other.previous_operation_result
        )


class UnificationDict(UserDict):
    """
    A dictionary which maps all expressions that are on a single variable to the original variable id.
    """

    def __getitem__(self, key: Selectable[T]) -> T:
        key = self._id_expression_map_[key._id_]
        return super().__getitem__(key)

    @cached_property
    def _id_expression_map_(self) -> Dict[uuid.UUID, Selectable[T]]:
        return {key._id_: key for key in self.data.keys()}


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
    The type of the selectable.
    """

    def __post_init__(self):
        super().__post_init__()
        if self._type_ is None:
            self._type_ = self._type__

    def _build_operation_result_and_update_truth_value_(
        self,
        bindings: Bindings,
        child_result: Optional[OperationResult] = None,
    ) -> OperationResult:
        """
        Build an OperationResult instance for this binding.

        :param bindings: The bindings of the result.
        :param child_result: The result of the child operation, if any.
        :return: The OperationResult instance.
        """
        return OperationResult(bindings, False, self, child_result)

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
        return result[self._id_]

    @cached_property
    def _name_(self):
        if self._type_:
            return self._type_.__name__
        return self.__class__.__name__
