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
from functools import cached_property, lru_cache

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

from ..failures import NoExpressionFoundForGivenID
from ..utils import make_list, T, make_set, is_iterable
from ...symbol_graph.symbol_graph import SymbolGraph

if TYPE_CHECKING:
    from ..rules.conclusion import Conclusion
    from .variable import Variable

Bindings = Dict[uuid.UUID, Any]
"""
A dictionary for variable bindings in EQL operations
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
    _expression_: SymbolicExpression = field(init=False, repr=False)
    """
    Useful when this expression is a builder that wires multiple components together to create the final expression.
    This defaults to Self.
    """
    _limit_: Optional[int] = field(init=False, repr=False, default=None)
    """
    The maximum number of results to return during evaluation.
    """

    def __post_init__(self):
        self._expression_ = self

    @lru_cache
    def _get_expression_by_id_(self, id_: uuid.UUID) -> SymbolicExpression:
        try:
            return next(
                expression
                for expression in self._all_expressions_
                if expression._id_ == id_
            )
        except StopIteration:
            raise NoExpressionFoundForGivenID(self, id_)

    @property
    def _is_true_(self) -> bool:
        """
        :return: Whether this expression evaluates to True.
        """
        return not self._is_false__

    @property
    def _is_false_(self) -> bool:
        """
        :return: Whether this expression evaluates to False.
        """
        return self._is_false__

    @_is_false_.setter
    def _is_false_(self, value: bool):
        """
        Set the current truth value of an evaluation result for this expression.
        """
        self._is_false__ = value

    def tolist(self):
        """
        Evaluate and return the results as a list.
        """
        return make_list(self.evaluate())

    def first(self):
        """
        Evaluate and return the first result of the query object descriptor.
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
        from .variable import Literal

        children = [
            v if isinstance(v, SymbolicExpression) else Literal(_value_=v)
            for v in children
        ]
        for v in children:
            v._parent_ = self
        return tuple(v._expression_ for v in children)

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
        sources: Optional[Bindings | OperationResult] = None,
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
            if isinstance(sources, OperationResult):
                sources = sources.bindings
            sources = copy(sources) if sources is not None else {}
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
        for conclusion in self._conclusions_:
            current_result.bindings = next(
                conclusion._evaluate_(current_result.bindings, parent=self)
            ).bindings
        return current_result

    @cached_property
    def _binding_id_(self) -> uuid.UUID:
        """
        The binding id is the id used in the bindings (the results dictionary of operations). It is sometimes different
        from the id of the symbolic expression itself because some operations do not have results themselves, but their
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
        """
        pass

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
        from .variable import Variable

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

    def __and__(self, other):
        from ..operators.core_logical_operators import AND

        return AND(self, other)

    def __or__(self, other):
        from ..operators.core_logical_operators import OR

        return OR(self, other)

    def _invert_(self):
        """
        Invert the symbolic expression.
        """
        from ..operators.core_logical_operators import Not

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

    @property
    def _binding_id_(self) -> int:
        return self._original_expression_._binding_id_

    @property
    def _is_false_(self) -> bool:
        return self._original_expression_._is_false_

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
    is_false: bool
    """
    Whether the operation resulted in a false value (i.e., The operation condition was not satisfied)
    """
    operand: SymbolicExpression
    """
    The operand that produced the result.
    """
    previous_operation_result: Optional[OperationResult] = None
    """
    The result of the operation that was evaluated before this one.
    """

    @property
    def all_bindings(self) -> Bindings:
        """
        :return: All the bindings from all the evaluated operations until this one, including this one.
        """
        if (
            self.previous_operation_result is None
            or self.previous_operation_result.bindings is self.bindings
        ):
            return self.bindings
        return self.previous_operation_result.bindings | self.bindings

    @property
    def has_value(self) -> bool:
        return self.operand._binding_id_ in self.bindings

    @property
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
        key = self._id_expression_map_[key._binding_id_]
        return super().__getitem__(key)

    @cached_property
    def _id_expression_map_(self) -> Dict[int, Selectable[T]]:
        return {key._binding_id_: key for key in self.data.keys()}


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

    def _build_operation_result_and_update_truth_value_(
        self, bindings: Bindings, child_result: Optional[OperationResult] = None
    ) -> OperationResult:
        """
        Build an OperationResult instance and update the truth value based on the bindings.

        :param bindings: The bindings of the result.
        :param child_result: The result of the child operation, if any.
        :return: The OperationResult instance with an updated truth value.
        """
        self._update_truth_value_(bindings[self._binding_id_])
        return OperationResult(bindings, self._is_false_, self, child_result)

    def _update_truth_value_(self, current_value: Any) -> None:
        """
        Updates the truth value of the variable based on the current value.

        :param current_value: The current value of the variable.
        """
        # Calculating the truth value is not always done for efficiency. The truth value is updated only when this
        # operation is a child of a TruthValueOperator.
        if not isinstance(self._parent_, TruthValueOperator):
            return
        is_true = (
            len(current_value) > 0
            if is_iterable(current_value)
            else bool(current_value)
        )
        self._is_false_ = not is_true

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

    @cached_property
    def _name_(self):
        if self._type_:
            return self._type_.__name__
        else:
            return self.__class__.__name__
