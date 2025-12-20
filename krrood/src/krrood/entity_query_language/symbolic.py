"""
Core symbolic expression system used to build and evaluate entity queries.

This module defines the symbolic types (variables, sources, logical and
comparison operators) and the evaluation mechanics.
"""

from __future__ import annotations

import operator
import typing
from abc import abstractmethod, ABC
from collections import UserDict
from copy import copy
from dataclasses import dataclass, field, fields, MISSING, is_dataclass
from functools import lru_cache, cached_property

from krrood.entity_query_language.failures import VariableCannotBeEvaluated
from typing_extensions import (
    Iterable,
    Any,
    Optional,
    Type,
    Dict,
    ClassVar,
    Union as TypingUnion,
    Generic,
    TYPE_CHECKING,
    List,
    Tuple,
    Callable,
    Self,
    Set,
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
    InvalidEntityType,
    UnSupportedOperand,
    NonPositiveLimitValue,
    InvalidChildType,
    CannotProcessResultOfGivenChildType,
    LiteralConditionError,
)
from .result_quantification_constraint import (
    ResultQuantificationConstraint,
    Exactly,
)
from .rxnode import RWXNode, ColorLegend
from .symbol_graph import SymbolGraph
from .utils import (
    IDGenerator,
    is_iterable,
    generate_combinations,
    make_list,
    make_set,
    T,
)
from ..class_diagrams import ClassRelation
from ..class_diagrams.class_diagram import Association, WrappedClass
from ..class_diagrams.failures import ClassIsUnMappedInClassDiagram
from ..class_diagrams.wrapped_field import WrappedField

if TYPE_CHECKING:
    from .conclusion import Conclusion
    from .entity import ConditionType

id_generator = IDGenerator()

RWXNode.enclosed_name = "Selected Variable"


@dataclass
class OperationResult:
    """
    A data structure that carries information about the result of an operation in EQL.
    """

    bindings: Dict[int, Any]
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
    def is_true(self):
        return not self.is_false

    @property
    def value(self) -> Optional[Any]:
        return self.bindings.get(self.operand._id_, None)

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
class SymbolicExpression(Generic[T], ABC):
    """
    Base class for all symbolic expressions.

    Symbolic expressions form a tree and are evaluated lazily to produce
    bindings for variables, subject to logical constraints.

    :ivar _child_: Optional child expression.
    :ivar _id_: Unique identifier of this node.
    :ivar _node_: Backing anytree.Node for visualization and traversal.
    :ivar _conclusion_: Set of conclusion actions attached to this node.
    :ivar _is_false_: Internal flag indicating evaluation result for this node.
    """

    _child_: Optional[SymbolicExpression] = field(init=False)
    _id_: int = field(init=False, repr=False, default=None)
    _node_: RWXNode = field(init=False, default=None, repr=False)
    _id_expression_map_: ClassVar[Dict[int, SymbolicExpression]] = {}
    _conclusion_: typing.Set[Conclusion] = field(init=False, default_factory=set)
    _symbolic_expression_stack_: ClassVar[List[SymbolicExpression]] = []
    _is_false_: bool = field(init=False, repr=False, default=False)
    _eval_parent_: Optional[SymbolicExpression] = field(
        default=None, init=False, repr=False
    )
    _plot_color__: Optional[ColorLegend] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self._id_:
            self._id_ = id_generator(self)
            self._create_node_()
            self._id_expression_map_[self._id_] = self
        if hasattr(self, "_child_") and self._child_ is not None:
            self._update_child_()

    def _update_child_(self, child: Optional[SymbolicExpression] = None):
        child = child or self._child_
        self._child_ = self._update_children_(child)[0]

    def _update_children_(
        self, *children: SymbolicExpression
    ) -> Tuple[SymbolicExpression, ...]:
        children: Dict[int, SymbolicExpression] = dict(enumerate(children))
        for k, v in children.items():
            if not isinstance(v, SymbolicExpression):
                children[k] = Literal(v)
        for k, v in children.items():
            # With graph structure, do not copy nodes; just connect an edge.
            v._node_.parent = self._node_
        return tuple(children.values())

    def _create_node_(self):
        self._node_ = RWXNode(self._name_, data=self, color=self._plot_color_)

    def _process_result_(
        self, result: OperationResult
    ) -> TypingUnion[T, UnificationDict]:
        """
        Map the result to the correct output data structure for user usage. This returns the selected variables only.
        This method should be implemented by subclasses that can be children of a ResultProcessor.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        raise CannotProcessResultOfGivenChildType(type(self))

    @abstractmethod
    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the symbolic expression and set the operands indices.
        This method should be implemented by subclasses.
        """
        pass

    def _add_conclusion_(self, conclusion: Conclusion):
        self._conclusion_.add(conclusion)

    @property
    def _parent_(self) -> Optional[SymbolicExpression]:
        if self._eval_parent_ is not None:
            return self._eval_parent_
        elif self._node_.parent is not None:
            return self._node_.parent.data
        return None

    @_parent_.setter
    def _parent_(self, value: Optional[SymbolicExpression]):
        self._node_.parent = value._node_ if value is not None else None
        if value is not None and hasattr(value, "_child_"):
            value._child_ = self

    @cached_property
    def _conditions_root_(self) -> SymbolicExpression:
        """
        Get the root of the symbolic expression tree that contains conditions.
        """
        conditions_root = self._root_
        while conditions_root._child_ is not None:
            conditions_root = conditions_root._child_
            if isinstance(conditions_root._parent_, QueryObjectDescriptor):
                break
        return conditions_root

    @property
    def _root_(self) -> SymbolicExpression:
        """
        Get the root of the symbolic expression tree.
        """
        return self._node_.root.data

    @property
    @abstractmethod
    def _name_(self) -> str:
        pass

    @property
    def _all_nodes_(self) -> List[SymbolicExpression]:
        return [self] + self._descendants_

    @property
    def _descendants_(self) -> List[SymbolicExpression]:
        return [d.data for d in self._node_.descendants]

    @property
    def _children_(self) -> List[SymbolicExpression]:
        return [c.data for c in self._node_.children]

    @classmethod
    def _current_parent_(cls) -> Optional[SymbolicExpression]:
        if cls._symbolic_expression_stack_:
            return cls._symbolic_expression_stack_[-1]
        return None

    @property
    def _sources_(self):
        vars = [v.data for v in self._node_.leaves]
        while any(isinstance(v, SymbolicExpression) for v in vars):
            vars = {
                (
                    v._domain_source_
                    if isinstance(v, Variable) and v._domain_source_ is not None
                    else v
                )
                for v in vars
            }
            for v in copy(vars):
                if isinstance(v, SymbolicExpression):
                    vars.remove(v)
                    vars.update(set(v._all_variable_instances_))
        return set(vars)

    @cached_property
    def _unique_variables_(self) -> Set[Variable]:
        return make_set(self._all_variable_instances_)

    @cached_property
    @abstractmethod
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        ...

    @property
    def _plot_color_(self) -> ColorLegend:
        return self._plot_color__

    @_plot_color_.setter
    def _plot_color_(self, value: ColorLegend):
        self._plot_color__ = value
        self._node_.color = value

    def __and__(self, other):
        return AND(self, other)

    def __or__(self, other):
        return optimize_or(self, other)

    def _invert_(self):
        return Not(self)

    def __enter__(self) -> Self:
        node = self
        if (node is self._root_) or (node._parent_ is self._root_):
            node = node._conditions_root_
        SymbolicExpression._symbolic_expression_stack_.append(node)
        return self

    def __exit__(self, *args):
        SymbolicExpression._symbolic_expression_stack_.pop()

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return self._name_


ResultMapping = Callable[[Iterable[Dict[int, Any]]], Iterable[Dict[int, Any]]]
"""
A function that maps the results of a query object descriptor to a new set of results.
"""


@dataclass(eq=False, repr=False)
class Selectable(SymbolicExpression[T], ABC):
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
    def _type__(self):
        return self._var_._type_ if self._var_ else None

    def _process_result_(self, result: OperationResult) -> T:
        """
        Map the result to the correct output data structure for user usage.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        return result[result.operand._id_]

    @property
    def _is_iterable_(self):
        """
        Whether the selectable is iterable.

        :return: True if the selectable is iterable, False otherwise.
        """
        if self._var_ and self._var_ is not self:
            return self._var_._is_iterable_
        return False


@dataclass(eq=False, repr=False)
class CanBehaveLikeAVariable(Selectable[T], ABC):
    """
    This class adds the monitoring/tracking behavior on variables that tracks attribute access, calling,
    and comparison operations.
    """

    _path_: List[ClassRelation] = field(init=False, default_factory=list)
    """
    The path of the variable in the symbol graph as a sequence of relation instances.
    """
    _attributes_: Dict[str, Attribute[T]] = field(init=False, default_factory=dict)
    """
    A storage of created symbolic attributes to prevent recreating same attribute multiple times.
    """

    def __getattr__(self, name: str) -> CanBehaveLikeAVariable[T]:
        # Prevent debugger/private attribute lookups from being interpreted as symbolic attributes
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )
        if name in self._attributes_:
            return self._attributes_[name]
        attr = Attribute(self, name, self._type__)
        self._attributes_[name] = attr
        return attr

    def __getitem__(self, key) -> CanBehaveLikeAVariable[T]:
        return Index(self, key)

    def __call__(self, *args, **kwargs) -> CanBehaveLikeAVariable[T]:
        return Call(self, args, kwargs)

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
class ResultProcessor(CanBehaveLikeAVariable[T], ABC):
    """
    Base class for processors that return concrete results from queries, including quantifiers
    (e.g., An, The) and aggregators (e.g., Count, Sum, Max, Min).
    """

    _child_: SymbolicExpression[T]

    def __post_init__(self):
        super().__post_init__()
        self._var_ = (
            self._child_._var_ if isinstance(self._child_, Selectable) else None
        )
        self._node_.wrap_subtree = True

    @cached_property
    def _type_(self):
        if self._var_:
            return self._var_._type_
        else:
            raise ValueError("No type available as _var_ is None")

    @property
    def _name_(self) -> str:
        return f"{self.__class__.__name__}()"

    def evaluate(
        self,
    ) -> Iterable[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression[T]], T]]]:
        """
        Evaluate the query and map the results to the correct output data structure.
        This is the exposed evaluation method for users.
        """
        SymbolGraph().remove_dead_instances()
        yield from map(self._child_._process_result_, self._evaluate__())

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_

    def _invert_(self):
        raise UnsupportedNegation(self.__class__)

    def visualize(
        self,
        figsize=(35, 30),
        node_size=7000,
        font_size=25,
        spacing_x: float = 4,
        spacing_y: float = 4,
        layout: str = "tidy",
        edge_style: str = "orthogonal",
        label_max_chars_per_line: Optional[int] = 13,
    ):
        """
        Visualize the query graph, for arguments' documentation see `rustworkx_utils.RWXNode.visualize`.
        """
        self._node_.visualize(
            figsize=figsize,
            node_size=node_size,
            font_size=font_size,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            layout=layout,
            edge_style=edge_style,
            label_max_chars_per_line=label_max_chars_per_line,
        )


@dataclass(eq=False, repr=False)
class Aggregator(ResultProcessor[T], ABC):
    _default_value_: T = field(kw_only=True, default=None)
    """
    The default value to be returned if the child results are empty.
    """

    def evaluate(
        self,
    ) -> Iterable[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression[T]], T]]]:
        """
        Evaluate the query and map the results to the correct output data structure.
        This is the exposed evaluation method for users.
        """
        return list(super().evaluate())[0]

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        sources = sources or {}
        if self._id_ in sources:
            yield OperationResult(sources, False, self)
            return

        values = self._apply_aggregation_function_(self._child_._evaluate__(sources))
        if values:
            yield OperationResult(values, False, self)
        else:
            yield OperationResult({self._id_: self._default_value_}, False, self)

    @abstractmethod
    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Dict[int, Any]:
        """
        Apply the aggregation function to the results of the child.

        :param child_results: The results of the child.
        """
        ...

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("Aggregator", "#F54927")

    @_plot_color_.setter
    def _plot_color_(self, value: ColorLegend):
        self._plot_color__ = value
        self._node_.color = value


@dataclass(eq=False, repr=False)
class Count(Aggregator[T]):
    """
    Count the number of child results.
    """

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Dict[int, Any]:
        return {self._id_: len(list(child_results))}


@dataclass
class EntityAggregator(Aggregator[T], ABC):
    _child_: Selectable[T]
    """
    The child entity to be aggregated.
    """
    _key_func_: Callable = field(kw_only=True, default=lambda x: x)
    """
    An optional function that extracts the value to be used in the aggregation.
    """

    def __post_init__(self):
        if not isinstance(self._child_, Selectable):
            raise InvalidChildType(type(self._child_), [Selectable])
        super().__post_init__()

    def _get_child_value_from_result_(self, result: OperationResult) -> Any:
        """
        Extract the value of the child from the result dictionary.
         In addition, it applies the key function if given.
        """
        value = result[self._child_._var_._id_]
        if self._key_func_:
            return self._key_func_(value)
        return value


@dataclass(eq=False, repr=False)
class Sum(EntityAggregator[T]):
    """
    Calculate the sum of the child results. If given, make use of the key function to extract the value to be summed.
    """

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Dict[int, Any]:
        entered = False
        sum_val = 0
        for val in map(self._get_child_value_from_result_, child_results):
            entered = True
            sum_val += val
        if entered:
            return {self._id_: sum_val}
        return {}


@dataclass(eq=False, repr=False)
class Average(EntityAggregator[T]):
    """
    Calculate the average of the child results. If given, make use of the key function to extract the value to be
     averaged.
    """

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Dict[int, Any]:
        sum_val = 0
        count = 0
        for val in map(self._get_child_value_from_result_, child_results):
            sum_val += val
            count += 1
        if count:
            return {self._id_: sum_val / count}
        return {}


@dataclass(eq=False, repr=False)
class Extreme(EntityAggregator[T], ABC):
    """
    Find and return the extreme value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Dict[int, Any]:
        try:
            bindings_with_extreme_val = self._extreme_function_(
                child_results, key=self._get_child_value_from_result_
            ).bindings
            bindings_with_extreme_val[self._id_] = bindings_with_extreme_val[
                self._child_._var_._id_
            ]
            return bindings_with_extreme_val
        except ValueError:
            # Means that the child results were empty, so do not return any results,
            # the default value will be returned instead (see Aggregator._evaluate__)
            return {}

    @property
    @abstractmethod
    def _extreme_function_(self) -> Callable: ...


@dataclass(eq=False, repr=False)
class Max(Extreme[T]):
    """
    Find and return the maximum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    @property
    def _extreme_function_(self) -> Callable[[Any], Any]:
        return max


@dataclass(eq=False, repr=False)
class Min(Extreme[T]):
    """
    Find and return the minimum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    @property
    def _extreme_function_(self) -> Callable[[Any], Any]:
        return min


@dataclass(eq=False)
class ResultQuantifier(ResultProcessor[T], ABC):
    """
    Base for quantifiers that return concrete results from entity/set queries
    (e.g., An, The).
    """

    _child_: QueryObjectDescriptor[T]
    """
    A child of a result quantifier. It must be a QueryObjectDescriptor.
    """
    _quantification_constraint_: Optional[ResultQuantificationConstraint] = None
    """
    The quantification constraint that must be satisfied by the result quantifier if present.
    """

    def __post_init__(self):
        if not isinstance(self._child_, QueryObjectDescriptor):
            raise InvalidEntityType(type(self._child_), [QueryObjectDescriptor])
        super().__post_init__()

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[T]:
        sources = sources or {}
        self._eval_parent_ = parent
        if self._id_ in sources:
            yield OperationResult(sources, False, self)
            return
        result_count = 0
        values = self._child_._evaluate__(sources, parent=self)
        for value in values:
            result_count += 1
            self._assert_satisfaction_of_quantification_constraints_(
                result_count, done=False
            )
            if self._var_:
                value[self._id_] = value[self._var_._id_]
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

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        if self._quantification_constraint_:
            name += f"({self._quantification_constraint_})"
        return name

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("ResultQuantifier", "#9467bd")

    @_plot_color_.setter
    def _plot_color_(self, value: ColorLegend):
        self._plot_color__ = value
        self._node_.color = value


class UnificationDict(UserDict):
    """
    A dictionary which maps all expressions that are on a single variable to the original variable id.
    """

    def __getitem__(self, key: Selectable[T]) -> T:
        key = key._id_expression_map_[key._var_._id_]
        return super().__getitem__(key)


@dataclass(eq=False, repr=False)
class An(ResultQuantifier[T]):
    """Quantifier that yields all matching results one by one."""

    def evaluate(
        self,
        limit: Optional[int] = None,
    ) -> Iterable[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression[T]], T]]]:
        """
        Evaluate the query and map the results to the correct output data structure.
        This is the exposed evaluation method for users.
        """
        results = super().evaluate()
        if limit is None:
            yield from results
        elif not isinstance(limit, int) or limit <= 0:
            raise NonPositiveLimitValue(limit)
        else:
            for res_num, result in enumerate(results, 1):
                yield result
                if res_num == limit:
                    return


@dataclass(eq=False, repr=False)
class The(ResultQuantifier[T]):
    """
    Quantifier that expects exactly one result; raises MultipleSolutionFound if more.
    """

    _quantification_constraint_: ResultQuantificationConstraint = field(
        init=False, default_factory=lambda: Exactly(1)
    )

    def evaluate(
        self,
    ) -> TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression[T]], T]]:
        return list(super().evaluate())[0]

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression[T]], T]]]:
        try:
            yield from super()._evaluate__(sources, parent=parent)
        except LessThanExpectedNumberOfSolutions:
            raise NoSolutionFound(self)
        except GreaterThanExpectedNumberOfSolutions:
            raise MultipleSolutionFound(self)


@dataclass(frozen=True)
class OrderByParams:
    """
    Parameters for ordering the results of a query object descriptor.
    """

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


@dataclass(eq=False, repr=False)
class QueryObjectDescriptor(SymbolicExpression[T], ABC):
    """
    Describes the queried object(s), could be a query over a single variable or a set of variables,
    also describes the condition(s)/properties of the queried object(s).
    """

    _child_: Optional[SymbolicExpression[T]] = field(default=None)
    """
    The child of the query object descriptor is the root of the conditions in the query/sub-query graph.
    """
    _selected_variables: List[Selectable[T]] = field(default_factory=list)
    """
    The variables that are selected by the query object descriptor.
    """
    _order_by: Optional[OrderByParams] = field(default=None, init=False)
    """
    Parameters for ordering the results of the query object descriptor.
    """
    _results_mapping: List[ResultMapping] = field(init=False, default_factory=list)
    """
    Mapping functions that map the results of the query object descriptor to a new set of results.
    """

    def __post_init__(self):
        super().__post_init__()
        for variable in self._selected_variables:
            variable._var_._node_.enclosed = True

    def where(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query object descriptor.
        """
        condition_list = list(conditions)

        # If there are no conditions raise error.
        if len(condition_list) == 0:
            raise ValueError("No conditions provided")

        # If there's a constant condition raise error.
        literal_expressions = [
            exp for exp in condition_list if not isinstance(exp, SymbolicExpression)
        ]
        if literal_expressions:
            raise LiteralConditionError(literal_expressions)

        # Build the expression from the conditions
        expression = (
            chained_logic(AND, *condition_list)
            if len(condition_list) > 1
            else condition_list[0]
        )

        # set the child of the query object descriptor to the expression and return self.
        self._update_child_(expression)
        return self

    def order_by(
        self,
        variable: Selectable,
        descending: bool = False,
        key: Optional[Callable] = None,
    ) -> Self:
        """
        Order the results by the given variable, using the given key function in descending or ascending order.

        :param variable: The variable to order by.
        :param descending: Whether to order the results in descending order.
        :param key: A function to extract the key from the variable value.
        """
        self._order_by = OrderByParams(variable, descending, key)
        return self

    def _order(
        self, results: Iterable[Dict[int, Any]] = None
    ) -> Iterable[Dict[int, Any]]:
        """
        Order the results by the given order variable.

        :param results: The results to be ordered.
        :return: The ordered results.
        """

        def key(result: Dict[int, Any]) -> Any:
            variable_value = result[self._order_by.variable._var_._id_]
            if self._order_by.key:
                return self._order_by.key(variable_value)
            else:
                return variable_value

        results = sorted(
            results,
            key=key,
            reverse=self._order_by.descending,
        )
        return results

    def distinct(
        self,
        *on: Selectable[T],
    ) -> Self:
        """
        Apply distinctness constraint to the query object descriptor results.

        :param on: The variables to be used for distinctness.
        :return: This query object descriptor.
        """
        on_ids = tuple([v._var_._id_ for v in on]) if on else tuple([v._var_._id_ for v in self._selected_variables])
        seen_results = SeenSet(keys=on_ids)

        def get_distinct_results(
            results_gen: Iterable[Dict[int, Any]],
        ) -> Iterable[Dict[int, Any]]:
            for res in results_gen:
                bindings = (
                    res if not on else {k: v for k, v in res.items() if k in on_ids}
                )
                if seen_results.check(bindings):
                    continue
                yield res
                seen_results.add(bindings)

        self._results_mapping.append(get_distinct_results)
        return self

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        sources = sources or {}
        self._eval_parent_ = parent
        for values in self.get_constrained_values(sources):
            self.evaluate_conclusions_and_update_bindings(values)
            if self.any_selected_variable_is_inferred_and_unbound(values):
                continue
            selected_vars_bindings = self._evaluate_selected_variables(values.bindings)
            for result in self._apply_results_mapping(selected_vars_bindings):
                yield OperationResult({**values.bindings, **result}, False, self)

    @staticmethod
    def variable_is_inferred(var: CanBehaveLikeAVariable[T]) -> bool:
        """
        Whether the variable is inferred or not.

        :param var: The variable.
        :return: True if the variable is inferred, otherwise False.
        """
        return isinstance(var, Variable) and var._is_inferred_

    def any_selected_variable_is_inferred_and_unbound(
        self, values: OperationResult
    ) -> bool:
        """
        Check if any of the selected variables is inferred and is not bound.

        :param values: The current result with the current bindings.
        :return: True if any of the selected variables is inferred and is not bound, otherwise False.
        """
        return any(
            not self.variable_is_bound_or_its_children_are_bound(var, values)
            for var in self._selected_variables
            if self.variable_is_inferred(var)
        )

    @lru_cache(maxsize=None)
    def variable_is_bound_or_its_children_are_bound(
        self, var: CanBehaveLikeAVariable[T], result: OperationResult
    ) -> bool:
        """
        Whether the variable is directly bound or all its children are bound.

        :param var: The variable.
        :param result: The current result containing the current bindings.
        :return: True if the variable is bound, otherwise False.
        """
        if var._id_ in result:
            return True
        unique_vars = [uv for uv in var._unique_variables_ if uv is not var]
        if unique_vars and all(
            self.variable_is_bound_or_its_children_are_bound(uv, result)
            for uv in unique_vars
        ):
            return True
        return False

    def evaluate_conclusions_and_update_bindings(self, child_result: OperationResult):
        """
        Update the bindings of the results by evaluating the conclusions using the received bindings from the child as
        sources.

        :param child_result: The result of the child operation.
        """
        if not self._child_:
            return
        for conclusion in self._child_._conclusion_:
            child_result.bindings = next(
                iter(conclusion._evaluate__(child_result.bindings, parent=self))
            ).bindings

    def get_constrained_values(
        self, sources: Optional[Dict[int, Any]]
    ) -> Iterable[OperationResult]:
        """
        Evaluate the child (i.e., the conditions that constrain the domain of the selected variables).

        :param sources: The current bindings.
        :return: The bindings after applying the constraints of the child.
        """
        if self._child_:
            # QueryObjectDescriptor does not yield when it's False
            yield from (
                res
                for res in self._child_._evaluate__(sources, parent=self)
                if res.is_true
            )
        else:
            yield from [OperationResult(sources, False, self)]

    def _evaluate_selected_variables(
        self, sources: Dict[int, Any]
    ) -> Iterable[Dict[int, Any]]:
        """
        Evaluate the selected variables by generating combinations of values from their evaluation generators.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each combination of values.
        """
        var_val_gen = {
            var: var._evaluate__(copy(sources), parent=self)
            for var in self._selected_variables
        }
        for sol in generate_combinations(var_val_gen):
            yield {var._id_: sol[var][var._id_] for var in self._selected_variables}

    def _apply_results_mapping(
        self, results: Iterable[Dict[int, Any]]
    ) -> Iterable[Dict[int, Any]]:
        """
        Process and transform an iterable of results based on predefined mappings and ordering.

        This method applies a sequence of result transformations defined in the instance,
        using a series of mappings to modify the results.

        :param results: An iterable containing dictionaries that represent the initial result set to be transformed.
        :return: An iterable containing dictionaries that represent the transformed data.
        """
        for result_mapping in self._results_mapping:
            results = result_mapping(results)
        if self._order_by:
            results = self._order(results)
        return results

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        vars = []
        if self._selected_variables:
            vars.extend(self._selected_variables)
        if self._child_:
            vars.extend(self._child_._all_variable_instances_)
        return vars

    def _invert_(self):
        raise UnsupportedNegation(self.__class__)

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("ObjectDescriptor", "#d62728")

    @property
    def _name_(self) -> str:
        return f"({', '.join(var._name_ for var in self._selected_variables)})"


@dataclass(eq=False, repr=False)
class SetOf(QueryObjectDescriptor[T]):
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
        selected_variables_ids = [v._id_ for v in self._selected_variables]
        return UnificationDict(
            {v._var_: result[v._id_] for v in self._selected_variables}
        )


@dataclass(eq=False, repr=False)
class Entity(QueryObjectDescriptor[T], Selectable[T]):
    """
    A query over a single variable.
    """

    def __post_init__(self):
        self._var_ = self.selected_variable
        super().__post_init__()

    @property
    def selected_variable(self):
        return self._selected_variables[0] if self._selected_variables else None


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

        super().__post_init__()

        # has to be after super init because this needs the node of this variable to be initialized first.
        self._update_child_vars_from_kwargs_()

    def _update_domain_(self, domain):
        """
        Set the domain and ensure it is a lazy re-enterable iterable.
        """
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

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        """
        A variable either is already bound in sources by other constraints (Symbolic Expressions).,
        or will yield from current domain if exists,
        or has no domain and will instantiate new values by constructing the type if the type is given.
        """
        self._eval_parent_ = parent
        sources = sources or {}
        if self._id_ in sources:
            yield self._build_operation_result_and_update_truth_value_(sources)
        elif self._domain_:
            yield from self._iterator_over_domain_values_(sources)
        elif self._is_inferred_ or self._predicate_type_:
            yield from self._instantiate_using_child_vars_and_yield_results_(sources)
        else:
            raise VariableCannotBeEvaluated(self)

    def _iterator_over_domain_values_(self, sources: Dict[int, Any]) -> Iterable[OperationResult]:
        """
        Iterate over the values in the variable's domain, yielding OperationResult instances.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        if isinstance(self._domain_, CanBehaveLikeAVariable):
            yield from self._iterator_over_variable_domain_values_(sources)
        else:
            yield from self._iterator_over_iterable_domain_values_(sources)

    def _iterator_over_variable_domain_values_(self, sources: Dict[int, Any]):
        """
        Iterate over the values in the variable's domain, where the domain is another variable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for domain in self._domain_._evaluate__(sources, parent=self):
            for v in domain.value:
                bindings = {**sources, **domain.bindings, self._id_: v}
                yield self._build_operation_result_and_update_truth_value_(bindings)

    def _iterator_over_iterable_domain_values_(self, sources: Dict[int, Any]):
        """
        Iterate over the values in the variable's domain, where the domain is an iterable.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each value in the domain.
        """
        for v in self._domain_:
            bindings = {**sources, self._id_: v}
            yield self._build_operation_result_and_update_truth_value_(bindings)

    def _instantiate_using_child_vars_and_yield_results_(
        self, sources: Dict[int, Any]
    ) -> Iterable[OperationResult]:
        """
        Create new instances of the variable type and using as keyword arguments the child variables values.
        """
        for kwargs in self._generate_combinations_for_child_vars_values_(sources):
            # Build once: unwrapped hashed kwargs for already provided child vars
            bound_kwargs = {k: v[self._child_vars_[k]._id_] for k, v in kwargs.items()}
            instance = self._type_(**bound_kwargs)
            yield self._process_output_and_update_values_(instance, kwargs)

    def _generate_combinations_for_child_vars_values_(
        self, sources: Optional[Dict[int, Any]] = None
    ):
        yield from generate_combinations(
            {k: var._evaluate__(sources) for k, var in self._child_vars_.items()}
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
        values = {self._id_: instance}
        for d in kwargs.values():
            values.update(d.bindings)
        return self._build_operation_result_and_update_truth_value_(values)

    def _build_operation_result_and_update_truth_value_(self, bindings: Dict[int, Any]) -> OperationResult:
        """
        Build an OperationResult instance and update the truth value based on the bindings.

        :param bindings: The bindings of the result.
        :return: The OperationResult instance with updated truth value.
        """
        if isinstance(self._parent_, LogicalOperator) or (self is self._conditions_root_):
            self._is_false_ = not bool(bindings[self._id_])
        else:
            self._is_false_ = False
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

    @property
    def _plot_color_(self) -> ColorLegend:
        if self._plot_color__:
            return self._plot_color__
        else:
            return ColorLegend("Variable", "cornflowerblue")

    @_plot_color_.setter
    def _plot_color_(self, value: ColorLegend):
        self._plot_color__ = value
        self._node_.color = value


@dataclass(eq=False, init=False, repr=False)
class Literal(Variable[T]):
    """
    Literals are variables that are not constructed by their type but by their given data.
    """

    def __init__(
        self, data: Any, name: Optional[str] = None, type_: Optional[Type] = None
    ):
        original_data = data
        if not isinstance(data, CanBehaveLikeAVariable):
            data = [data]
        if not type_:
            original_data_lst = make_list(original_data)
            first_value = original_data_lst[0] if len(original_data_lst) > 0 else None
            type_ = type(first_value) if first_value else None
        if name is None:
            if type_:
                name = type_.__name__
            else:
                if isinstance(data, CanBehaveLikeAVariable):
                    name = data._name_
                else:
                    name = type(original_data).__name__
        super().__init__(_name__=name, _type_=type_, _domain_source_=data)

    @property
    def _plot_color_(self) -> ColorLegend:
        if self._plot_color__:
            return self._plot_color__
        else:
            return ColorLegend("Literal", "#949292")


@dataclass(eq=False, repr=False)
class DomainMapping(CanBehaveLikeAVariable[T], ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """

    _child_: Selectable[T]

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
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        """
        Apply the domain mapping to the child's values.
        """

        sources = sources or {}

        self._eval_parent_ = parent

        if self._id_ in sources:
            yield OperationResult(sources, self._is_false_, self)
            return

        yield from (
            self._build_operation_result_and_update_truth_value_(
                child_result, mapped_value
            )
            for child_result in self._child_._evaluate__(sources, parent=self)
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
        if isinstance(self._parent_, LogicalOperator) or self is self._conditions_root_:
            self._is_false_ = not bool(current_value)
        return OperationResult(
            {**child_result.bindings, self._id_: current_value},
            self._is_false_,
            self,
        )

    @abstractmethod
    def _apply_mapping_(self, value: Any) -> Iterable[Any]:
        """
        Apply the domain mapping to a symbolic value.
        """
        pass

    @property
    def _plot_color_(self) -> ColorLegend:
        if self._plot_color__:
            return self._plot_color__
        else:
            return ColorLegend("DomainMapping", "#8FC7B8")

    @_plot_color_.setter
    def _plot_color_(self, value: ColorLegend):
        self._plot_color__ = value
        self._node_.color = value


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
    _kwargs_: Dict[str, Any] = field(default_factory=dict)

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

    def __post_init__(self):
        if not isinstance(self._child_, SymbolicExpression):
            self._child_ = Literal(self._child_)
        super().__post_init__()
        self._path_ = self._child_._path_

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


@dataclass(eq=False, repr=False)
class BinaryOperator(SymbolicExpression, ABC):
    """
    A base class for binary operators that can be used to combine symbolic expressions.
    """

    left: SymbolicExpression
    right: SymbolicExpression
    _child_: SymbolicExpression = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self.left, self.right = self._update_children_(self.left, self.right)

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        return self.left._all_variable_instances_ + self.right._all_variable_instances_


def not_contains(container, item) -> bool:
    """
    The inverted contains operation.

    :param container: The container.
    :param item: The item to test if contained in the container.
    :return:
    """
    return not operator.contains(container, item)


@dataclass(eq=False, repr=False)
class Comparator(BinaryOperator):
    """
    A symbolic equality check that can be used to compare symbolic variables using a provided comparison operation.
    """

    left: CanBehaveLikeAVariable
    right: CanBehaveLikeAVariable
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
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        """
        Compares the left and right symbolic variables using the "operation".
        """
        sources = sources or {}
        self._eval_parent_ = parent

        if self._id_ in sources:
            yield OperationResult(sources, self._is_false_, self)
            return

        first_operand, second_operand = self.get_first_second_operands(sources)

        yield from (
            OperationResult(
                second_val.bindings, not self.apply_operation(second_val), self
            )
            for first_val in first_operand._evaluate__(sources, parent=self)
            if first_val.is_true
            for second_val in second_operand._evaluate__(
                first_val.bindings, parent=self
            )
            if second_val.is_true
        )

    def apply_operation(self, operand_values: OperationResult) -> bool:
        left_value, right_value = (
            operand_values.bindings[self.left._id_],
            operand_values.bindings[self.right._id_],
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
        self, sources: Dict[int, Any]
    ) -> Tuple[SymbolicExpression, SymbolicExpression]:
        left_has_the = any(isinstance(desc, The) for desc in self.left._descendants_)
        right_has_the = any(isinstance(desc, The) for desc in self.right._descendants_)
        if left_has_the and not right_has_the:
            return self.left, self.right
        elif not left_has_the and right_has_the:
            return self.right, self.left
        if sources and any(
            v._var_._id_ in sources for v in self.right._unique_variables_
        ):
            return self.right, self.left
        else:
            return self.left, self.right

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("Comparator", "#ff7f0e")


@dataclass(eq=False, repr=False)
class LogicalOperator(SymbolicExpression[T], ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions using logical constraints on their
    truth values. Examples are conjunction (AND), disjunction (OR), negation (NOT), and conditional quantification
    (ForALL, Exists).
    """

    @property
    def _name_(self):
        return self.__class__.__name__

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("LogicalOperator", "#2ca02c")


@dataclass(eq=False, repr=False)
class Not(LogicalOperator[T]):
    """
    The logical negation of a symbolic expression. Its truth value is the opposite of its child's truth value. This is
    used when you want bindings that satisfy the negated condition (i.e., that doesn't satisfy the original condition).
    """

    _child_: SymbolicExpression[T]

    def __post_init__(self):
        if isinstance(self._child_, ResultQuantifier):
            raise UnSupportedOperand(self.__class__, self._child_)
        super().__post_init__()

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        sources = sources or {}
        self._eval_parent_ = parent
        for v in self._child_._evaluate__(sources, parent=self):
            self._is_false_ = v.is_true
            yield OperationResult(v.bindings, self._is_false_, self)

    @property
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_


@dataclass(eq=False, repr=False)
class LogicalBinaryOperator(LogicalOperator[T], BinaryOperator, ABC):
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
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        sources = sources or {}
        self._eval_parent_ = parent
        left_values = self.left._evaluate__(sources, parent=self)
        for left_value in left_values:
            self._is_false_ = left_value.is_false
            if self._is_false_:
                yield OperationResult(left_value.bindings, self._is_false_, self)
            else:
                yield from self.evaluate_right(left_value)

    def evaluate_right(self, left_value: OperationResult) -> Iterable[OperationResult]:
        right_values = self.right._evaluate__(left_value.bindings, parent=self)
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
        sources: Dict[int, Any],
    ) -> Iterable[OperationResult]:
        """
        Evaluate the left operand, taking into consideration if it should yield when it is False.

        :param sources: The current bindings to use for evaluation.
        :return: The new bindings after evaluating the left operand (and possibly right operand).
        """
        left_values = self.left._evaluate__(sources, parent=self)

        for left_value in left_values:
            self.left_evaluated = True
            left_is_false = left_value.is_false
            if left_is_false:
                yield from self.evaluate_right(left_value.bindings)
            else:
                self._is_false_ = False
                yield OperationResult(left_value.bindings, self._is_false_, self)

    def evaluate_right(self, sources: Dict[int, Any]) -> Iterable[OperationResult]:
        """
        Evaluate the right operand.

        :param sources: The current bindings to use during evaluation.
        :return: The new bindings after evaluating the right operand.
        """

        self.left_evaluated = False

        right_values = self.right._evaluate__(sources, parent=self)

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
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        sources = sources or {}
        self._eval_parent_ = parent

        yield from self.evaluate_left(sources)
        yield from self.evaluate_right(sources)


@dataclass(eq=False, repr=False)
class ElseIf(OR):
    """
    A version of the OR operator that evaluates the right operand only when the left operand is False.
    """

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle ElseIf logic.
        """
        sources = sources or {}
        self._eval_parent_ = parent
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
            v._id_
            for v in self.condition._unique_variables_.difference(
                self.left._unique_variables_
            )
        ]

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        sources = sources or {}
        self._eval_parent_ = parent

        solution_set = None

        for var_val in self.variable._evaluate__(sources, parent=self):
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

    def get_all_candidate_solutions(self, sources: Dict[int, Any]):
        values_that_satisfy_condition = []
        # Evaluate the condition under this particular universal value
        for condition_val in self.condition._evaluate__(sources, parent=self):
            if condition_val.is_false:
                continue
            condition_val_bindings = {
                k: v
                for k, v in condition_val.bindings.items()
                if k in self.condition_unique_variable_ids
            }
            values_that_satisfy_condition.append(condition_val_bindings)
        return values_that_satisfy_condition

    def evaluate_condition(self, sources: Dict[int, Any]) -> bool:
        for condition_val in self.condition._evaluate__(sources, parent=self):
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
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        sources = sources or {}
        self._eval_parent_ = parent
        seen_var_values = []
        for val in self.condition._evaluate__(sources, parent=self):
            var_val = val[self.variable._id_]
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
