"""
Core symbolic expression system used to build and evaluate entity queries.

This module defines the symbolic types (variables, sources, logical and
comparison operators) and the evaluation mechanics.
"""

from __future__ import annotations

import operator
import typing
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
    Generic,
    TYPE_CHECKING,
    List,
    Tuple,
    Callable,
    Self,
    Set,
    Iterator,
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
    NonAggregatedSelectedVariablesError,
    NoConditionsProvidedToWhereStatementOfDescriptor,
    AggregatorInWhereConditionsError,
    HavingUsedBeforeWhereError,
    NonAggregatorInHavingConditionsError,
    UnsupportedAggregationOfAGroupedByVariable,
    NestedAggregationError,
)
from .failures import VariableCannotBeEvaluated
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
    chain_stages,
    merge_args_and_kwargs,
    convert_args_and_kwargs_into_a_hashable_key,
    ensure_hashable,
)
from ..class_diagrams import ClassRelation
from ..class_diagrams.class_diagram import WrappedClass
from ..class_diagrams.failures import ClassIsUnMappedInClassDiagram
from ..class_diagrams.wrapped_field import WrappedField

if TYPE_CHECKING:
    from .conclusion import Conclusion
    from .entity import ConditionType

id_generator = IDGenerator()

RWXNode.enclosed_name = "Selected Variable"

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
class SymbolicExpression(Generic[T], ABC):
    """
    Base class for all symbolic expressions.

    Symbolic expressions form a tree and are evaluated lazily to produce
    bindings for variables, subject to logical constraints.
    """

    _child_: Optional[SymbolicExpression] = field(init=False)
    """
    Optional child expression of this symbolic expression.
    """
    _id_: int = field(init=False, repr=False, default=None)
    """
    Unique identifier of this node.
    """
    _node_: RWXNode = field(init=False, default=None, repr=False)
    """
    The rustworkx node of this symbolic expression.
    """
    _id_expression_map_: ClassVar[Dict[int, SymbolicExpression]] = {}
    """
    A mapping of symbolic expression IDs to symbolic expressions. This is used to retrieve symbolic expressions by ID
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
    _is_false_: bool = field(init=False, repr=False, default=False)
    """
    Internal flag indicating current truth value of evaluation result for this expression.
    """
    _eval_parent_: Optional[SymbolicExpression] = field(
        default=None, init=False, repr=False
    )
    """
    The parent symbolic expression of this expression. This is used to determine the current parent expression during
    evaluation.
    """
    _plot_color__: Optional[ColorLegend] = field(default=None, init=False, repr=False)
    """
    The color to use for plotting the node of this symbolic expression.
    """

    def __post_init__(self):
        if not self._id_:
            self._id_ = id_generator(self)
            self._create_node_()
            self._id_expression_map_[self._id_] = self
        if hasattr(self, "_child_") and self._child_ is not None:
            self._update_child_()

    def _update_child_(
        self,
        child: Optional[SymbolicExpression] = None,
    ):
        """
        Update the child expression of this symbolic expression.
        """
        child = child or self._child_
        self._child_ = self._update_children_(child)[0]

    def _update_children_(
        self, *children: SymbolicExpression
    ) -> Tuple[SymbolicExpression, ...]:
        """
        Update multiple children expressions of this symbolic expression.
        """
        children: Dict[int, SymbolicExpression] = dict(enumerate(children))
        for k, v in children.items():
            if not isinstance(v, SymbolicExpression):
                children[k] = Literal(v)
        for k, v in children.items():
            # With graph structure, do not copy nodes; just connect an edge.
            v._node_.parent = self._node_
        return tuple(children.values())

    def _create_node_(self):
        """
        Create the rustworkx node for this symbolic expression.
        """
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
                return
            yield from self._evaluate__(sources)
        finally:
            self._eval_parent_ = previous_parent

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
        elif self._node_.parent is not None:
            return self._node_.parent.data
        return None

    @_parent_.setter
    def _parent_(self, value: Optional[SymbolicExpression]):
        """
        Set the parent symbolic expression of this expression.

        :param value: The new parent symbolic expression of this expression.
        """
        self._node_.parent = value._node_ if value is not None else None
        if value is not None and hasattr(value, "_child_"):
            value._child_ = self

    @cached_property
    def _conditions_root_(self) -> SymbolicExpression:
        """
        :return: The root of the symbolic expression tree that contains conditions.
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
        :return: The root of the symbolic expression tree.
        """
        return self._node_.root.data

    @property
    @abstractmethod
    def _name_(self) -> str:
        """
        :return: The name of this symbolic expression.
        """
        pass

    @property
    def _all_nodes_(self) -> List[SymbolicExpression]:
        """
        :return: All nodes in the symbolic expression tree.
        """
        return [self._root_] + self._root_._descendants_

    @property
    def _descendants_(self) -> List[SymbolicExpression]:
        """
        :return: All descendants of this symbolic expression.
        """
        return [d.data for d in self._node_.descendants]

    @property
    def _children_(self) -> List[SymbolicExpression]:
        """
        :return: All children of this symbolic expression.
        """
        return [c.data for c in self._node_.children]

    @classmethod
    def _current_parent_(cls) -> Optional[SymbolicExpression]:
        """
        :return: The current parent symbolic expression in the enclosing context of the ``with`` statement.
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

    @property
    def _plot_color_(self) -> ColorLegend:
        """
        :return: The color legend for this symbolic expression.
        """
        return self._plot_color__

    @_plot_color_.setter
    def _plot_color_(self, value: ColorLegend):
        """
        Set the color legend for this symbolic expression.

        :param value: The new color legend for this symbolic expression.
        """
        self._plot_color__ = value
        self._node_.color = value

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
        node = self
        if (node is self._root_) or (node._parent_ is self._root_):
            node = node._conditions_root_
        SymbolicExpression._symbolic_expression_stack_.append(node)
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


ResultMapping = Callable[[Iterable[Bindings]], Iterator[Bindings]]
"""
A function that maps the results of a query object descriptor to a new set of results.
"""


@dataclass(eq=False, repr=False)
class ConstraintSpecifier(SymbolicExpression[T], ABC):
    """
    A constraint specifier is a symbolic expression that specifies the conditions that must be satisfied by the results
    produced by a query.
    """

    _child_: SymbolicExpression[T]
    """
    The conditions expression which generate the valid bindings that satisfy the constraints.
    """

    @cached_property
    def conditions(self) -> SymbolicExpression:
        """
        The conditions expression.
        """
        return self._child_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterator[OperationResult]:

        yield from self._child_._evaluate_(sources, self)

    @property
    def _is_false_(self) -> bool:
        return self._child_._is_false_

    @_is_false_.setter
    def _is_false_(self, value: bool):
        self._child_._is_false_ = value

    @property
    def _name_(self) -> str:
        return self.__class__.__name__

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_


@dataclass(eq=False, repr=False)
class Where(ConstraintSpecifier[T]):
    """
    A symbolic expression that represents the `where()` statement of `QueryObjectDescriptor`.
    """

    ...


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
    def _binding_id_(self) -> int:
        return (
            self._var_._binding_id_
            if self._var_ and self._var_ is not self
            else self._id_
        )

    @cached_property
    def _type__(self):
        return self._var_._type_ if self._var_ and self._var_ is not self else None

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

    _path_: List[ClassRelation] = field(init=False, default_factory=list)
    """
    The path of the variable in the symbol graph as a sequence of relation instances.
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
        if isinstance(self._parent_, (LogicalOperator, ConstraintSpecifier)):
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
class ResultProcessor(CanBehaveLikeAVariable[T], ABC):
    """
    Base class for processors that return concrete results from queries, including quantifiers
    (e.g., An, The) and aggregators (e.g., Count, Sum, Max, Min).
    """

    _child_: SymbolicExpression[T]
    """
    The child expression of the processor where the results to be processed come from.
    """

    def __post_init__(self):
        if isinstance(self._child_, QueryObjectDescriptor):
            self._child_._build_()
        super().__post_init__()

    def tolist(self):
        """
        Evaluate and return the results as a list.
        """
        return list(self.evaluate())

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
        yield from map(self._child_._process_result_, self._evaluate_())

    def _process_result_(self, result: OperationResult) -> T:
        return self._child_._process_result_(result)

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
    _distinct_: bool = field(kw_only=True, default=False)
    """
    Whether to consider only distinct values from the child results when applying the aggregation function.
    """

    def __post_init__(self):
        if isinstance(self._child_, Aggregator):
            raise NestedAggregationError(self)
        super().__post_init__()
        self._var_ = self

    def evaluate(self) -> Iterator[T]:
        yield from An(_child_=Entity(_selected_variables=[self])).evaluate()

    def grouped_by(self, *variables: Variable) -> Entity[T]:
        """
        Group the results by the given variables.
        """
        return Entity(_selected_variables=[self]).grouped_by(*variables)

    def _process_result_(
        self, result: OperationResult
    ) -> TypingUnion[UnificationDict, Any]:
        """
        Map the result to the correct output data structure for user usage. This returns the selected variables only.
        Return a dictionary with the selected variables as keys and the values as values.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        if (
            isinstance(self._child_, QueryObjectDescriptor)
            and self._child_._variables_to_group_by_
        ):
            return UnificationDict(
                {self._id_expression_map_[id_]: v for id_, v in result.bindings.items()}
            )
        else:
            return super()._process_result_(result)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        child_results = self._child_._evaluate_(sources, parent=self)
        values = self._apply_aggregation_function_(child_results)

        for value in values:
            yield OperationResult({**sources, **value}, False, self)

    @abstractmethod
    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Iterator[Bindings]:
        """
        Apply the aggregation function to the results of the child.

        :param child_results: The results of the child.
        :return: An iterator of bindings containing the aggregated results.
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

    _child_: Optional[SymbolicExpression[T]] = None
    """
    The child expression to be counted. If not given, the count of all results (by group) is returned.
    """

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Iterator[Bindings]:
        for res in child_results:
            if self._distinct_:
                yield {self._binding_id_: len(set(res.value))}
            else:
                yield {self._binding_id_: len(res.value)}


@dataclass(eq=False, repr=False)
class EntityAggregator(Aggregator[T], ABC):
    _child_: Selectable[T]
    """
    The child entity to be aggregated.
    """
    _key_func_: Callable[[Any], Any] = field(kw_only=True, default=lambda x: x)
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
        if self._key_func_:
            return self._key_func_(result)
        return result


@dataclass(eq=False, repr=False)
class Sum(EntityAggregator[T]):
    """
    Calculate the sum of the child results. If given, make use of the key function to extract the value to be summed.
    """

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Iterator[Bindings]:
        for res in child_results:
            if not res.has_value or len(res.value) == 0:
                yield {self._binding_id_: self._default_value_}
                continue
            results = list(map(self._get_child_value_from_result_, res.value))
            if self._distinct_:
                results = set(results)
            sum_val = sum(results)
            yield {self._binding_id_: sum_val}


@dataclass(eq=False, repr=False)
class Average(EntityAggregator[T]):
    """
    Calculate the average of the child results. If given, make use of the key function to extract the value to be
     averaged.
    """

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Iterator[Bindings]:
        for res in child_results:
            if not res.has_value:
                yield {self._binding_id_: self._default_value_}
                continue
            results = list(map(self._get_child_value_from_result_, res.value))
            if self._distinct_:
                results = set(results)
            sum_val = sum(results)
            yield {self._binding_id_: sum_val / len(results)}


@dataclass(eq=False, repr=False)
class Extreme(EntityAggregator[T], CanBehaveLikeAVariable[T], ABC):
    """
    Find and return the extreme value among the child results. If given, make use of the key function to extract
    the value to be compared.
    """

    def __post_init__(self):
        self._var_ = (
            self._child_._var_ if isinstance(self._child_, Selectable) else None
        )
        super().__post_init__()

    def _apply_aggregation_function_(
        self, child_results: Iterable[OperationResult]
    ) -> Iterator[Bindings]:
        for res in child_results:
            try:
                results = res.value
                if self._distinct_:
                    results = set(results)
                extreme_val = self._extreme_function_(results, key=self._key_func_)
                bindings = res.bindings.copy()
                bindings[self._binding_id_] = extreme_val
                yield bindings
            except ValueError:
                # Means that the child results were empty;
                # the default value will be returned instead.
                yield {self._binding_id_: self._default_value_} | res.bindings

    @abstractmethod
    def _extreme_function_(
        self, values: Iterable, key: Optional[Callable[[Any], Any]] = None
    ) -> Callable[[Any], Any]: ...


@dataclass(eq=False, repr=False)
class Max(Extreme[T]):
    """
    Find and return the maximum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def _extreme_function_(
        self, values: Iterable, key: Optional[Callable[[Any], Any]] = None
    ) -> Any:
        return max(values, key=key)


@dataclass(eq=False, repr=False)
class Min(Extreme[T]):
    """
    Find and return the minimum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def _extreme_function_(
        self, values: Iterable, key: Optional[Callable[[Any], Any]] = None
    ) -> Any:
        return min(values, key=key)


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
        self._var_ = (
            self._child_._var_ if isinstance(self._child_, Selectable) else None
        )
        self._node_.wrap_subtree = True

    def _distinct__(self, *on: Selectable):
        self._child_.distinct(*on)

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
        key = key._id_expression_map_[key._binding_id_]
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
        sources: Bindings,
    ) -> Iterable[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression[T]], T]]]:
        try:
            yield from super()._evaluate__(sources)
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


GroupBindings = Dict[GroupKey, OperationResult]
"""
A dictionary for grouped bindings which maps a group key to its corresponding bindings.
"""


@dataclass(eq=False, repr=False)
class GroupBy(SymbolicExpression[T]):
    """
    Represents a group-by operation in the entity query language. This operation groups the results of a query by
    specific variables. This is useful for aggregating results separately for each group.
    """

    query_descriptor: QueryObjectDescriptor[T]
    """
    The query object descriptor that is being grouped.
    """

    def __post_init__(self):
        self._child_ = self.query_descriptor._where_expression_
        super().__post_init__()

    def _evaluate__(self, sources: Bindings = None) -> Iterator[OperationResult]:
        """
        Generate results grouped by the specified variables in the grouped_by clause.

        :param sources: The current bindings.
        :return: An iterator of OperationResult objects, each representing a group of child results.
        """

        if any(self.aggregators_of_grouped_by_variables_that_are_not_count()):
            raise UnsupportedAggregationOfAGroupedByVariable(
                self.query_descriptor, self
            )

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

        for res in self.get_constrained_values(sources):

            group_key = tuple(
                ensure_hashable(res[var._binding_id_])
                for var in self.variables_to_group_by
            )

            if self.count_occurrences_of_each_group_key:
                group_key_count[group_key] += 1

            self.update_group_from_bindings(groups[group_key], res)

        if len(groups) == 0:
            for var in self.aggregated_variables:
                groups[()][var._binding_id_] = []

        return groups, group_key_count

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
        expression = self._id_expression_map_[var_id]
        return (
            len(self.variables_to_group_by) == 1
            and isinstance(expression, DomainMapping)
            and expression._child_._binding_id_ in self.ids_of_variables_to_group_by
        )

    def get_constrained_values(self, sources: Bindings) -> Iterator[Bindings]:
        """
        :param sources: The current bindings.
        :return: An iterator over the constrained value by the where clause, grouped by variables, and aggregated
        variables of the group by expression.
        """
        constrained_values = self.evaluate_where_expression(sources)
        constrained_values = self.evaluate_grouped_by_variables(constrained_values)
        constrained_values = self.evaluate_aggregated_variables(constrained_values)
        return constrained_values

    @cached_property
    def count_occurrences_of_each_group_key(self) -> bool:
        """
        :return: True if there are any aggregators of type Count in the selected variables of the query descriptor that
         are counting values of variables that are in the grouped_by clause, False otherwise.
        """
        return len(self.aggregators_of_grouped_by_variables) > 0

    def evaluate_where_expression(self, sources: Bindings) -> Iterator[Bindings]:
        """
        :param sources: The current bindings.
        :return: An iterator of Bindings that satisfy the where expression of the query object descriptor.
         If there is no where expression, it returns the input bindings as is.
        """
        if self.where_expression:
            yield from (
                res.bindings
                for res in self.where_expression._evaluate_(sources, parent=self)
                if res.is_true
            )
        else:
            yield sources

    def evaluate_aggregated_variables(
        self, sources: Iterator[Bindings]
    ) -> Iterator[Bindings]:
        """
        Evaluate the aggregated variables by generating combinations of values from their evaluation generators.

        :param sources: The current bindings.
        :return: An Iterable of Bindings for each combination of values.
        """
        for values in sources:
            yield from QueryObjectDescriptor._chain_evaluate_variables(
                self.aggregated_variables,
                values,
                parent=self,
            )

    @cached_property
    def aggregated_variables(self):
        """
        :return: A list of the aggregated variables from the aggregators in the selected variables of the query descriptor.
        """
        return [var._child_ for var in self.aggregators if var._child_ is not None]

    def aggregators_of_grouped_by_variables_that_are_not_count(
        self,
    ) -> Iterator[Aggregator]:
        """
        :return: Aggregators in the selected variables of the query descriptor that are aggregating over
         expressions having variables that are in the grouped_by clause and are not Count.
        """
        yield from (
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
    def aggregators(self):
        """
        :return: A list of aggregators in the selected variables of the query descriptor.
        """
        return [
            var
            for var in self.query_descriptor._selected_variables
            if isinstance(var, Aggregator)
        ]

    def evaluate_grouped_by_variables(
        self, sources: Iterator[Bindings]
    ) -> Iterator[Bindings]:
        """
        Evaluate the grouped_by variables by generating combinations of values from their evaluation generators.

        :param sources: The current bindings.
        :return: An Iterable of Bindings for each combination of values.
        """
        for values in sources:
            yield from QueryObjectDescriptor._chain_evaluate_variables(
                self.variables_to_group_by, values, parent=self
            )

    @property
    def where_expression(self) -> Optional[SymbolicExpression[T]]:
        return self._child_

    @cached_property
    def ids_of_variables_to_group_by(self) -> Tuple[int, ...]:
        return tuple(var._binding_id_ for var in self.variables_to_group_by)

    @cached_property
    def variables_to_group_by(self) -> Tuple[Selectable, ...]:
        return tuple(self.query_descriptor._variables_to_group_by_)

    @property
    def _name_(self) -> str:
        return f"{self.__class__.__name__}({', '.join([var._name_ for var in self.variables_to_group_by])})"

    @cached_property
    def _all_variable_instances_(self) -> List[Selectable]:
        return self.where_expression._all_variable_instances_ + list(
            self.variables_to_group_by
        )


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
    _variables_to_group_by_: Tuple[Selectable, ...] = field(default=(), init=False)
    """
    The variables to group the results by.
    """
    _order_by: Optional[OrderByParams] = field(default=None, init=False)
    """
    Parameters for ordering the results of the query object descriptor.
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
    _where_condition_list_: List[ConditionType] = field(
        default_factory=list, init=False
    )
    """
    The condition list of the query object descriptor.
    """
    _having_condition_list_: List[ConditionType] = field(
        default_factory=list, init=False
    )
    """
    The condition list of the query object descriptor.
    """
    _having_conditions_expression_: Optional[SymbolicExpression] = field(
        default=None, init=False
    )
    """
    The having expression of the query object descriptor.
    """
    _where_expression_: Optional[Where] = field(default=None, init=False)
    """
    The where expression of the query object descriptor.
    """
    _limit_: Optional[int] = field(default=None, init=False)
    """
    The limit on the number of results returned by the query.
    """

    def __post_init__(self):
        super().__post_init__()
        if self._aggregated_and_non_aggregated_variables_in_selection_[0]:
            self._group_ = True
        self._enclose_plotted_nodes_of_selected_variables_()

    def tolist(self) -> List[T]:
        """
        Map the results of the query object descriptor to a list of the selected variable values.

        :return: A list of the selected variable values.
        """
        return list(An(_child_=self).evaluate())

    def where(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query object descriptor.
        """

        self._where_condition_list_ = list(conditions)

        self._assert_correct_where_conditions_()

        # Build the expression from the conditions
        expression = (
            chained_logic(AND, *self._where_condition_list_)
            if len(self._where_condition_list_) > 1
            else self._where_condition_list_[0]
        )

        self._where_expression_ = Where(expression)
        return self

    def having(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query object descriptor.
        """
        self._having_condition_list_ = list(conditions)

        self._assert_correct_having_conditions_()

        # Build the expression from the conditions
        expression = (
            chained_logic(AND, *self._having_condition_list_)
            if len(self._having_condition_list_) > 1
            else self._having_condition_list_[0]
        )
        self._having_conditions_expression_ = expression
        return self

    def order_by(
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
        self._order_by = OrderByParams(variable, descending, key)
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
        self._distinct_on = on if on else self._selected_variables
        self._seen_results = SeenSet(keys=self._distinct_on_ids_)
        self._results_mapping.append(self._get_distinct_results_)
        return self

    def grouped_by(
        self, *variables: TypingUnion[Selectable, Any]
    ) -> TypingUnion[Self, T]:
        """
        Specify the variables to group the results by.

        :param variables: The variables to group the results by.
        :return: This query object descriptor.
        """
        self._variables_to_group_by_ = tuple(variables)
        return self

    def _build_(self):
        """
        Build the query object descriptor by wiring the nodes together in the correct order of evaluation.
        """
        group_by = None
        having = None
        if self._group_:
            group_by = GroupBy(self)
            if self._having_conditions_expression_:
                having = Having(
                    group_by=group_by, _child_=self._having_conditions_expression_
                )
        if having:
            self._update_child_(having)
        elif group_by:
            self._update_child_(group_by)
        elif self._where_expression_:
            self._update_child_(self._where_expression_)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the query descriptor by constraining values, updating conclusions,
        and selecting variables.
        """
        self._assert_correct_selected_variables_()

        if all(var._binding_id_ in sources for var in self._selected_variables):
            yield OperationResult(sources, False, self)
            return

        results_generator = self._generate_results_(sources)

        if self._order_by:
            yield from self._order_(results_generator)
        else:
            yield from results_generator

        if self._seen_results is not None:
            self._seen_results.clear()

    def _generate_results_(self, sources: Dict[int, Any]) -> Iterator[OperationResult]:
        """
        Internal generator to process constrained values and selected variables.
        """
        for values in self._get_constrained_values_(sources):

            selected_vars_bindings = self._evaluate_selected_variables(values)

            yield from (
                OperationResult({**values, **result}, False, self)
                for result in self._apply_results_mapping_(selected_vars_bindings)
            )

    def _order_(
        self, results: Iterator[OperationResult] = None
    ) -> List[OperationResult]:
        """
        Order the results by the given order variable.

        :param results: The results to be ordered.
        :return: The ordered results.
        """

        def key(result: OperationResult) -> Any:
            var = self._order_by.variable
            var_id = var._binding_id_
            if var_id not in result:
                result[var_id] = next(var._evaluate_(result.bindings, self)).value
            variable_value = result.bindings[var_id]
            if self._order_by.key:
                return self._order_by.key(variable_value)
            else:
                return variable_value

        return sorted(
            results,
            key=key,
            reverse=self._order_by.descending,
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

    def _enclose_plotted_nodes_of_selected_variables_(self):
        """
        Enclose the selected variables in the query object descriptor.
        """
        for variable in self._selected_variables:
            variable._var_._node_.enclosed = True

    @lru_cache
    def _assert_correct_selected_variables_(self):
        """
        Assert that the selected variables are correct.

        :raises UsageError: If the selected variables are not valid.
        """
        aggregated_variables, non_aggregated_variables = (
            self._aggregated_and_non_aggregated_variables_in_selection_
        )
        if aggregated_variables and any(
            var._binding_id_ not in self._ids_of_variables_to_group_by_
            for var in non_aggregated_variables
        ):
            raise NonAggregatedSelectedVariablesError(
                self,
                non_aggregated_variables,
                aggregated_variables,
            )

    @cached_property
    def _ids_of_variables_to_group_by_(self) -> List[int]:
        """
        Get the ids of the variables to group by.
        """
        return [var._binding_id_ for var in self._variables_to_group_by_]

    @cached_property
    def _aggregated_and_non_aggregated_variables_in_selection_(
        self,
    ) -> Tuple[List[Selectable], List[Selectable]]:
        """
        :return: The aggregated and non-aggregated variables from the selected variables.
        """
        aggregated_variables = []
        non_aggregated_variables = []
        for variable in self._selected_variables:
            if isinstance(variable, Aggregator):
                aggregated_variables.append(variable)
            else:
                non_aggregated_variables.append(variable)
        return aggregated_variables, non_aggregated_variables

    @cached_property
    def _group_(self) -> bool:
        """
        :return: Whether the results should be grouped or not. Is true when an aggregator is selected.
        """
        return len(self._aggregated_and_non_aggregated_variables_in_selection_[0]) > 0

    def _assert_correct_conditions_(self, conditions: List[ConditionType]):
        """
        :param conditions: The conditions that describe the query object.
        :raises UsageError: If the conditions are not valid.
        """
        # If there are no conditions raise error.
        if len(conditions) == 0:
            raise NoConditionsProvidedToWhereStatementOfDescriptor(self)

        # If there's a constant condition raise error.
        literal_expressions = [
            exp for exp in conditions if not isinstance(exp, SymbolicExpression)
        ]
        if literal_expressions:
            raise LiteralConditionError(self, literal_expressions)

    def _assert_correct_where_conditions_(self):
        """
        Assert that the where conditions are correct.

        :raises UsageError: If the where conditions are not valid.
        """
        if self._having_condition_list_:
            raise HavingUsedBeforeWhereError(self)
        self._assert_correct_conditions_(self._where_condition_list_)
        aggregators, non_aggregators = (
            self._aggregators_and_non_aggregators_in_conditions_(
                tuple(self._where_condition_list_)
            )
        )
        if aggregators:
            raise AggregatorInWhereConditionsError(self, aggregators)

    def _assert_correct_having_conditions_(self):
        """
        Assert that the having conditions are correct.

        :raises UsageError: If the having conditions are not valid.
        """
        self._assert_correct_conditions_(self._having_condition_list_)
        aggregators, non_aggregators = (
            self._aggregators_and_non_aggregators_in_conditions_(
                tuple(self._having_condition_list_)
            )
        )
        if non_aggregators:
            raise NonAggregatorInHavingConditionsError(self, non_aggregators)

    def _aggregators_and_non_aggregators_in_conditions_(
        self, conditions: Tuple[ConditionType, ...]
    ) -> Tuple[List[Aggregator], List[Selectable]]:
        """
        :param conditions: The conditions that describe the query object.
        :return: A tuple containing the aggregators and non-aggregators in the where condition.
        """
        aggregators, non_aggregators = [], []
        for cond in conditions:
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
        return aggregators, non_aggregators

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
            for var in self._selected_variables
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

    def _evaluate_conclusions_and_update_bindings_(
        self, child_result: Bindings
    ) -> Bindings:
        """
        Update the bindings of the results by evaluating the conclusions using the received bindings from the child as
        sources.

        :param child_result: The result of the child operation.
        """
        if not self._child_:
            return child_result
        for conclusion in self._child_._conclusion_:
            child_result = next(
                conclusion._evaluate_(child_result, parent=self)
            ).bindings
        return child_result

    def _get_constrained_values_(self, sources: Bindings) -> Iterator[Bindings]:
        """
        Evaluate the child (i.e., the conditions that constrain the domain of the selected variables).

        :param sources: The current bindings.
        :return: The bindings after applying the constraints of the child.
        """

        if not self._child_:
            yield sources
            return

        for res in self._get_child_true_results__(sources):

            self._evaluate_conclusions_and_update_bindings_(res.bindings)

            if self._any_selected_variable_is_inferred_and_unbound_(res.bindings):
                continue

            yield res.bindings

    def _get_child_true_results__(self, sources: Bindings) -> Iterator[OperationResult]:
        """
        :param sources: The current bindings.
        :return: An iterator of child results that are true.
        """
        yield from (
            res for res in self._child_._evaluate_(sources, parent=self) if res.is_true
        )

    def _evaluate_children_of_aggregators_(
        self, sources: Iterator[Bindings]
    ) -> Iterator[Bindings]:
        """
        Evaluate the children of aggregators by generating combinations of values from their evaluation generators.

        :param sources: The current bindings.
        :return: An Iterable of Bindings for each combination of values.
        """
        for values in sources:
            yield from self._chain_evaluate_variables(
                [
                    var._child_
                    for var in self._selected_variables
                    if isinstance(var, Aggregator)
                ],
                values,
                parent=self,
            )

    def _evaluate_selected_variables(self, sources: Bindings) -> Iterator[Bindings]:
        """
        Evaluate the selected variables by generating combinations of values from their evaluation generators.

        :param sources: The current bindings.
        :return: An Iterable of Bindings for each combination of values.
        """
        yield from self._chain_evaluate_variables(
            self._selected_variables, sources, parent=self
        )

    @staticmethod
    def _chain_evaluate_variables(
        variables: Iterable[SymbolicExpression[T]],
        sources: Bindings,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterator[Bindings]:
        """
        Evaluate the selected variables by generating combinations of values from their evaluation generators.

        :param sources: The current bindings.
        :return: An Iterable of OperationResults for each combination of values.
        """
        var_val_gen = [
            (
                lambda bindings, var=var: (
                    v.bindings for v in var._evaluate_(copy(bindings), parent=parent)
                )
            )
            for var in variables
        ]

        yield from chain_stages(var_val_gen, sources)

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
        return UnificationDict(
            {v._var_: result[v._binding_id_] for v in self._selected_variables}
        )

    def __getitem__(
        self, selected_variable: TypingUnion[CanBehaveLikeAVariable[T], T]
    ) -> TypingUnion[T, SetOfSelectable[T]]:
        """
        Select one of the set of variables, this is useful when you have another query that uses this set of and
        wants to select a specific variable out of the set of variables.

        :param selected_variable: The selected variable from the set of variables.
        """
        self._build_()
        return SetOfSelectable(self, selected_variable)


@dataclass(eq=False, repr=False)
class SetOfSelectable(CanBehaveLikeAVariable[T]):
    """
    A selected variable from the SetOf operation selected variables.
    """

    _set_of_: SetOf[T]
    """
    The SetOf operation from which `_selected_var_` was selected.
    """
    _selected_var_: CanBehaveLikeAVariable[T]
    """
    The selected variable in the SetOf.
    """

    def __post_init__(self):
        self._child_ = self._set_of_
        self._var_ = self
        super().__post_init__()

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
    def _all_variable_instances_(self) -> List[Variable]:
        return self._set_of_._all_variable_instances_


@dataclass(eq=False, repr=False)
class Entity(QueryObjectDescriptor[T], CanBehaveLikeAVariable[T]):
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

    @property
    def _plot_color_(self) -> ColorLegend:
        if self._plot_color__:
            return self._plot_color__
        else:
            return ColorLegend("Literal", "#949292")


@dataclass(eq=False, repr=False)
class Concatenate(CanBehaveLikeAVariable[T]):
    """
    Concatenation of two or more variables.
    """

    _variables_: List[Selectable[T]] = field(default_factory=list)
    """
    The variables to concatenate.
    """

    def __post_init__(self):
        self._child_ = None
        super().__post_init__()
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
    def _plot_color_(self) -> ColorLegend:
        if self._plot_color__:
            return self._plot_color__
        else:
            return ColorLegend("Concatenate", "#949292")

    @property
    def _all_variable_instances_(self) -> List[Variable]:
        all_vars = []
        for var in self._variables_:
            all_vars.extend(var._all_variable_instances_)
        return all_vars

    @property
    def _name_(self):
        return self.__class__.__name__


@dataclass(eq=False, repr=False)
class DomainMapping(CanBehaveLikeAVariable[T], ABC):
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

    @cached_property
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        return self.left._all_variable_instances_ + self.right._all_variable_instances_


@dataclass(eq=False, repr=False)
class Having(ConstraintSpecifier[T]):
    """
    A symbolic having expression that can be used to filter the grouped results of a query. Is constructed through
    the `QueryObjectDescriptor` using the `having()` method.
    """

    group_by: GroupBy
    """
    The group by expression that is used to group the results of the query. This is a child of the Having expression.
    As the results need to be grouped before filtering.
    """

    def __post_init__(self):
        super().__post_init__()
        self._update_children_(self.group_by)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        yield from (
            OperationResult(
                condition_result.bindings,
                self.get_and_update_truth_value(condition_result),
                self,
            )
            for grouped_result in self.group_by._evaluate_(sources, parent=self)
            for condition_result in self.conditions._evaluate_(
                grouped_result.bindings, parent=self
            )
        )

    def get_and_update_truth_value(self, result: OperationResult) -> bool:
        """
        Retrieves and updates the truth value of the given operation result.

        :param result: The operation result to process.
        :return: The truth value of the operation result.
        """
        self.is_false = result.is_false
        return result.is_false

    @cached_property
    def _all_variable_instances_(self) -> List[Selectable]:
        return self.group_by._all_variable_instances_ + super()._all_variable_instances_


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
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        for v in self._child_._evaluate_(sources, parent=self):
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
