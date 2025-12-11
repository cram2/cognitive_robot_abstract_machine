from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Optional, Type, Dict, Any, List, Union, Self, Iterable, Generic, Callable

from .entity import (
    ConditionType,
    contains,
    in_,
    flatten,
    let,
    set_of,
    entity,
    exists,
)
from .failures import NoneWrappedFieldError, WrongSelectableType, UnquantifiedMatchError
from .predicate import HasType
from .rxnode import RWXNode
from .symbolic import (
    CanBehaveLikeAVariable,
    Attribute,
    Comparator,
    QueryObjectDescriptor,
    Selectable,
    SymbolicExpression,
    OperationResult,
    Literal,
    ResultQuantifier,
    An,
    Variable,
    Flatten,
    Exists,
    DomainType, ResultProcessor, OrderByParams
)
from .utils import is_iterable, T


@dataclass
class ResultProcessorData:
    """
    A class representing a result processor in a Match statement. This is used to process the result of the match.
    """
    type_: Type[ResultProcessor]
    """
    The type of the result processor.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    The keyword arguments to pass to the result processor.
    """

    def apply(self, expr: Selectable[T]) -> Union[ResultProcessor[T], T]:
        return self.type_(_child_=expr, **self.kwargs)


@dataclass
class SelectableMatchExpression(CanBehaveLikeAVariable[T]):
    """
    Base class for all match expressions.

    Match expressions are structured in a graph that is a higher level representation for the entity query graph.
    """
    _match_expression_: AbstractMatchExpression[T]
    """
    The match expression that this class wraps and makes selectable.
    """
    _selectable_attribute_matches_: Dict[str, SelectableMatchExpression] = field(init=False, default_factory=dict)
    """
    A dictionary mapping attribute names to their corresponding selectable match expressions.
    """

    def __post_init__(self):
        """
        This is to avoid running __post_init__ of the BaseClass CanBehaveLikeAVariable and to set _var_, _node_, and
        _id_ manually from the match_expression.
        """
        self._var_ = self._match_expression_.variable
        self._node_ = self._var_._node_
        self._id_ = self._var_._id_

    def evaluate(self):
        """
        Evaluate the match expression and return the result.
        """
        return self._match_expression_.expression.evaluate()

    def __getattr__(self, name):
        # Prevent debugger/private attribute lookups from being interpreted as symbolic attributes
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )
        if name not in self._match_expression_.attribute_matches:
            attr = getattr(self._match_expression_.expression, name)
            attribute_expression = AttributeMatch(parent=self._match_expression_, attr_name=name, variable=attr)
            self._match_expression_.attribute_matches[name] = attribute_expression
        if name not in self._selectable_attribute_matches_:
            attribute_expression = self._match_expression_.attribute_matches[name]
            selectable_attribute_expression = SelectableMatchExpression(_match_expression_=attribute_expression)
            self._selectable_attribute_matches_[name] = selectable_attribute_expression
        return self._selectable_attribute_matches_[name]

    def _evaluate__(self, sources: Optional[Dict[int, Any]] = None, parent: Optional[SymbolicExpression] = None) -> \
            Iterable[OperationResult]:
        self._eval_parent_ = parent
        yield from self._var_._evaluate__(sources, self)

    @property
    def _name_(self) -> str:
        return f"Selectable({self._match_expression_.name})"

    def _all_variable_instances_(self) -> List[Variable]:
        return self._var_._all_variable_instances_

    def __str__(self):
        return self._name_

    def __repr__(self):
        return self._name_


@dataclass
class AbstractMatchExpression(Generic[T], ABC):
    _type: Optional[Type[T]] = field(default=None, kw_only=True)
    """
    The type of the variable.
    """
    variable: Optional[Selectable[T]] = field(default=None, kw_only=True)
    """
    The created variable from the type and kwargs.
    """
    is_selected: bool = field(default=False, kw_only=True)
    """
    Whether the variable should be selected in the result.
    """
    existential: bool = field(default=False, kw_only=True)
    """
    Whether the match is an existential match check or not.
    """
    universal: bool = field(default=False, kw_only=True)
    """
    Whether the match is a universal match (i.e., must match for all values of the variable/attribute) check or not.
    """
    conditions: List[ConditionType] = field(init=False, default_factory=list)
    """
    The conditions that define the match.
    """
    parent: Optional[Match] = field(init=False, default=None)
    """
    The parent match if this is a nested match.
    """
    node: Optional[RWXNode] = field(init=False, default=None)
    """
    The RWXNode representing the match expression in the match query graph.
    """
    attribute_matches: Dict[str, AttributeMatch] = field(init=False, default_factory=dict)
    """
    A dictionary mapping attribute names to their corresponding AttributeMatch instances.
    """
    _order_by: Optional[OrderByParams] = field(default=None, init=False)
    """
    Parameters for ordering the results of the query object descriptor.
    """
    _distinct: bool = field(init=False, default=False)
    """
    Whether to get distinct results or not.
    """
    _distinct_on: List[Selectable] = field(init=False, default_factory=list)
    """
    List of variables that need to be distinct together as a combination.
    """
    result_processor_data: Optional[ResultProcessorData] = field(init=False, default_factory=lambda: ResultProcessorData(An))
    """
    The result processor data to be applied to the match expression.
    """
    selectable_match: Optional[SelectableMatchExpression] = field(init=False, default=None)
    """
    The selectable match expression of this match.
    """

    def __post_init__(self):
        self.node = RWXNode(self.name, data=self)
        if self.parent:
            self.node.parent = self.parent.node

    @cached_property
    @abstractmethod
    def expression(self) -> Union[ResultQuantifier[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        ...

    def apply_result_processor(
            self, result_processor: Type[ResultProcessor[T]], **result_processor_kwargs
    ) -> Union[ResultProcessor[T], T]:
        """
        Record the result processor to be applied to the result of the match.
        """
        self.result_processor_data = ResultProcessorData(result_processor, result_processor_kwargs)
        self.resolve()
        if self.selectable_match is None:
            self.selectable_match = SelectableMatchExpression(_match_expression_=self)
        return self.selectable_match

    @abstractmethod
    def resolve(self, *args, **kwargs):
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def id(self):
        return self.node.id

    @cached_property
    def type(self) -> Optional[Type[T]]:
        """
        If type is predefined return it, else if the variable is available return its type, else return None.
        """
        if self._type is not None:
            return self._type
        if self.variable is None:
            return None
        return self.variable._type_

    @property
    def root(self) -> Match:
        return self.node.root.data

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return id(self)


@dataclass(eq=False)
class Match(AbstractMatchExpression[T]):
    """
    Construct a query that looks for the pattern provided by the type and the keyword arguments.
    Example usage where we look for an object of type Drawer with body of type Body that has the name"drawer_1":
        >>> @dataclass
        >>> class Body:
        >>>     name: str
        >>> @dataclass
        >>> class Drawer:
        >>>     body: Body
        >>> drawer = a(matching(Drawer)(body=matching(Body)(name="drawer_1")))
    """

    domain: DomainType = field(default=None, kw_only=True)
    """
    The domain to use for the variable created by the match.
    """
    kwargs: Dict[str, Any] = field(init=False, default_factory=dict)
    """
    The keyword arguments to match against.
    """
    resolved: bool = field(init=False, default=False)
    """
    Whether the match is resolved or not.
    """

    def __call__(self, **kwargs) -> Union[Self, T, CanBehaveLikeAVariable[T]]:
        """
        Update the match with new keyword arguments to constrain the type we are matching with.

        :param kwargs: The keyword arguments to match against.
        :return: The current match instance after updating it with the new keyword arguments.
        """
        self.kwargs = kwargs
        return self

    @cached_property
    def expression(self) -> Union[ResultQuantifier[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        if not self.variable:
            self.resolve()
        query_descriptor = entity(self.variable, *self.conditions)
        return self.result_processor_data.apply(query_descriptor)

    def __getattr__(self, item):
        """
        Implemented to raise the appropriate error.
        """
        if item.startswith("__"):
            raise AttributeError(item)
        raise UnquantifiedMatchError(self)

    def from_(self, domain: DomainType):
        """
        Record the domain to use for the variable created by the match.
        """
        self.domain = domain
        return self

    def resolve(
            self,
            variable: Optional[Selectable] = None,
            parent: Optional[Match] = None,
    ):
        """
        Resolve the match by creating the variable and conditions expressions.

        :param variable: An optional pre-existing variable to use for the match; if not provided, a new variable will
         be created.
        :param parent: The parent match if this is a nested match.
        :return:
        """
        if self.resolved:
            return
        self.update_fields(variable, parent)
        for attr_name, attr_assigned_value in self.kwargs.items():
            if isinstance(attr_assigned_value, SelectableMatchExpression):
                attr_assigned_value = attr_assigned_value._match_expression_
            attr_match = AttributeMatch(
                parent=self, attr_name=attr_name, assigned_value=attr_assigned_value
            )
            self.attribute_matches[attr_name] = attr_match
            if attr_match.is_an_unresolved_match:
                attr_match.resolve(self)
                self.conditions.extend(attr_match.conditions)
            else:
                condition = (
                    attr_match.infer_condition_between_attribute_and_assigned_value()
                )
                self.conditions.append(condition)
        self.resolved = True
        return self

    def update_fields(
            self,
            variable: Optional[Selectable] = None,
            parent: Optional[Match] = None,
    ):
        """
        Update the match variable, and parent.

        :param variable: The variable to use for the match.
         If None, a new variable will be created.
        :param parent: The parent match if this is a nested match.
        """

        if variable is not None:
            self.variable = variable
        elif self.variable is None:
            self.variable = let(self.type, self.domain)

        self.parent = parent

    def evaluate(self):
        """
        Evaluate the match expression and return the result.
        """
        return self.expression.evaluate()

    @property
    def name(self) -> str:
        return f"Match({self.type})"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass(eq=False)
class AttributeMatch(AbstractMatchExpression[T]):
    """
    A class representing an attribute assignment in a Match statement.
    """

    parent: AbstractMatchExpression = field(kw_only=True)
    """
    The parent match expression.
    """
    attr_name: str = field(kw_only=True)
    """
    The name of the attribute to assign the value to.
    """
    assigned_value: Optional[Union[Literal, Match]] = None
    """
    The value to assign to the attribute, which can be a Match instance or a Literal.
    """
    variable: Union[Attribute, Flatten] = field(default=None, kw_only=True)
    """
    The symbolic variable representing the attribute.
    """
    flattened_attribute: Flatten = field(init=False, default=None)
    """
    The flattened attribute if the attribute is an iterable and has been flattened.
    """

    @cached_property
    def expression(self) -> Union[ResultQuantifier[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        if not self.variable:
            self.resolve()
        return self.result_processor_data.apply(self.variable)

    def resolve(self, parent_match: Optional[Match] = None,):
        """
        Resolve the attribute assignment by creating the conditions and applying the necessary mappings
        to the attribute.

        :param parent_match: The parent match of the attribute assignment.
        """
        if self.variable is not None:
            return
        possibly_flattened_attr = self.attribute
        if self.attribute._is_iterable_ and (
                self.assigned_value.kwargs or self.is_type_filter_needed
        ):
            self.flattened_attribute = flatten(self.attribute)
            possibly_flattened_attr = self.flattened_attribute

        self.assigned_value.resolve(possibly_flattened_attr, parent_match)

        if self.is_type_filter_needed:
            self.conditions.append(
                HasType(possibly_flattened_attr, self.assigned_value.type)
            )

        self.conditions.extend(self.assigned_value.conditions)

        # Update _var_, _id_, and _node_, these are needed for the query graph, and for selection mechanics that use _var_.
        if self.flattened_attribute is None:
            self.variable = self.attribute
        else:
            self.variable = self.flattened_attribute

    def infer_condition_between_attribute_and_assigned_value(
            self,
    ) -> Union[Comparator, Exists]:
        """
        Find and return the appropriate condition for the attribute and its assigned value. This can be one of contains,
        in_, or == depending on the type of the assigned value and the type of the attribute. In addition, if the
        assigned value is a Match instance with an existential flag set, an Exists expression is created over the
         comparator condition.

        :return: A Comparator or an Exists expression representing the condition.
        """
        if self.attribute._is_iterable_ and not self.is_iterable_value:
            condition = contains(self.attribute, self.assigned_variable)
        elif not self.attribute._is_iterable_ and self.is_iterable_value:
            condition = in_(self.attribute, self.assigned_variable)
        elif (
                self.attribute._is_iterable_
                and self.is_iterable_value
                and not (
                isinstance(self.assigned_value, AbstractMatchExpression) and self.assigned_value.universal
        )
        ):
            self.flattened_attribute = flatten(self.attribute)
            condition = contains(self.assigned_variable, self.flattened_attribute)
        else:
            condition = self.attribute == self.assigned_variable

        if isinstance(self.assigned_value, AbstractMatchExpression) and self.assigned_value.existential:
            condition = exists(self.attribute, condition)

        return condition

    @cached_property
    def assigned_variable(self) -> CanBehaveLikeAVariable:
        """
        :return: The symbolic variable representing the assigned value.
        """
        return (
            self.assigned_value.variable
            if isinstance(self.assigned_value, AbstractMatchExpression)
            else self.assigned_value
        )

    @cached_property
    def attribute(self) -> Attribute:
        """
        :return: the attribute of the variable.
        :raises NoneWrappedFieldError: If the attribute does not have a WrappedField.
        """
        if self.variable is not None:
            return self.variable
        attr: Attribute = getattr(self.parent.variable, self.attr_name)
        if attr._wrapped_field_ is None:
            raise NoneWrappedFieldError(self.parent.type, self.attr_name)
        self.variable = attr
        return attr

    @property
    def is_an_unresolved_match(self) -> bool:
        """
        :return: True if the value is an unresolved Match instance, else False.
        """
        return (
                isinstance(self.assigned_value, AbstractMatchExpression) and not self.assigned_value.variable
        )

    @cached_property
    def is_iterable_value(self) -> bool:
        """
        :return: True if the value is an iterable or a Match instance with an iterable type, else False.
        """
        if isinstance(self.assigned_value, Selectable):
            return self.assigned_value._is_iterable_
        elif not isinstance(self.assigned_value, AbstractMatchExpression) and is_iterable(
                self.assigned_value
        ):
            return True
        elif (
                isinstance(self.assigned_value, AbstractMatchExpression)
                and self.assigned_value.variable._is_iterable_
        ):
            return True
        return False

    @cached_property
    def is_type_filter_needed(self):
        """
        :return: True if a type filter condition is needed for the attribute assignment, else False.
        """
        attr_type = self.type
        return (not attr_type) or (
                (self.assigned_value.type and self.assigned_value.type is not attr_type)
                and issubclass(self.assigned_value.type, attr_type)
        )

    @property
    def name(self) -> str:
        return f"{self.parent.name}.{self.attr_name}"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass(eq=False)
class Select(AbstractMatchExpression[T]):
    selected_variables: List[Selectable]
    """
    A list of selected variables.
    """

    @cached_property
    def expression(self) -> Union[ResultQuantifier[T], T]:
        if len(self.selected_variables) > 1:
            query_descriptor = set_of(self.selected_variables, *self.conditions)
        else:
            query_descriptor = entity(self.selected_variables[0], *self.conditions)
        if self._distinct:
            query_descriptor = query_descriptor.distinct(*self._distinct_on)
        if self._order_by:
            query_descriptor._order_by = self._order_by
        return self.result_processor_data.apply(query_descriptor)

    def resolve(self, *args, **kwargs):
        pass

    def where(self, *conditions: ConditionType) -> Self:
        self.conditions.extend(conditions)
        return self

    def order_by(self, variable: Selectable, descending: bool = False, key: Optional[Callable] = None) -> Self:
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
            *on: Selectable[T],
    ) -> Self:
        """
        Apply distinctness constraint to the query object descriptor results.

        :param on: The variables to be used for distinctness.
        :return: This query object descriptor.
        """
        self._distinct = True
        self._distinct_on = list(on)
        return self

    @property
    def name(self) -> str:
        return f"Select({','.join(map(str, self.selected_variables))})"


def matching(
        type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Create and return a Match instance that looks for the pattern provided by the type and the
    keyword arguments.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The Match instance.
    """
    return entity_matching(type_, None)


def match_any(
        type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Equivalent to matching(type_) but for existential checks.
    """
    match_ = matching(type_)
    match_.existential = True
    return match_


def match_all(
        type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Equivalent to matching(type_) but for universal checks.
    """
    match_ = matching(type_)
    match_.universal = True
    return match_


def select(
        *variables: Any,
) -> Match:
    """
    Select the variables to be included in the result.
    """
    if not variables:
        raise ValueError("select() requires at least one variable to be provided.")
    if isinstance(variables[0], SelectableMatchExpression):
        root = variables[0]._match_expression_.root
    else:
        root = variables[0].root
    for variable in variables:
        if isinstance(variable, SelectableMatchExpression):
            root.update_selected_variables(variable._match_expression_.variable)
        elif isinstance(variable, AbstractMatchExpression):
            root.update_selected_variables(variable.variable)
        else:
            raise WrongSelectableType(type(variable), [SelectableMatchExpression, AbstractMatchExpression])
    return root


def entity_matching(
        type_: Union[Type[T], CanBehaveLikeAVariable[T]], domain: DomainType
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Same as :py:func:`krrood.entity_query_language.match.match` but with a domain to use for the variable created
     by the match.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :param domain: The domain used for the variable created by the match.
    :return: The MatchEntity instance.
    """
    if isinstance(type_, CanBehaveLikeAVariable):
        return Match(_type=type_._type_, domain=domain, variable=type_)
    elif type_ and not isinstance(type_, type):
        return Match(_type=type_, domain=domain, variable=Literal(type_))
    return Match(_type=type_, domain=domain)
