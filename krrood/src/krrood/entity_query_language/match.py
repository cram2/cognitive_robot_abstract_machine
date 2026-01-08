from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import (
    Optional,
    Type,
    Dict,
    Any,
    List,
    Union,
    Self,
    Iterable,
    Generic,
    Callable,
)

from .entity import (
    ConditionType,
    contains,
    in_,
    flatten,
    variable,
    set_of,
    entity,
    exists,
)
from .failures import (
    NoneWrappedFieldError,
    WrongSelectableType,
    UsageError,
    NoKwargsInMatchVar,
)
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
    DomainType,
    ResultProcessor,
    OrderByParams,
)
from .utils import is_iterable, T


@dataclass
class AbstractMatchExpression(Generic[T], ABC):
    """
    Abstract base class for constructing and handling a match expression.

    This class is intended to provide a framework for defining and managing match expressions,
    which are used to structural pattern matching in the form of nested match expressions with keyword arguments.
    """

    type_: Optional[Type[T]] = field(default=None, kw_only=True)
    """
    The type of the variable.
    """
    variable: Optional[Selectable[T]] = field(default=None, kw_only=True)
    """
    The created variable from the type and kwargs.
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
    resolved: bool = field(init=False, default=False)
    """
    Whether the match is resolved or not.
    """

    def __post_init__(self):
        self.node = RWXNode(self.name, data=self)
        if self.parent:
            self.node.parent = self.parent.node

    @cached_property
    @abstractmethod
    def expression(self) -> Union[CanBehaveLikeAVariable[T], T]:
        """
        :return: the entity expression corresponding to the match query.
        """
        ...

    def resolve(self, *args, **kwargs):
        """
        Resolve the match by creating the variable and conditions expressions.
        """
        if self.resolved:
            return
        self._resolve(*args, **kwargs)
        self.resolved = True

    @abstractmethod
    def _resolve(self, *args, **kwargs):
        """
        This method serves as an abstract definition to be implemented by subclasses,
        aimed at handling specific resolution logic for the derived class. The method
        is designed to be flexible in accepting any number and type of input
        parameters through positional (*args) and keyword (**kwargs) arguments. Subclasses
        must extend this method to provide concrete implementations tailored to their
        unique behaviors and requirements.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def id(self):
        return self.node.id

    @cached_property
    def type(self) -> Optional[Type[T]]:
        """
        If type is predefined return it, else if the variable is available return its type, else return None.
        """
        if self.type_ is not None:
            return self.type_
        if self.variable is None:
            return None
        return self.variable._type_

    @property
    def root(self) -> Match:
        """
        :return: The root match expression.
        """
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
        >>> drawer = match_variable(Drawer, domain=None)(body=match(Body)(name="drawer_1")))
    """

    kwargs: Dict[str, Any] = field(init=False, default_factory=dict)
    """
    The keyword arguments to match against.
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
    def expression(self) -> Union[An[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        if not self.variable:
            self.resolve()
        entity_ = entity(self.variable)
        if self.conditions:
            entity_ = entity_.where(*self.conditions)
        return An(entity_)

    def _resolve(
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
        self.update_fields(variable, parent)
        for attr_name, attr_assigned_value in self.kwargs.items():
            attr_match = AttributeMatch(
                parent=self,
                attribute_name=attr_name,
                assigned_value=attr_assigned_value,
            )
            attr_match.resolve()
            self.conditions.extend(attr_match.conditions)

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
            self.create_variable()

        self.parent = parent

    def create_variable(self):
        self.variable = variable(self.type, domain=None)

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
class MatchVariable(Match[T]):
    """
    Represents a match variable that operates within a specified domain.

    A class designed to create and manage a variable constrained by a defined
    domain. It provides functionality to add additional constraints via
    keyword arguments and return an expression representing the resolved
    constraints.
    """

    domain: DomainType = field(default=None, kw_only=True)
    """
    The domain to use for the variable created by the match.
    """

    def create_variable(self):
        self.variable = variable(self.type, domain=self.domain)

    def __call__(self, **kwargs) -> Union[An[T], T]:
        """
        Add kwargs constraints and return the resolved expression as An() instance.
        """
        if not kwargs:
            raise NoKwargsInMatchVar(self)
        super().__call__(**kwargs)
        return self.expression


@dataclass(eq=False)
class AttributeMatch(AbstractMatchExpression[T]):
    """
    A class representing an attribute assignment in a Match statement.
    """

    parent: AbstractMatchExpression = field(kw_only=True)
    """
    The parent match expression.
    """
    attribute_name: str = field(kw_only=True)
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

    @cached_property
    def expression(self) -> Union[CanBehaveLikeAVariable[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        if not self.variable:
            self.resolve()
        return self.variable

    def _resolve(
        self,
        parent_match: Optional[Match] = None,
    ):
        """
        Resolve the attribute assignment by creating the conditions and applying the necessary mappings
        to the attribute.

        :param parent_match: The parent match of the attribute assignment.
        """
        if not isinstance(self.assigned_value, AbstractMatchExpression) or (
            self.assigned_value.variable or self.assigned_value.resolved
        ):
            self.conditions.append(self.attribute == self.assigned_variable)
            return
        self.assigned_value.resolve(self.attribute, parent_match)

        if self.is_type_filter_needed:
            self.conditions.append(HasType(self.attribute, self.assigned_value.type))

        self.conditions.extend(self.assigned_value.conditions)

    @cached_property
    def assigned_variable(self) -> Selectable:
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
        attr: Attribute = getattr(self.parent.variable, self.attribute_name)
        self.variable = attr
        return attr

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
        return f"{self.parent.name}.{self.attribute_name}"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


def match(
    type_: Optional[Union[Type[T], Selectable[T]]] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Create a symbolic variable matching the type and the provided keyword arguments. This is used for easy variable
     definitions when there are structural constraints.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The Match instance.
    """
    return Match(type_=type_)


def match_variable(
    type_: Union[Type[T], Selectable[T]], domain: DomainType
) -> Union[Type[T], An[T], CanBehaveLikeAVariable[T], MatchVariable[T]]:
    """
    Same as :py:func:`krrood.entity_query_language.match.match` but with a domain to use for the variable created
     by the match.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :param domain: The domain used for the variable created by the match.
    :return: The Match instance.
    """
    return MatchVariable(type_=type_, domain=domain)
