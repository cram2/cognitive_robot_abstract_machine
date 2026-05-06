import enum
import types
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import assert_never

import random_events_lib as rl
from typing_extensions import Self, Dict, Any, Optional, Iterable, Type, Union, List

from random_events.interval import reals, Interval, closed, singleton, SimpleInterval
from random_events.set import Set, SetElement
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.utils import CPPWrapper
from krrood.adapters.json_serializer import SubclassJSONSerializer

compatible_types = (
    int,
    float,
    bool,
    enum.Enum,
)  # types that can be expressed variable


@dataclass
class Variable(CPPWrapper):
    """
    Parent class for all random variables.
    """

    name: str
    """
    The name of the variable.
    """

    domain: AbstractCompositeSet = field(kw_only=True, default=None)
    """
    The domain of the variable.
    The domain is a composite set that can be used to create values of the variable.
    """

    def __lt__(self, other: Self) -> bool:
        return self.cpp_object < other.cpp_object

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, other):
        return self.cpp_object == other.cpp_object

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.domain})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @property
    @abstractmethod
    def is_numeric(self):
        """
        :return: Rather, this variable has a numeric domain or not.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_cpp(cls, cpp_object):
        """
        Create a variable from a C++ object.
        """
        if type(cpp_object) == rl.Symbolic:
            return Symbolic._from_cpp(cpp_object)
        elif type(cpp_object) == rl.Continuous:
            return Continuous._from_cpp(cpp_object)
        elif type(cpp_object) == rl.Integer:
            return Integer._from_cpp(cpp_object)

    @abstractmethod
    def make_value(self, value: Any) -> AbstractCompositeSet:
        """
        Create a value of the domain from an arbitrary value.
        This method tries to parse the value and wrap it in a composite set.

        :param value: The value.
        :return: The value wrapped in a composite set.
        """
        raise NotImplementedError


@dataclass(eq=False, repr=False)
class Continuous(Variable):
    """
    Class for continuous random variables.

    The domain of a continuous variable is the real line.
    """

    domain: Interval = field(kw_only=True, default=reals())

    def __post_init__(self):
        self.cpp_object = rl.Continuous(self.name)

    @property
    def is_numeric(self):
        return True

    @classmethod
    def _from_cpp(cls, cpp_object):
        return cls(cpp_object.name)

    def make_value(self, value: Any) -> Interval:
        if isinstance(value, (int, float)):
            return singleton(value)
        elif isinstance(value, SimpleInterval):
            return value.as_composite_set()
        elif isinstance(value, Interval):
            return value
        elif isinstance(value, Iterable):
            return closed(*value)
        else:
            raise ValueError(f"Value {value} cannot be parsed to an interval.")


@dataclass(eq=False, repr=False)
class Symbolic(Variable):
    """
    Class for unordered, finite, discrete random variables.

    The domain of a symbolic variable is a set.
    """

    domain: Set = field(kw_only=True)

    def __post_init__(self):
        self.domain = self.domain.__deepcopy__()
        self.cpp_object = rl.Symbolic(self.name, self.domain.cpp_object)

    @property
    def is_numeric(self):
        return False

    @classmethod
    def _from_cpp(cls, cpp_object):
        return cls(
            name=cpp_object.name, domain=cpp_object.domain._from_cpp(cpp_object.domain)
        )

    def make_value(self, value) -> Set:
        if not isinstance(value, Iterable):
            value = [value]

        parsed_value = []

        # try to match the values to the set elements
        for v in value:

            # if the value is already a SetElement, append it
            if isinstance(v, SetElement):
                parsed_value.append(v)

            # if not, try to find the matching element
            else:
                matches = [
                    elem for elem in self.domain.simple_sets if elem.element == v
                ]
                if len(matches) == 0:
                    raise ValueError(
                        f"Value {value} not in domain of variable {self}. "
                        f"Domain is {self.domain}"
                    )
                parsed_value += matches

        return Set.from_simple_sets(*parsed_value)


@dataclass(eq=False, repr=False)
class Integer(Variable):
    """
    Class for ordered, discrete random variables.

    The domain of an integer variable is the number line.
    """

    domain: Interval = field(kw_only=True, default=reals())

    def __post_init__(self):
        self.cpp_object = rl.Integer(self.name)

    @property
    def is_numeric(self):
        return True

    @classmethod
    def _from_cpp(cls, cpp_object):
        return cls(name=cpp_object.name)

    def make_value(self, value: Any) -> Interval:
        return Continuous.make_value(self, value)


def variable_from_name_and_type(name: str, type_: Type) -> Variable:
    """
    Create a variable from a name and type.

    :param name: The name of the variable
    :param type_: The type of the variable
    :return: The created variable
    """

    if issubclass(type_, enum.Enum):
        result = Symbolic(name=name, domain=Set.from_iterable(type_))
    elif issubclass(type_, bool):
        result = Symbolic(name=name, domain=Set.from_iterable([True, False]))
    elif issubclass(type_, int):
        result = Integer(name)
    elif issubclass(type_, float):
        result = Continuous(name)
    else:
        assert_never((name, type_))

    return result


def most_appropriate_variable_type(
    union: Iterable[Type],
) -> Optional[Union[*compatible_types]]:
    """
    Get the most appropriate type for a random events variable from a union of types.
    The most appropriate type is the one, where the mathematical interpretation as set has the highest cardinality.

    :param union: The union of types.
    :return: The most appropriate type.
    """

    if float in union:
        return float

    elif int in union:
        return int

    elif enum.Enum in union:
        return enum.Enum

    elif bool in union:
        return bool
    else:
        return None
