from abc import abstractmethod

import random_events_lib as rl
from typing_extensions import Self, Dict, Any, Optional, Iterable

from .interval import reals, Interval, closed, singleton, SimpleInterval
from .set import Set, SetElement
from .sigma_algebra import AbstractCompositeSet
from .utils import SubclassJSONSerializer, CPPWrapper


class Variable(SubclassJSONSerializer, CPPWrapper):
    """
    Parent class for all random variables.
    """

    name: str
    """
    The name of the variable.
    """

    domain: AbstractCompositeSet
    """
    The domain of the variable.
    """

    def __init__(self, name: str, domain: AbstractCompositeSet):
        """
        Construct a new random variable.

        :param name: The name of the variable.
        :param domain: The domain of the variable.
        """
        self.name = name
        self.domain = domain

    def __lt__(self, other: Self) -> bool:
        return self._cpp_object < other._cpp_object

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, other):
        return self._cpp_object == other._cpp_object

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.domain})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "name": self.name, "domain": self.domain.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["name"], AbstractCompositeSet.from_json(data["domain"]))

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


class Continuous(Variable):
    """
    Class for continuous random variables.

    The domain of a continuous variable is the real line.
    """

    def __init__(self, name: str, *args, **kwargs):
        """
        Construct a continuous variable.

        :param name: The name.
        """
        super().__init__(name, reals())

        self._cpp_object = rl.Continuous(self.name)

    @property
    def is_numeric(self):
        return True

    def _from_cpp(self, cpp_object):
        return self.__class__(cpp_object.name)

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


class Symbolic(Variable):
    """
    Class for unordered, finite, discrete random variables.

    The domain of a symbolic variable is a set.
    """

    domain: Set

    def __init__(self, name: Optional[str], domain: Set):
        """
        Construct a symbolic variable.

        :param name: The name.
        :param domain: The domain of the variable.
        """
        # TODO this deepcopy here needs investigation.
        #  The bug seems to be that not doing a deepcopy here destroys the domain as soon as the object is deleted.
        super().__init__(name, domain.__deepcopy__())

        self._cpp_object = rl.Symbolic(self.name, self.domain._cpp_object)

    @property
    def is_numeric(self):
        return False

    def _from_cpp(self, cpp_object):
        return self.__class__(cpp_object.name, self.domain._from_cpp(cpp_object.domain))

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
                matches = [elem for elem in self.domain.simple_sets if elem.element == v]
                if len(matches) == 0:
                    raise ValueError(f"Value {value} not in domain of variable {self}. "
                                     f"Domain is {self.domain}")
                parsed_value += matches

        return Set(*parsed_value)


class Integer(Variable):
    """
    Class for ordered, discrete random variables.

    The domain of an integer variable is the number line.
    """

    def __init__(self, name: str, domain: AbstractCompositeSet = reals()):
        """
        Construct an integer variable.
        :param name: The name.
        :param domain: The domain of the variable.
        """
        super().__init__(name, reals())

        self._cpp_object = rl.Integer(self.name)

    @property
    def is_numeric(self):
        return True

    def _from_cpp(self, cpp_object):
        return self.__class__(cpp_object.name)

    def make_value(self, value: Any) -> Interval:
        return Continuous.make_value(self, value)
