from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Tuple, Dict, Any

import random_events_lib as rl
from typing_extensions import Self, Iterable

from .utils import SubclassJSONSerializer, CPPWrapper

EMPTY_SET_SYMBOL = "∅"


class AbstractSimpleSet(SubclassJSONSerializer, CPPWrapper):
    """
    Abstract class for simple sets.

    Simple sets are sets that can be represented as a single object.

    This class is a wrapper for the C++ class AbstractSimpleSet.
    """

    _cpp_object: rl.AbstractSimpleSet

    def intersection_with(self, other: Self) -> Self:
        """
        Form the intersection of this object with another object.

        :param other: The other SimpleSet
        :return: The intersection of this set with the other set
        """
        return self._from_cpp(self._cpp_object.intersection_with(other._cpp_object))

    def complement(self) -> SimpleSetContainer:
        """
        :return: The complement of this set as disjoint set of simple sets.
        """
        return tuple(self._from_cpp(cpp_simple_set) for cpp_simple_set in self._cpp_object.complement())

    def is_empty(self) -> bool:
        """
        :return: Rather this set is empty or not.
        """
        return self._cpp_object.is_empty()

    @abstractmethod
    def contains(self, item) -> bool:
        """
        Check if this set contains an item.
        :param item: The item to check
        :return: Rather if the item is in the set or not
        """
        raise NotImplementedError

    def __hash__(self):
        return hash(self._cpp_object)

    def non_empty_to_string(self) -> str:
        """
        :return: A string representation of this set if it is not empty.
        """
        raise NotImplementedError

    def difference_with(self, other: Self) -> SimpleSetContainer:
        """
        Form the difference of this object with another object.

        :param other: The other SimpleSet
        :return: The difference as a disjoint set of simple sets.
        """
        return self._from_cpp(self._cpp_object.difference_with(other._cpp_object))

    def to_string(self):
        """
        :return: A string representation of this set.
        """
        if self.is_empty():
            return EMPTY_SET_SYMBOL
        return self.non_empty_to_string()

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def __lt__(self, other: Self):
        return self._cpp_object < other._cpp_object

    def __eq__(self, other):
        return self._cpp_object == other._cpp_object

    @abstractmethod
    def as_composite_set(self) -> AbstractCompositeSet:
        """
        Convert this simple set to a composite set.

        :return: The composite set
        """
        raise NotImplementedError

    @abstractmethod
    def __deepcopy__(self):
        raise NotImplementedError


class AbstractCompositeSet(SubclassJSONSerializer, CPPWrapper, ABC):
    """
    Abstract class for composite sets.

    AbstractCompositeSet is a set composed of a union of simple sets.
    If any operation is called on this, the resulting union will also be disjoint and simplified.
    A simplified composite set is a set with as few simple sets in it as possible
    to represent the necessary information.

    This class wraps the C++ class AbstractCompositeSet.
    """

    _cpp_object: rl.AbstractCompositeSet

    simple_set_example: AbstractSimpleSet
    """
    An example of a simple set that is used to create new simple sets. 
    Fields that are python only are read from this instance when reading from cpp.
    """

    @property
    def simple_sets(self) -> SimpleSetContainer:
        """
        :return: The simple sets contained in the union described by this set.
        """
        return tuple(self.simple_set_example._from_cpp(cpp_object) for cpp_object in self._cpp_object.simple_sets)

    def union_with(self, other: Self) -> Self:
        """
        :param other: The other set
        :return: The union of this set with the other set
        """
        return self._from_cpp(self._cpp_object.union_with(other._cpp_object))

    def __or__(self, other: Self) -> Self:
        return self.union_with(other)

    def intersection_with(self, other: Self) -> Self:
        """
        :param other: The other set
        :return: The intersection of this set with the other set
        """
        return self._from_cpp(self._cpp_object.intersection_with(other._cpp_object))

    def __and__(self, other) -> Self:
        return self.intersection_with(other)

    def difference_with(self, other: Self) -> Self:
        """
        :param other: The other set
        :return: The difference of this set with the other set
        """
        return self._from_cpp(self._cpp_object.difference_with(other._cpp_object))

    def __sub__(self, other) -> Self:
        return self.difference_with(other)

    def complement(self) -> Self:
        """
        :return: The complement of this set
        """
        return self._from_cpp(self._cpp_object.complement())

    def __invert__(self) -> Self:
        return self.complement()

    def is_empty(self) -> bool:
        """
        :return: If this set is empty or not.
        """
        return self._cpp_object.is_empty()

    def contains(self, item) -> bool:
        """
        :param item: The item to check
        :return: If the item is in the set or not
        """
        for simple_set in self.simple_sets:
            if simple_set.contains(item):
                return True
        return False

    def __contains__(self, item) -> bool:
        return self.contains(item)

    def to_string(self) -> str:
        """
        :return: A string representation of this set.
        """
        if self.is_empty():
            return EMPTY_SET_SYMBOL
        return " u ".join([simple_set.to_string() for simple_set in self.simple_sets])

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def is_disjoint(self) -> bool:
        """
        :return: If the union described by this is disjoint or not.
        """
        return self._cpp_object.is_disjoint()

    def make_disjoint(self) -> Self:
        """
        Create an equal composite set that contains a disjoint union of simple sets.

        :return: The disjoint set.
        """
        return self._from_cpp(self._cpp_object.make_disjoint())

    def add_simple_set(self, simple_set: AbstractSimpleSet):
        """
        Add a simple set to this composite set if it is not empty.

        :param simple_set: The simple set to add
        """
        if simple_set.is_empty():
            return
        self._cpp_object.add_new_simple_set(simple_set._cpp_object)

    def simplify(self) -> Self:
        """
        Simplify this set into an equivalent, more compact version.

        :return: The simplified set
        """
        return self._from_cpp(self._cpp_object.simplify())

    def __eq__(self, other: Self) -> bool:
        return self._cpp_object == other._cpp_object

    def __hash__(self) -> int:
        return hash(tuple(self.simple_sets))

    def __iter__(self) -> Iterable[AbstractSimpleSet]:
        return iter(self.simple_sets)

    def __lt__(self, other: Self):
        """
        Compare this set with another set.
        The sets are compared by comparing the simple sets in order.
        If the pair of simple sets are equal, the next pair is compared.
        If all pairs are equal, the set with the least amount of simple sets is considered smaller.

        ..note:: This does not define a total order in the mathematical sense. In the mathematical sense, this defines
            a partial order.

        :param other: The other set
        :return: Rather this set is smaller than the other set
        """

        return self._cpp_object < other._cpp_object

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "simple_sets": [simple_set.to_json() for simple_set in self.simple_sets]}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(*[AbstractSimpleSet.from_json(simple_set) for simple_set in data["simple_sets"]])

    def __deepcopy__(self):
        return self.__class__(*[ss.__deepcopy__() for ss in self.simple_sets])


# Type definitions
SimpleSetContainer = Tuple[AbstractSimpleSet, ...]
