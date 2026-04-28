from __future__ import annotations

from functools import cached_property

from typing_extensions import Hashable, Optional
from typing import Any, Iterable, Self, Dict, Optional, Tuple
from dataclasses import field, dataclass

from krrood.adapters.json_serializer import to_json, from_json
from random_events.sigma_algebra import AbstractSimpleSet, AbstractCompositeSet
import random_events_lib as rl

# Type Definitions
HashMap: Dict[int, Hashable]
AllElements: Tuple[Hashable]


@dataclass(eq=False)
class SetElement(AbstractSimpleSet):
    """
    Represents a SetElement.
    A SetElement consists of an element and all possible elements.
    It is necessary to know of all possible elements to determine the index and complement of any element.
    All elements are a tuple to preserve ordering.

    Beware that an empty set element is an invariant of this class and is represented by None.
    All elements are not consistent with invariants of this class.

    This class is a wrapper for the C++ class SetElement.
    The elements in the C++ class are represented by their index in the all_elements tuple.
    The C++ object gets as all elements the hash values of all elements.
    A hash map is created to map the hash of each element to the element.

    .. attention::
        Use :py:func:`from_data` class method to create a set element from a dictionary, do not use the constructor directly.
    """

    cpp_object: rl.SetElement = field(default_factory=lambda: rl.SetElement(set()))

    element: Hashable = field(init=False)
    """
    The element.
    """

    all_elements: AllElements = field(init=False)
    """
    The set of all elements.
    """

    @classmethod
    def from_data(cls, element: Hashable, all_elements: AllElements) -> Self:
        """
        Create a set element from data.
        :param element: The element to create the set element from. If None, create an empty set element.
        :param all_elements: The set of all elements.
        """
        instance = cls.__new__(cls)
        instance.all_elements = all_elements
        if element is not None and element not in all_elements:
            raise ValueError(
                f"Element {element} is not in the set of all elements. "
                f"All elements: {all_elements}"
            )
        if element is None:
            instance.cpp_object = rl.SetElement(set())
        else:
            if not isinstance(all_elements, Tuple):
                instance.all_elements = tuple(all_elements)
            element_index = instance.all_elements.index(element)
            instance.cpp_object = rl.SetElement(
                element_index, set(instance.hash_map.keys())
            )
            instance.element = element
        return instance

    @cached_property
    def hash_map(self) -> HashMap:
        """
        :return:A map that maps the hashes of each element in all_elements to the element.
        """
        return {hash(elem): elem for elem in self.all_elements}

    def _from_cpp(self, cpp_object):
        if cpp_object.element_index == -1:
            return SetElement.from_data(None, set())
        return SetElement.from_data(
            self.all_elements[cpp_object.element_index], self.all_elements
        )

    def contains(self, item: Self) -> bool:
        return self == item

    def non_empty_to_string(self) -> str:
        return str(self.element)

    def __hash__(self):
        return hash(self.element)

    def __repr__(self):
        return self.non_empty_to_string()

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "value": to_json(self.element),
            "content": to_json(list(self.all_elements)),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls.from_data(from_json(data["value"]), from_json(data["content"]))

    def as_composite_set(self) -> AbstractCompositeSet:
        return Set.from_simple_sets(self)

    def __deepcopy__(self):
        return SetElement.from_data(self.element, self.all_elements)


@dataclass(eq=False)
class Set(AbstractCompositeSet):
    """
    Represents a set.

    A set is a union of simple sets.

    A set is simplified if no element is contained twice.

    This class is a wrapper for the C++ class Set.

    Beware that an empty set is an invariant of this class.
    All elements are not consistent with invariants of this class.

    .. attention::
        Use :py:func:`from_simple_sets` class method to create a set from a list of simple sets, do not use the constructor directly.
    """

    cpp_object: rl.Set = field(default_factory=lambda: rl.Set(set(), set()))
    simple_set_example: SetElement = field(init=False)
    all_elements: Tuple[Hashable] = field(init=False)

    @classmethod
    def from_simple_sets(cls, *simple_sets: SetElement) -> Self:
        instance = cls.__new__(cls)
        if len(simple_sets) > 0:
            instance.simple_set_example = simple_sets[0]
            instance.cpp_object = rl.Set(
                {simple_set.cpp_object for simple_set in simple_sets},
                instance.simple_set_example.cpp_object.all_elements,
            )
            instance.all_elements = simple_sets[0].all_elements

        else:
            instance.cpp_object = rl.Set(set(), set())
            instance.all_elements = tuple()
        return instance

    def _from_cpp(self, cpp_object):
        return Set.from_simple_sets(
            *[
                self.simple_set_example._from_cpp(cpp_simple_set)
                for cpp_simple_set in cpp_object.simple_sets
            ]
        )

    @classmethod
    def from_iterable(cls, iterable: Iterable) -> Self:
        all_elements = tuple(iterable)
        return cls.from_simple_sets(
            *[SetElement.from_data(elem, all_elements) for elem in all_elements]
        )

    @property
    def hash_map(self) -> HashMap:
        """
        :return: A map that maps the hashes of each element in all_elements to the element.
        """
        return {hash(elem): elem for elem in self.all_elements}
