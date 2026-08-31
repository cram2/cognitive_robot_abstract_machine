from __future__ import annotations

import itertools
import math
import textwrap
from dataclasses import field, dataclass
from typing import Any, Iterable, Self, Dict, Optional, Tuple, Mapping

from sortedcontainers import SortedDict, SortedValuesView, SortedSet
from typing_extensions import List, TYPE_CHECKING, Union, Set as teSet

from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from random_events.sigma_algebra import AbstractSimpleSet, AbstractCompositeSet
from random_events.variable import Variable, Continuous
import random_events_lib as rl

# Type definitions
if TYPE_CHECKING:
    VariableMapSuperClassType = SortedDict[Variable, Any]
else:
    VariableMapSuperClassType = SortedDict

VariableMapKey = Union[str, Variable]
VariableSet = teSet[Variable]


class VariableMap(VariableMapSuperClassType):
    """
    A map from variables to values.

    Accessing a variable by name is also supported.
    """

    @property
    def variables(self) -> Iterable[Variable]:
        return self.keys()

    @property
    def assignments(self) -> SortedValuesView:
        return self.values()

    def get_variable(self, key: VariableMapKey) -> Variable:
        """
        Get the variable matching the key.

        :param key: The variable or its name.
        :return: The matching variable.
        """

        if isinstance(key, Variable):
            return key

        variable = [variable for variable in self.variables if variable.name == key]
        if len(variable) == 0:
            raise KeyError(f"Variable {key} not found in event {self}")
        return variable[0]

    def __getitem__(self, key: Union[str, Variable]):
        return super().__getitem__(self.get_variable(key))

    def __setitem__(self, key: Union[str, Variable], value: Any):
        super().__setitem__(self.get_variable(key), value)

    def __copy__(self):
        return self.__class__({variable: value for variable, value in self.items()})


@dataclass(eq=False)
class SimpleEvent(AbstractSimpleSet, VariableMap):
    """
    A simple event is a set of assignments of variables to values.

    A simple event is logically equivalent to a conjunction of assignments.

    .. attention::
        Use :py:func:`from_data` class method to create a simple event from a dictionary, do not use the constructor directly.
    """

    cpp_object: rl.SimpleEvent = field(default_factory=rl.SimpleEvent)

    @classmethod
    def from_data(cls, *args, **kwargs) -> Self:
        instance = cls.__new__(cls)
        VariableMap.__init__(instance, *args, **kwargs)
        for key, value in instance.items():
            instance._setitem_without_cpp(key, value)
        instance._update_cpp_object()
        return instance

    def _update_cpp_object(self):
        self.cpp_object = rl.SimpleEvent(
            {variable.cpp_object: value.cpp_object for variable, value in self.items()}
        )

    def _setitem_without_cpp(self, key: VariableMapKey, value: Any):
        """
        See __setitem__ for more information.
        """
        key = self.get_variable(key)
        value = key.make_value(value)
        super().__setitem__(key, value)

    def _from_cpp(self, cpp_object: rl.SimpleEvent):
        variables = [
            variable
            for variable in self.variables
            if variable.cpp_object in cpp_object.variable_map
        ]
        result = {
            variable: self[variable]._from_cpp(
                cpp_object.variable_map[variable.cpp_object]
            )
            for variable in variables
        }
        return SimpleEvent.from_data(result)

    def as_composite_set(self) -> Event:
        return Event.from_simple_sets(self)

    def contains(self, item: Tuple) -> bool:
        for assignment, value in zip(self.assignments, item):
            if not assignment.contains(value):
                return False
        return True

    @property
    def size(self) -> float:
        """
        The size of the box this event describes, as the product over its variables of
        the size of the values it assigns them.

        A variable left out of the event is left out of the product, so the size is the
        one taken over the variables the event constrains. An event that constrains no
        variable at all is the empty event rather than an empty product, so it has no
        size.

        :return: The size of this event, which is zero as soon as one variable is
            assigned nothing of size and infinite when one is unbounded.
        """
        if self.is_empty():
            return 0.0
        factors = [assignment.size for assignment in self.assignments]
        if any(factor == 0.0 for factor in factors):
            return 0.0
        return math.prod(factors)

    def __setitem__(self, key: VariableMapKey, value: Any):
        """
        Set the value of a variable in the event.
        Also allows for assigning variables to values outside the classes of this package.
        If this is the case, this tries to convert the value to a CompositeSet.

        :param key: The variable (or its name) to set the value for
        :param value: The value to set
        """
        key = self.get_variable(key)
        self._setitem_without_cpp(key, value)
        self._update_cpp_object()

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        items = itertools.chain(self.items(), other.items())
        return self.__class__.from_data({variable: value for variable, value in items})

    def marginal(self, variables: VariableSet) -> SimpleEvent:
        """
        Create the marginal event, that only contains the variables given.

        :param variables: The variables to contain in the marginal event
        :return: The marginal event
        """
        return self._from_cpp(
            self.cpp_object.marginal({variable.cpp_object for variable in variables})
        )

    def non_empty_to_string(self) -> str:
        return (
            "{\n"
            + textwrap.indent(
                ",\n".join(
                    f"{variable.name} ∈ {assignment}"
                    for variable, assignment in self.items()
                ),
                "    ",
            )
            + "\n}"
        )

    def variables_to_json(self) -> List:
        return [to_json(variable) for variable in self.keys()]

    def assignments_to_json(self) -> List:
        return [assignment.to_json() for assignment in self.values()]

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "variables": self.variables_to_json(),
            "assignments": self.assignments_to_json(),
        }

    def to_json_assignments_only(self) -> Dict[str, Any]:
        return {**super().to_json(), "assignments": self.assignments_to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        variables = [from_json(variable) for variable in data["variables"]]
        assignments = [
            AbstractCompositeSet.from_json(assignment)
            for assignment in data["assignments"]
        ]
        return cls.from_data(
            {
                variable: assignment
                for variable, assignment in zip(variables, assignments)
            }
        )

    @classmethod
    def from_json_given_variables(
        cls, data: Dict[str, Any], variables: List[Variable]
    ) -> Self:
        assignments = [
            AbstractCompositeSet.from_json(assignment)
            for assignment in data["assignments"]
        ]
        return cls.from_data(
            {
                variable: assignment
                for variable, assignment in zip(variables, assignments)
            }
        )

    def update_variables(self, new_variables: Dict[Variable, Variable]) -> Self:
        """
        Construct a new simple event where the own variables are replaced with the new variables.
        If the new variables are missing mappings, the old variables are kept for the missing updates.

        :param new_variables: A dictionary mapping current variables to new variables
        :return: A new SimpleEvent with the updated variables
        """
        return SimpleEvent.from_data(
            {
                new_variables.get(variable, variable): value
                for variable, value in self.items()
            }
        )

    def fill_missing_variables(self, variables: Iterable[Variable]):
        """
        Fill this with the variables that are not in self but in `variables` in-place.
        The variables are mapped to their domain.

        :param variables: The variables to fill the event with
        """
        missing = [v for v in variables if v not in self]
        if not missing:
            return
        for variable in missing:
            self._setitem_without_cpp(variable, variable.domain)
        self._update_cpp_object()

    def fill_missing_variables_pure(self, variables: Iterable[Variable]):
        """
        Fill this with the variables that are not in self but in `variables`.
        The variables are mapped to their domain.
        """
        return SimpleEvent.from_data(
            {variable: self.get(variable, variable.domain) for variable in variables}
        )

    def __deepcopy__(self):
        return SimpleEvent.from_data(
            {
                variable: assignment.__deepcopy__()
                for variable, assignment in self.items()
            }
        )


@dataclass(eq=False)
class Event(AbstractCompositeSet):
    """
    An event is a disjoint set of simple events.

    Every simple event added to this event that is missing variables that any other event in this event has, will be
    extended with the missing variable. The missing variables are mapped to their domain.

    .. attention:
        Use :py:func:`from_simple_sets` class method to create an event from a list of simple events, do not use the constructor directly.
    """

    cpp_object: rl.Event = field(default_factory=lambda: rl.Event(set()))
    simple_set_example: SimpleEvent = field(init=False)

    _variables: Optional[SortedSet] = field(init=False, default=None)
    """
    Cache for the variables of the event.
    """

    @classmethod
    def from_simple_sets(cls, *simple_sets: SimpleEvent):
        if isinstance(simple_sets, SimpleEvent):
            simple_sets = (simple_sets,)
        instance = cls.__new__(cls)
        instance._variables = None

        if not simple_sets:
            instance.simple_set_example = SimpleEvent.from_data()
            instance.cpp_object = rl.Event(set())
            instance._variables = SortedSet()
            return instance

        # Compute the union of all variables from Python-side inputs — no C++ round-trip.
        all_variables = SortedSet(
            variable for simple_set in simple_sets for variable in simple_set.variables
        )

        # Fill missing variables in each input SimpleEvent before building the C++ Event,
        # so every cpp_object is up-to-date when we pass it to rl.Event.
        for simple_set in simple_sets:
            simple_set.fill_missing_variables(all_variables)

        instance.simple_set_example = simple_sets[0]
        instance.cpp_object = rl.Event(
            {simple_set.cpp_object for simple_set in simple_sets}
        )
        instance._variables = all_variables
        return instance

    def _from_cpp(self, cpp_object: rl.Event) -> Event:
        # O(1) fast path: reuse the existing simple_set_example and variable cache.
        # Correct for all operations that preserve the variable set (make_disjoint,
        # simplify, difference_with, complement, intersection_with).
        instance = Event.__new__(Event)
        instance.cpp_object = cpp_object
        instance.simple_set_example = self.simple_set_example
        instance._variables = self._variables
        return instance

    @property
    def simple_sets(self) -> Tuple[SimpleEvent, ...]:
        return super().simple_sets

    @property
    def variables(self) -> SortedSet:
        if self._variables is not None:
            return self._variables
        # Fallback: materialise from C++ and cache (edge cases / legacy callers).
        variables_set = SortedSet(
            variable
            for simple_set in self.simple_sets
            for variable in simple_set.variables
        )
        self._variables = variables_set
        return variables_set

    def get_variable(self, key: VariableMapKey) -> Variable:
        """
        Get the variable matching the key.

        :param key: The variable or its name.
        :return: The matching variable.
        """

        if isinstance(key, Variable):
            return key

        variable = [variable for variable in self.variables if variable.name == key]
        if len(variable) == 0:
            raise KeyError(f"Variable {key} not found in event {self}")
        return variable[0]

    def update_simple_set_example(self):
        """
        Update the simple set example to the first simple set in the event.
        Use this whenever the simple sets change in-place
        """
        simple_sets = self.simple_sets
        self.simple_set_example = (
            simple_sets[0] if simple_sets else SimpleEvent.from_data()
        )

    def fill_missing_variables(self, variables: Optional[Iterable[Variable]] = None):
        """
        Fill all simple sets with the missing variables in-place.

        :param variables: The variables to fill the event with. If None, all variables are used.
        """
        if variables is None:
            variables = set()

        # Use the cached variable set to avoid materialising simple_sets from C++.
        cached = self._variables
        if cached is not None:
            all_variables = cached | SortedSet(variables)
        else:
            all_variables = self.variables | SortedSet(variables)

        self.simple_set_example.fill_missing_variables(all_variables)
        self.simple_set_example.cpp_object.fill_missing_variables(
            {variable.cpp_object for variable in all_variables}
        )
        self.cpp_object.fill_missing_variables(
            {variable.cpp_object for variable in all_variables}
        )
        self._variables = SortedSet(all_variables)

    def fill_missing_variables_pure(
        self, variables: Optional[Iterable[Variable]] = None
    ):
        """
        Fill all simple sets with the missing variables.

        :param variables: The variables to fill the event with.
        """

        if variables is None:
            variables = set()

        all_variables = self.variables | SortedSet(variables)

        return Event.from_simple_sets(
            *[
                simple_set.fill_missing_variables_pure(all_variables)
                for simple_set in self.simple_sets
            ]
        )

    def union_with(self, other: Self) -> Self:
        """
        :param other: The other event
        :return: The union of this event with the other event
        """
        result = self._from_cpp(self.cpp_object.union_with(other.cpp_object))
        # If the two events have different variable sets, update _variables to the union.
        if (
            other._variables is not None
            and self._variables is not None
            and other._variables != self._variables
        ):
            result._variables = self._variables | other._variables
        return result

    def marginal(self, variables: VariableSet) -> Event:
        """
        Create the marginal event, that only contains the variables given.

        :param variables: The variables to contain in the marginal event
        :return: The marginal event
        """
        result_cpp = self.cpp_object.marginal(
            {self.get_variable(v.name).cpp_object for v in variables}
        )
        instance = Event.__new__(Event)
        instance.cpp_object = result_cpp
        instance.simple_set_example = self.simple_set_example
        # Resolve variable names to Python Variable objects from our cached set.
        instance._variables = SortedSet(self.get_variable(v.name) for v in variables)
        return instance

    def bounding_box(self) -> SimpleEvent:
        """
        Compute the bounding box of the event.
        The bounding box is the smallest simple event that contains this event. It is computed by taking the union
        of all simple events variable wise.

        :return: The bounding box as a simple event
        """
        simple_sets = self.simple_sets
        if not simple_sets:
            return SimpleEvent.from_data()
        result = {}
        variables = SortedSet(v for ss in simple_sets for v in ss.variables)
        for variable in variables:
            for simple_set in simple_sets:
                if variable not in result:
                    result[variable] = simple_set[variable]
                else:
                    result[variable] = result[variable].union_with(simple_set[variable])
        return SimpleEvent.from_data(result)

    def update_variables(self, new_variables: Dict[Variable, Variable]) -> Event:
        """
        see :func:`~random_events.product_algebra.SimpleEvent.update_variables`
        """
        return Event.from_simple_sets(
            *[
                simple_event.update_variables(new_variables)
                for simple_event in self.simple_sets
            ]
        )

    def add_simple_set(self, simple_set: AbstractSimpleSet):
        """
        Add a simple set to this event.

        :param simple_set: The simple set to add
        """
        super().add_simple_set(simple_set)
        self._variables = None  # Invalidate cache; new simple set may add variables
        self.fill_missing_variables()

    def to_json(self) -> Dict[str, Any]:
        variables = [to_json(variable) for variable in self.variables]
        simple_sets = [
            simple_set.to_json_assignments_only() for simple_set in self.simple_sets
        ]
        return {
            **SubclassJSONSerializer.to_json(self),
            "variables": variables,
            "simple_sets": simple_sets,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        variables = [from_json(variable) for variable in data["variables"]]
        simple_sets = [
            SimpleEvent.from_json_given_variables(simple_set, variables)
            for simple_set in data["simple_sets"]
        ]
        return cls.from_simple_sets(*simple_sets)
