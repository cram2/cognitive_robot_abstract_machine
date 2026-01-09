from __future__ import annotations

import itertools
import textwrap

import numpy as np
import plotly.graph_objects as go
from sortedcontainers import SortedDict, SortedValuesView, SortedSet
from typing_extensions import List, TYPE_CHECKING, Union, Set as teSet

from .sigma_algebra import *
from .variable import *

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


class SimpleEvent(AbstractSimpleSet, VariableMap):
    """
    A simple event is a set of assignments of variables to values.

    A simple event is logically equivalent to a conjunction of assignments.
    """

    _cpp_object: rl.SimpleEvent

    def __init__(self, *args, **kwargs):
        """
        Create a new simple event.

        :param args: The assignments of variables to values
        """
        VariableMap.__init__(self, *args, **kwargs)
        for key, value in self.items():
            self._setitem_without_cpp(key, value)
        self._update_cpp_object()

    def _update_cpp_object(self):
        self._cpp_object = rl.SimpleEvent({variable._cpp_object: value._cpp_object for variable, value in self.items()})

    def _setitem_without_cpp(self, key: VariableMapKey, value: Any):
        """
        See __setitem__ for more information.
        """
        key = self.get_variable(key)
        value = key.make_value(value)
        super().__setitem__(key, value)

    def _from_cpp(self, cpp_object: rl.SimpleEvent):
        variables = [variable for variable in self.variables if variable._cpp_object in cpp_object.variable_map]
        result = {variable: self[variable]._from_cpp(cpp_object.variable_map[variable._cpp_object]) for variable in
                  variables}
        return SimpleEvent(result)

    def as_composite_set(self) -> Event:
        return Event(self)

    def contains(self, item: Tuple) -> bool:
        for assignment, value in zip(self.assignments, item):
            if not assignment.contains(value):
                return False
        return True

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

    def marginal(self, variables: VariableSet) -> SimpleEvent:
        """
        Create the marginal event, that only contains the variables given.

        :param variables: The variables to contain in the marginal event
        :return: The marginal event
        """
        return self._from_cpp(self._cpp_object.marginal({variable._cpp_object for variable in variables}))

    def non_empty_to_string(self) -> str:
        return ("{\n" + textwrap.indent(
            ",\n".join(f"{variable.name} ∈ {assignment}" for variable, assignment in self.items()), "    ") + "\n}")

    def variables_to_json(self) -> List:
        return [variable.to_json() for variable in self.keys()]

    def assignments_to_json(self) -> List:
        return [assignment.to_json() for assignment in self.values()]

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "variables": self.variables_to_json(), "assignments": self.assignments_to_json()}

    def to_json_assignments_only(self) -> Dict[str, Any]:
        return {**super().to_json(), "assignments": self.assignments_to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variables = [Variable.from_json(variable) for variable in data["variables"]]
        assignments = [AbstractCompositeSet.from_json(assignment) for assignment in data["assignments"]]
        return cls({variable: assignment for variable, assignment in zip(variables, assignments)})

    @classmethod
    def from_json_given_variables(cls, data: Dict[str, Any], variables: List[Variable]) -> Self:
        assignments = [AbstractCompositeSet.from_json(assignment) for assignment in data["assignments"]]
        return cls({variable: assignment for variable, assignment in zip(variables, assignments)})

    def update_variables(self, new_variables: Dict[Variable, Variable]) -> Self:
        """
        Construct a new simple event where the own variables are replaced with the new variables.
        If the new variables are missing mappings, the old variables are kept for the missing updates.

        :param new_variables: A dictionary mapping current variables to new variables
        :return: A new SimpleEvent with the updated variables
        """
        return SimpleEvent({
            new_variables.get(variable, variable): value for variable, value in self.items()
        })

    def plot(self) -> Union[List[go.Scatter], List[go.Mesh3d]]:
        """
        Plot the event.
        """
        assert all(isinstance(variable, Continuous) for variable in
                   self.keys()), "Plotting is only supported for events that consist of only continuous variables."
        if len(self.keys()) == 1:
            return self.plot_1d()
        if len(self.keys()) == 2:
            return self.plot_2d()
        elif len(self.keys()) == 3:
            return self.plot_3d()
        else:
            raise NotImplementedError("Plotting is only supported for two and three dimensional events")

    def plot_1d(self) -> List[go.Scatter]:
        """
        Plot the event in 1D.
        """
        xs = []
        ys = []

        interval: Interval = list(self.values())[0]
        for simple_interval in interval.simple_sets:
            simple_interval: SimpleInterval
            xs.extend([simple_interval.lower, simple_interval.upper, None])
            ys.extend([0, 0, None])

        return [go.Scatter(x=xs, y=ys, mode="lines", name="Event", fill="toself")]

    def plot_2d(self) -> List[go.Scatter]:
        """
        Plot the event in 2D.
        """

        # form cartesian product of all intervals
        intervals = [value.simple_sets for value in self.values()]
        interval_combinations = list(itertools.product(*intervals))

        xs = []
        ys = []

        # for every atomic interval
        for interval_combination in interval_combinations:
            # plot a rectangle
            points = np.asarray(list(itertools.product(*[[axis.lower, axis.upper] for axis in interval_combination])))
            y_points = points[:, 1]
            y_points[len(y_points) // 2:] = y_points[len(y_points) // 2:][::-1]
            xs.extend(points[:, 0].tolist() + [points[0, 0], None])
            ys.extend(y_points.tolist() + [y_points[0], None])

        return [go.Scatter(x=xs, y=ys, mode="lines", name="Event", fill="toself")]

    def plot_3d(self) -> List[go.Mesh3d]:
        """
        Plot the event in 3D.
        """

        # form cartesian product of all intervals
        intervals = [value.simple_sets for _, value in sorted(self.items())]
        simple_events = list(itertools.product(*intervals))
        traces = []

        # shortcut for the dimensions
        x, y, z = 0, 1, 2

        # for every atomic interval
        for simple_event in simple_events:
            # Create a 3D mesh trace for the rectangle
            traces.append(go.Mesh3d(  # 8 vertices of a cube
                x=[simple_event[x].lower, simple_event[x].lower, simple_event[x].upper, simple_event[x].upper,
                   simple_event[x].lower, simple_event[x].lower, simple_event[x].upper, simple_event[x].upper],
                y=[simple_event[y].lower, simple_event[y].upper, simple_event[y].upper, simple_event[y].lower,
                   simple_event[y].lower, simple_event[y].upper, simple_event[y].upper, simple_event[y].lower],
                z=[simple_event[z].lower, simple_event[z].lower, simple_event[z].lower, simple_event[z].lower,
                   simple_event[z].upper, simple_event[z].upper, simple_event[z].upper, simple_event[z].upper],
                # i, j and k give the vertices of triangles
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], flatshading=True))
        return traces

    def plotly_layout(self) -> Dict:
        """
        Create a layout for the plotly plot.
        """
        if len(self.variables) == 1:
            result = {"xaxis_title": self.variables[0].name}
        elif len(self.variables) == 2:
            result = {"xaxis_title": self.variables[0].name, "yaxis_title": self.variables[1].name}
        elif len(self.variables) == 3:
            result = dict(scene=dict(xaxis_title=self.variables[0].name, yaxis_title=self.variables[1].name,
                                     zaxis_title=self.variables[2].name))
        else:
            raise NotImplementedError("Plotting is only supported for two and three dimensional events")

        return result

    def fill_missing_variables(self, variables: Iterable[Variable]):
        """
        Fill this with the variables that are not in self but in `variables` in-place.
        The variables are mapped to their domain.

        :param variables: The variables to fill the event with
        """
        for variable in variables:
            if variable not in self:
                self[variable] = variable.domain

    def fill_missing_variables_pure(self, variables: Iterable[Variable]):
        """
        Fill this with the variables that are not in self but in `variables`.
        The variables are mapped to their domain.
        """
        return SimpleEvent({variable: self.get(variable, variable.domain) for variable in variables})

    def __deepcopy__(self):
        return self.__class__({variable: assignment.__deepcopy__() for variable, assignment in self.items()})


class Event(AbstractCompositeSet):
    """
    An event is a disjoint set of simple events.

    Every simple event added to this event that is missing variables that any other event in this event has, will be
    extended with the missing variable. The missing variables are mapped to their domain.

    """

    _cpp_object: rl.Event
    simple_set_example: SimpleEvent

    def __init__(self, *simple_sets):
        """
        Create a new event.
        :param simple_sets: The simple events that make up the event.
        """
        self._cpp_object = rl.Event({simple_set._cpp_object for simple_set in simple_sets})
        self.simple_set_example = SimpleEvent() if not simple_sets else simple_sets[0]
        self.fill_missing_variables()
        self.update_simple_set_example()

    def _from_cpp(self, cpp_object):
        return Event(*[self.simple_set_example._from_cpp(cpp_simple_set) for cpp_simple_set in cpp_object.simple_sets])

    @property
    def simple_sets(self) -> Tuple[SimpleEvent, ...]:
        return super().simple_sets

    @property
    def variables(self) -> SortedSet:
        return SortedSet([variable for simple_set in self.simple_sets for variable in simple_set.variables])

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
        self.simple_set_example = self.simple_sets[0] if self.simple_sets else SimpleEvent()

    def fill_missing_variables(self, variables: Optional[Iterable[Variable]] = None):
        """
        Fill all simple sets with the missing variables in-place.

        :param variables: The variables to fill the event with. If None, all variables are used.
        """
        if variables is None:
            variables = set()

        all_variables = self.variables | set(variables)
        self.simple_set_example.fill_missing_variables(all_variables)
        self.simple_set_example._cpp_object.fill_missing_variables({variable._cpp_object for variable in all_variables})
        self._cpp_object.fill_missing_variables({variable._cpp_object for variable in all_variables})

    def fill_missing_variables_pure(self, variables: Optional[Iterable[Variable]] = None):
        """
        Fill all simple sets with the missing variables.

        :param variables: The variables to fill the event with.
        """

        if variables is None:
            variables = set()

        all_variables = self.variables | set(variables)

        return Event(*[simple_set.fill_missing_variables_pure(all_variables) for simple_set in self.simple_sets])

    def marginal(self, variables: VariableSet) -> Event:
        """
        Create the marginal event, that only contains the variables given.

        :param variables: The variables to contain in the marginal event
        :return: The marginal event
        """
        return self._from_cpp(self._cpp_object.marginal({self.get_variable(v.name)._cpp_object for v in variables}))

    def bounding_box(self) -> SimpleEvent:
        """
        Compute the bounding box of the event.
        The bounding box is the smallest simple event that contains this event. It is computed by taking the union
        of all simple events variable wise.

        :return: The bounding box as a simple event
        """
        result = SimpleEvent()
        for variable in self.variables:
            for simple_set in self.simple_sets:
                if variable not in result:
                    result[variable] = simple_set[variable].__deepcopy__()
                else:
                    result[variable] = result[variable].__deepcopy__().union_with(simple_set[variable].__deepcopy__())
        return result

    def update_variables(self, new_variables: Dict[Variable, Variable]) -> Event:
        """
        see :func:`~random_events.product_algebra.SimpleEvent.update_variables`
        """
        return Event(*[simple_event.update_variables(new_variables) for simple_event in self.simple_sets])

    def plot(self, color="#636EFA") -> Union[List[go.Scatter], List[go.Mesh3d]]:
        """
        Plot the complex event.

        :param color: The color to use for this event
        """
        traces = []
        show_legend = True
        for index, event in enumerate(self.simple_sets):
            event_traces = event.plot()
            for event_trace in event_traces:
                if len(event.keys()) == 2:
                    event_trace.update(name="Event", legendgroup=id(self), showlegend=show_legend,
                                       line=dict(color=color))
                if len(event.keys()) == 3:
                    event_trace.update(name="Event", legendgroup=id(self), showlegend=show_legend, color=color)
                show_legend = False
                traces.append(event_trace)
        return traces

    def plotly_layout(self) -> Dict:
        """
        Create a layout for the plotly plot.
        """
        return self.simple_sets[0].plotly_layout()

    def add_simple_set(self, simple_set: AbstractSimpleSet):
        """
        Add a simple set to this event.

        :param simple_set: The simple set to add
        """
        super().add_simple_set(simple_set)
        self.fill_missing_variables()

    def to_json(self) -> Dict[str, Any]:
        variables = [variable.to_json() for variable in self.variables]
        simple_sets = [simple_set.to_json_assignments_only() for simple_set in self.simple_sets]
        return {**SubclassJSONSerializer.to_json(self), "variables": variables, "simple_sets": simple_sets}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variables = [Variable.from_json(variable) for variable in data["variables"]]
        simple_sets = [SimpleEvent.from_json_given_variables(simple_set, variables) for simple_set in
                       data["simple_sets"]]
        return cls(*simple_sets)
