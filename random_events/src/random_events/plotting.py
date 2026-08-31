"""
Plotly rendering for :class:`~random_events.product_algebra.SimpleEvent` and
:class:`~random_events.product_algebra.Event`.

Kept out of :mod:`random_events.product_algebra` so that the set algebra there does not
require plotly to be installed to be used, and kept as plotters that wrap an event
rather than as methods on the events themselves, so that the events stay set algebra and
nothing else.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from typing_extensions import ClassVar, Dict, List, Union

from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Symbolic


class EventContainsSymbolicVariableError(Exception):
    """
    Raised when plotting an event that constrains a symbolic variable -- only continuous
    variables can be plotted.
    """


@dataclass(frozen=True)
class SimpleEventPlotter:
    """
    Plots a single :class:`~random_events.product_algebra.SimpleEvent`.
    """

    event: SimpleEvent
    """
    The simple event to plot.
    """

    def plot(self) -> Union[List[go.Scatter], List[go.Mesh3d]]:
        """
        :return: The event's traces, dispatched on its dimensionality.
        :raises EventContainsSymbolicVariableError: If the event constrains a symbolic
            variable.
        :raises NotImplementedError: If the event constrains a variable other than one,
            two or three continuous variables.
        """
        if any(isinstance(variable, Symbolic) for variable in self.event.keys()):
            raise EventContainsSymbolicVariableError(
                "Plotting is only supported for events that consist of only "
                "continuous variables."
            )
        dimensionality = len(self.event.keys())
        if dimensionality == 1:
            return self.plot_1d()
        if dimensionality == 2:
            return self.plot_2d()
        if dimensionality == 3:
            return self.plot_3d()
        raise NotImplementedError(
            "Plotting is only supported for one, two and three dimensional events."
        )

    def plot_1d(self) -> List[go.Scatter]:
        """
        :return: The event's trace, drawn along a single axis.
        """
        xs = []
        ys = []

        interval = list(self.event.values())[0]
        for simple_interval in interval.simple_sets:
            xs.extend([simple_interval.lower, simple_interval.upper, None])
            ys.extend([0, 0, None])

        return [go.Scatter(x=xs, y=ys, mode="lines", name="Event", fill="toself")]

    def plot_2d(self) -> List[go.Scatter]:
        """
        :return: The event's trace, drawn as one filled outline per rectangle it
            decomposes into.
        """
        intervals = [value.simple_sets for value in self.event.values()]
        interval_combinations = list(itertools.product(*intervals))

        xs = []
        ys = []
        for interval_combination in interval_combinations:
            points = np.asarray(
                list(
                    itertools.product(
                        *[[axis.lower, axis.upper] for axis in interval_combination]
                    )
                )
            )
            y_points = points[:, 1]
            y_points[len(y_points) // 2 :] = y_points[len(y_points) // 2 :][::-1]
            xs.extend(points[:, 0].tolist() + [points[0, 0], None])
            ys.extend(y_points.tolist() + [y_points[0], None])

        return [go.Scatter(x=xs, y=ys, mode="lines", name="Event", fill="toself")]

    def plot_3d(self) -> List[go.Mesh3d]:
        """
        :return: One mesh trace per axis-aligned box the event decomposes into.
        """
        intervals = [value.simple_sets for _, value in sorted(self.event.items())]
        simple_events = list(itertools.product(*intervals))
        traces = []
        x, y, z = 0, 1, 2

        for simple_event in simple_events:
            traces.append(
                go.Mesh3d(  # 8 vertices of a cube
                    x=[
                        simple_event[x].lower,
                        simple_event[x].lower,
                        simple_event[x].upper,
                        simple_event[x].upper,
                        simple_event[x].lower,
                        simple_event[x].lower,
                        simple_event[x].upper,
                        simple_event[x].upper,
                    ],
                    y=[
                        simple_event[y].lower,
                        simple_event[y].upper,
                        simple_event[y].upper,
                        simple_event[y].lower,
                        simple_event[y].lower,
                        simple_event[y].upper,
                        simple_event[y].upper,
                        simple_event[y].lower,
                    ],
                    z=[
                        simple_event[z].lower,
                        simple_event[z].lower,
                        simple_event[z].lower,
                        simple_event[z].lower,
                        simple_event[z].upper,
                        simple_event[z].upper,
                        simple_event[z].upper,
                        simple_event[z].upper,
                    ],
                    # i, j and k give the vertices of triangles
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    flatshading=True,
                )
            )
        return traces

    def plotly_layout(self) -> Dict:
        """
        :return: The layout of the event's plot.
        :raises NotImplementedError: If the event constrains a variable other than one,
            two or three variables.
        """
        variables = self.event.variables
        if len(variables) == 1:
            return {"xaxis_title": variables[0].name}
        if len(variables) == 2:
            return {
                "xaxis_title": variables[0].name,
                "yaxis_title": variables[1].name,
            }
        if len(variables) == 3:
            return dict(
                scene=dict(
                    xaxis_title=variables[0].name,
                    yaxis_title=variables[1].name,
                    zaxis_title=variables[2].name,
                )
            )
        raise NotImplementedError(
            "Plotting is only supported for one, two and three dimensional events."
        )

    def corners_3d(self) -> np.ndarray:
        """
        Express the event as the eight corners of every axis-aligned box it decomposes
        into. A simple event decomposes into more than one box only when one of its
        three variables is assigned a disjoint (non-simple) interval.

        :return: The boxes' corners, shaped ``(box count, 8, 3)``, sorted by variable.
        :raises NotImplementedError: If the event does not constrain exactly three
            variables.
        """
        if len(self.event.keys()) != 3:
            raise NotImplementedError(
                "Corner extraction is only supported for three-dimensional events."
            )
        x, y, z = 0, 1, 2
        boxes = list(
            itertools.product(
                *(value.simple_sets for _, value in sorted(self.event.items()))
            )
        )
        return np.array(
            [
                [
                    [box[x].lower, box[y].lower, box[z].lower],
                    [box[x].upper, box[y].lower, box[z].lower],
                    [box[x].upper, box[y].upper, box[z].lower],
                    [box[x].lower, box[y].upper, box[z].lower],
                    [box[x].lower, box[y].lower, box[z].upper],
                    [box[x].upper, box[y].lower, box[z].upper],
                    [box[x].upper, box[y].upper, box[z].upper],
                    [box[x].lower, box[y].upper, box[z].upper],
                ]
                for box in boxes
            ]
        ).reshape(-1, 8, 3)


@dataclass(frozen=True)
class EventPlotter:
    """
    Plots a composite :class:`~random_events.product_algebra.Event`, delegating to a
    :class:`SimpleEventPlotter` per simple set it is the union of.
    """

    event: Event
    """
    The event to plot.
    """

    BOX_FACES: ClassVar[tuple[tuple[int, int, int], ...]] = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    """
    The two triangles of each of a box's six faces, as indices into :meth:`corners_3d`'s
    eight vertices per box.
    """

    BOX_EDGES: ClassVar[tuple[tuple[int, int], ...]] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    """
    The twelve edges of a box, as index pairs into :meth:`corners_3d`'s eight vertices
    per box.
    """

    def plot(self, color: str = "#636EFA") -> Union[List[go.Scatter], List[go.Mesh3d]]:
        """
        :param color: The color to use for the event.
        :return: The traces of every simple set the event is the union of.
        """
        traces = []
        show_legend = True
        for simple_event in self.event.simple_sets:
            for trace in SimpleEventPlotter(simple_event).plot():
                if len(simple_event.keys()) == 2:
                    trace.update(
                        name="Event",
                        legendgroup=id(self.event),
                        showlegend=show_legend,
                        line=dict(color=color),
                    )
                if len(simple_event.keys()) == 3:
                    trace.update(
                        name="Event",
                        legendgroup=id(self.event),
                        showlegend=show_legend,
                        color=color,
                    )
                show_legend = False
                traces.append(trace)
        return traces

    def plotly_layout(self) -> Dict:
        """
        :return: The layout of the event's plot.
        """
        return SimpleEventPlotter(self.event.simple_sets[0]).plotly_layout()

    def corners_3d(self) -> np.ndarray:
        """
        :return: Every axis-aligned box the event decomposes into, as arrays of eight
            x-y-z corners each.
        """
        if not self.event.simple_sets:
            return np.empty((0, 8, 3))
        return np.concatenate(
            [
                SimpleEventPlotter(simple_set).corners_3d()
                for simple_set in self.event.simple_sets
            ],
            axis=0,
        )

    def plot_3d_mesh(
        self, color: str, opacity: float = 1.0, name: str = "Event"
    ) -> go.Mesh3d:
        """
        Draw every box the event decomposes into as one solid mesh, so that any number
        of boxes costs one trace rather than one trace each.

        :param color: The color the boxes are filled with.
        :param opacity: How opaque the fill is.
        :param name: The name the trace carries into the legend.
        :return: The batched mesh trace.
        """
        corners = self.corners_3d()
        box_count = len(corners)
        vertices = corners.reshape(-1, 3)
        offsets = np.arange(box_count).repeat(len(self.BOX_FACES)) * 8
        faces = np.tile(np.array(self.BOX_FACES), (box_count, 1)) + offsets[:, None]
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            flatshading=True,
            hoverinfo="skip",
            name=name,
            legendgroup=name,
        )

    def plot_3d_wireframe(
        self, color: str, width: float = 1.0, name: str = "Event"
    ) -> go.Scatter3d:
        """
        Draw the edges of every box the event decomposes into as one line trace, which
        is what makes boxes behind other boxes readable at all.

        :param color: The color of the edges.
        :param width: The width of the edges in pixels.
        :param name: The name the trace carries into the legend.
        :return: The batched wireframe trace.
        """
        corners = self.corners_3d()
        box_count = len(corners)
        starts = corners[:, [edge[0] for edge in self.BOX_EDGES], :]
        ends = corners[:, [edge[1] for edge in self.BOX_EDGES], :]
        segments = np.full((box_count * len(self.BOX_EDGES), 3, 3), np.nan)
        segments[:, 0, :] = starts.reshape(-1, 3)
        segments[:, 1, :] = ends.reshape(-1, 3)
        points = segments.reshape(-1, 3)
        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            name=name,
            legendgroup=name,
        )
