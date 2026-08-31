import unittest

import numpy as np
import plotly.graph_objects as go

from random_events.interval import *
from random_events.plotting import (
    EventContainsSymbolicVariableError,
    EventPlotter,
    SimpleEventPlotter,
)
from random_events.product_algebra import SimpleEvent, Event
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic


class PlotTestCase(unittest.TestCase):
    x = Continuous(name="x")
    y = Continuous(name="y")
    z = Continuous(name="z")

    def test_plot_rejects_a_symbolic_variable(self):
        """
        A symbolic variable has no numeric axis to plot along, so plotting an event that
        constrains one must raise rather than attempt to draw it.
        """
        a = Symbolic(name="a", domain=Set.from_iterable({"a", "b"}))
        event = SimpleEvent.from_data({a: Set.from_iterable({"a"})})
        with self.assertRaises(EventContainsSymbolicVariableError):
            SimpleEventPlotter(event).plot()

    def test_plot_2d(self):
        event_1 = SimpleEvent.from_data(
            {
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )
        event_2 = SimpleEvent.from_data(
            {
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(1, 2)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(1, 2)),
            }
        )
        plotter = EventPlotter(Event.from_simple_sets(event_1, event_2))
        fig = go.Figure(plotter.plot(), plotter.plotly_layout())
        self.assertIsNotNone(fig)  # fig.show()

    def test_plot_3d(self):
        event_1 = SimpleEvent.from_data(
            {
                self.x: SimpleInterval.from_data(0, 1),
                self.y: SimpleInterval.from_data(0, 1),
                self.z: SimpleInterval.from_data(0, 1),
            }
        )
        event_2 = SimpleEvent.from_data(
            {
                self.x: SimpleInterval.from_data(1, 2),
                self.y: SimpleInterval.from_data(1, 2),
                self.z: SimpleInterval.from_data(1, 2),
            }
        )
        plotter = EventPlotter(Event.from_simple_sets(event_1, event_2))
        fig = go.Figure(plotter.plot(), plotter.plotly_layout())
        self.assertIsNotNone(fig)  # fig.show()

    def test_corners_3d_holds_the_eight_corners_of_a_box(self):
        """
        A three-dimensional simple event becomes the eight corners plotly draws faces
        and edges between.
        """
        event = SimpleEvent.from_data(
            {
                self.x: SimpleInterval.from_data(0, 1),
                self.y: SimpleInterval.from_data(0, 2),
                self.z: SimpleInterval.from_data(0, 3),
            }
        )
        corners = SimpleEventPlotter(event).corners_3d()

        self.assertEqual(corners.shape, (1, 8, 3))
        self.assertEqual(corners[0].min(axis=0).tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(corners[0].max(axis=0).tolist(), [1.0, 2.0, 3.0])

    def test_every_box_of_a_batched_mesh_indexes_its_own_corners(self):
        """
        Boxes share one trace, so each box's faces have to be offset onto its own eight
        corners; without the offset every box after the first would be drawn from the
        first one's geometry.
        """
        event_1 = SimpleEvent.from_data(
            {
                self.x: SimpleInterval.from_data(0, 1),
                self.y: SimpleInterval.from_data(0, 1),
                self.z: SimpleInterval.from_data(0, 1),
            }
        )
        event_2 = SimpleEvent.from_data(
            {
                self.x: SimpleInterval.from_data(5, 6),
                self.y: SimpleInterval.from_data(0, 1),
                self.z: SimpleInterval.from_data(0, 1),
            }
        )
        plotter = EventPlotter(Event.from_simple_sets(event_1, event_2))

        mesh = plotter.plot_3d_mesh("#000000", 1.0, "boxes")
        faces = np.stack([mesh.i, mesh.j, mesh.k], axis=-1)

        face_count = len(EventPlotter.BOX_FACES)
        self.assertEqual(len(faces), 2 * face_count)
        self.assertLess(faces[:face_count].max(), 8)
        self.assertGreaterEqual(faces[face_count:].min(), 8)

    def test_a_batched_wireframe_draws_every_edge_as_its_own_segment(self):
        """
        The edges are one trace, so consecutive edges have to be separated or the line
        would run from the end of one edge to the start of the next.
        """
        plotter = EventPlotter(
            Event.from_simple_sets(
                SimpleEvent.from_data(
                    {
                        self.x: SimpleInterval.from_data(0, 1),
                        self.y: SimpleInterval.from_data(0, 1),
                        self.z: SimpleInterval.from_data(0, 1),
                    }
                )
            )
        )

        wireframe = plotter.plot_3d_wireframe("#000000", 1.0, "boxes")
        points = np.stack([wireframe.x, wireframe.y, wireframe.z], axis=-1)

        edge_count = len(EventPlotter.BOX_EDGES)
        self.assertEqual(len(points), 3 * edge_count)
        self.assertTrue(np.isnan(points[2::3]).all())
        self.assertFalse(np.isnan(points[0::3]).any())


class IntegerVariablePlotTestCase(unittest.TestCase):
    count = Integer(name="count")

    def test_plot_1d_with_integer_variable(self):
        plotter = SimpleEventPlotter(
            SimpleEvent.from_data({self.count: SimpleInterval.from_data(0, 5)})
        )
        traces = plotter.plot()
        self.assertIsNotNone(traces)
        fig = go.Figure(traces, plotter.plotly_layout())
        self.assertIsNotNone(fig)
