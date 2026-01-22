import unittest

import numpy as np

from random_events.polytope import Polytope
import plotly.graph_objects as go


def rotation_matrix(theta: float):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class PolytopeTestCase(unittest.TestCase):
    np.random.seed(69)
    points = np.random.uniform(-1, 1, (100, 2))
    points = points @ rotation_matrix(0.3)

    def test_from_2d_points(self):
        polytope = Polytope.from_2d_points(self.points)
        self.assertTrue(polytope.contains(self.points.T).all())

    def test_maximum_inner_box(self):
        polytope = Polytope.from_2d_points(self.points)
        box = polytope.maximum_inner_box()
        # event = box.to_simple_event().as_composite_set()
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=self.points[:, 0], y=self.points[:, 1], mode='markers', name='points'))
        # fig.add_traces(event.plot())
        # fig.show()
        self.assertTrue(box <= polytope)

    def test_inner_box_approximation(self):
        polytope = Polytope.from_2d_points(self.points)
        result = polytope.inner_box_approximation(0.1)
        self.assertTrue(result.is_disjoint())
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=self.points[:, 0], y=self.points[:, 1], mode='markers', name='points'))
        # fig.add_traces(result.plot())
        # fig.update_layout(result.plotly_layout())
        # fig.show()

    def test_outer_box_approximation(self):
        polytope = Polytope.from_2d_points(self.points)
        result = polytope.outer_box_approximation(0.1)
        self.assertTrue(result.is_disjoint())
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=self.points[:, 0], y=self.points[:, 1], mode='markers', name='points'))
        # fig.add_traces(result.plot())
        # fig.update_layout(result.plotly_layout())
        # fig.show()


if __name__ == "__main__":
    unittest.main()
