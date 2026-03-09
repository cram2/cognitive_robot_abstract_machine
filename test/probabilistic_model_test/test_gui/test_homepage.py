import unittest
from unittest.mock import MagicMock
import sys
import os

# Create a fake Qt Application for testing widgets
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from probabilistic_model.gui.controller import ModelController
from probabilistic_model.gui.home_widget import HomeWidget
from random_events.variable import Continuous, Symbolic
from random_events.set import Set
from random_events.interval import closed
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from random_events.product_algebra import VariableMap


class TestGUIComponents(unittest.TestCase):

    def setUp(self):
        self.controller = ModelController()

        # Setup a simple model
        self.v1 = Continuous("v1", closed(0, 1))
        self.v2 = Symbolic("v2", Set.from_iterable(["a", "b"]))

        self.model = MagicMock()
        self.model.variables = [self.v1, self.v2]

        # Mock marginals
        self.dist1 = UniformDistribution(self.v1, closed(0, 1))
        self.dist2 = SymbolicDistribution(
            self.v2,
            {self.v2.domain.simple_sets[0]: 0.5, self.v2.domain.simple_sets[1]: 0.5},
        )

        def mock_marginal(vars):
            if vars == [self.v1]:
                return self.dist1
            if vars == [self.v2]:
                return self.dist2
            return None

        self.model.marginal.side_effect = mock_marginal

    def test_controller_set_model(self):
        self.controller.set_model(self.model)
        self.assertEqual(self.controller.model, self.model)
        self.assertIn("v1", self.controller.variable_map)
        self.assertIn("v2", self.controller.variable_map)
        self.assertIsNotNone(self.controller.priors)
        self.assertEqual(self.controller.priors[self.v1], self.dist1)
        self.assertEqual(self.controller.priors[self.v2], self.dist2)

    def test_home_widget_instantiation(self):
        self.controller.set_model(self.model)
        widget = HomeWidget(self.controller)
        self.assertIsNotNone(widget)
        # Verify it has some content
        self.assertTrue(widget.variable_list_layout.count() >= 2)

    def test_home_widget_refresh(self):
        widget = HomeWidget(self.controller)
        self.assertEqual(widget.variable_list_layout.count(), 0)

        self.controller.set_model(self.model)
        widget.refresh_variable_list()
        self.assertEqual(widget.variable_list_layout.count(), 2)

    def test_home_widget_logo_path(self):
        widget = HomeWidget(self.controller)
        logo_path = widget.get_logo_path()
        self.assertTrue(os.path.exists(logo_path))
        self.assertTrue(logo_path.endswith("Logo.svg"))


if __name__ == "__main__":
    unittest.main()
