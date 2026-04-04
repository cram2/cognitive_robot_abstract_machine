import unittest
from unittest.mock import MagicMock
import sys
import os
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from probabilistic_model.gui.controller import ModelController
from probabilistic_model.gui.mode_widget import ModeWidget
from probabilistic_model.gui.variable_constraint_widget import VariableConstraintWidget
from random_events.variable import Continuous, Symbolic
from random_events.set import Set, SetElement
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent


class TestModeGUI(unittest.TestCase):

    def setUp(self):
        self.controller = ModelController()

        # Setup a simple model
        self.v1 = Continuous(name="v1", domain=closed(0, 1))
        self.v2 = Symbolic(name="v2", domain=Set.from_iterable(["a", "b"]))

        self.model = MagicMock()
        self.model.variables = [self.v1, self.v2]

        # Configure marginal to return real numbers for bounds
        mock_marginal_v1 = MagicMock()
        mock_marginal_v1.support.simple_sets = [SimpleEvent.from_data({self.v1: closed(0, 1)})]
        
        mock_marginal_v2 = MagicMock()
        mock_marginal_v2.support.simple_sets = [SimpleEvent.from_data({self.v2: Set.from_iterable(["a", "b"])})]

        def side_effect(vars):
            if vars == [self.v1]:
                return mock_marginal_v1
            return mock_marginal_v2
        
        self.model.marginal.side_effect = side_effect

        # Mock mode result: (Event, Likelihood)
        self.mode_event = Event.from_simple_sets(
            SimpleEvent.from_data({self.v1: closed(0.5, 0.5), self.v2: Set.from_iterable(["a"])})
        )
        self.likelihood = 0.8

        # Mock truncated for mode
        self.mock_truncated_model = MagicMock()
        self.mock_truncated_model.mode.return_value = (self.mode_event, self.likelihood)

        self.model.truncated.return_value = (self.mock_truncated_model, 1.0)
        self.model.mode.return_value = (self.mode_event, self.likelihood)

        self.controller.set_model(self.model)

    def test_mode_widget_instantiation(self):
        widget = ModeWidget(self.controller)
        self.assertIsNotNone(widget)
        self.assertEqual(len(widget.evidence_widgets), 1)

    def test_mode_widget_calculate(self):
        widget = ModeWidget(self.controller)

        # Trigger calculate
        widget.on_calculate()

        self.model.mode.assert_called()
        self.assertEqual(len(widget.modes), 1)
        self.assertEqual(widget.likelihood, 0.8)
        self.assertIn("Result 1/1", widget.mode_info_label.text())

    def test_mode_widget_unsatisfiable(self):
        widget = ModeWidget(self.controller)

        # Mock build_evidence_event to return non-empty event
        widget.build_evidence_event = MagicMock(
            return_value=Event.from_simple_sets(SimpleEvent.from_data({self.v1: closed(2, 3)}))
        )

        # Mock calculate_mode to return None
        self.controller.calculate_mode = MagicMock(return_value=None)

        widget.on_calculate()
        self.assertEqual(widget.mode_info_label.text(), "Unsatisfiable")

    def test_variable_constraint_widget_set_constraint(self):
        widget = VariableConstraintWidget(self.model.variables)

        # Set continuous constraint
        c1 = closed(0.2, 0.8)
        widget.set_constraint(self.v1, c1)

        var, constraint = widget.get_constraint()
        self.assertEqual(var, self.v1)
        # Check values with small delta due to *1000 in slider
        self.assertAlmostEqual(constraint.simple_sets[0].lower, 0.2, places=3)
        self.assertAlmostEqual(constraint.simple_sets[0].upper, 0.8, places=3)

        # Set symbolic constraint
        c2 = Set.from_iterable(["b"])
        widget.set_constraint(self.v2, c2)

        var, constraint = widget.get_constraint()
        self.assertEqual(var, self.v2)
        # constraint is a Set, simple_sets contains SetElement
        elements = [str(e.element) for e in constraint.simple_sets]
        self.assertIn("b", elements)
        self.assertNotIn("a", elements)


if __name__ == "__main__":
    unittest.main()
