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
from probabilistic_model.gui.query_widget import QueryWidget
from probabilistic_model.gui.variable_constraint_widget import VariableConstraintWidget
from random_events.variable import Continuous, Symbolic
from random_events.set import Set, SetElement
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent


class TestQueryGUI(unittest.TestCase):

    def setUp(self):
        self.controller = ModelController()
        self.v1 = Continuous("v1", closed(0, 1))
        self.v2 = Symbolic("v2", Set.from_iterable(["a", "b"]))

        self.model = MagicMock()
        self.model.variables = [self.v1, self.v2]
        self.model.probability.return_value = 0.5

        mock_marginal_v1 = MagicMock()
        mock_marginal_v1.support.simple_sets = [SimpleEvent({self.v1: closed(0, 1)})]

        mock_marginal_v2 = MagicMock()
        mock_marginal_v2.support.simple_sets = [
            SimpleEvent({self.v2: Set.from_iterable(["a", "b"])})
        ]

        self.model.marginal.side_effect = lambda vars: (
            mock_marginal_v1 if vars == [self.v1] else mock_marginal_v2
        )
        self.controller.set_model(self.model)

    def test_variable_constraint_widget_continuous(self):
        widget = VariableConstraintWidget(self.model.variables)
        changed_mock = MagicMock()
        widget.changed.connect(changed_mock)

        # Select v1
        index = widget.variable_combo.findText("v1")
        widget.variable_combo.setCurrentIndex(index)
        app.processEvents()

        self.assertEqual(widget.variable_combo.currentData(), self.v1)
        self.assertIsNotNone(widget.constraint_widget)
        self.assertTrue(changed_mock.called)
        changed_mock.reset_mock()

        # Check slider value change triggers changed signal
        slider = widget.interval_widgets[0].slider
        slider.sliderMoved.emit((0.1, 0.9))
        self.assertTrue(changed_mock.called)

        # Check constraint retrieval
        var, constraint = widget.get_constraint()
        self.assertEqual(var, self.v1)
        self.assertAlmostEqual(constraint.simple_sets[0].lower, 0.1, places=3)
        self.assertAlmostEqual(constraint.simple_sets[0].upper, 0.9, places=3)

    def test_variable_constraint_widget_symbolic(self):
        widget = VariableConstraintWidget(self.model.variables)
        # Select v2
        index = widget.variable_combo.findText("v2")
        widget.variable_combo.setCurrentIndex(index)

        self.assertEqual(widget.variable_combo.currentData(), self.v2)
        self.assertIsNotNone(widget.constraint_widget)

        # Check constraint retrieval
        var, constraint = widget.get_constraint()
        self.assertEqual(var, self.v2)
        self.assertTrue(isinstance(constraint, Set))

    def test_query_widget_instantiation(self):
        widget = QueryWidget(self.controller)
        self.assertIsNotNone(widget)
        self.assertEqual(len(widget.query_widgets), 1)
        self.assertEqual(len(widget.evidence_widgets), 1)

    def test_query_widget_add_remove_row(self):
        widget = QueryWidget(self.controller)
        initial_count = len(widget.query_widgets)

        layout_container = widget.query_widgets[0].parentWidget()
        widget.add_variable_row(widget.query_widgets, layout_container)
        self.assertEqual(len(widget.query_widgets), initial_count + 1)

        row_widget = widget.query_widgets[-1].parentWidget()
        var_widget = widget.query_widgets[-1]
        widget.remove_variable_row(row_widget, var_widget, widget.query_widgets)
        self.assertEqual(len(widget.query_widgets), initial_count)

    def test_controller_calculate_probability(self):
        query = Event(SimpleEvent({self.v1: closed(0, 0.5)}))
        evidence = Event(SimpleEvent({self.v2: Set.from_iterable(["a"])}))

        self.model.probability.return_value = 0.5
        # Mock intersection_with
        joint = query.intersection_with(evidence)

        prob = self.controller.calculate_probability(query, evidence)
        self.model.probability.assert_any_call(evidence)
        self.assertEqual(prob, 1.0)  # 0.5 / 0.5 in this mock case

    def test_variable_constraint_widget_labels(self):
        from random_events.product_algebra import VariableMap

        priors = VariableMap()
        dist = MagicMock()
        dist.support.simple_sets = [SimpleEvent({self.v1: closed(0.1, 0.9)})]
        priors[self.v1] = dist

        widget = VariableConstraintWidget(self.model.variables, priors)
        index = widget.variable_combo.findText("v1")
        widget.variable_combo.setCurrentIndex(index)

        self.assertIn("Range:", widget.value_label.text())
        self.assertIn("0.10", widget.value_label.text())
        self.assertIn("0.90", widget.value_label.text())

        from PySide6.QtWidgets import QLabel

        labels = widget.findChildren(QLabel)
        mark_labels = [l for l in labels if l != widget.value_label]
        self.assertGreaterEqual(len(mark_labels), 2)
        mark_texts = [l.text() for l in mark_labels]
        self.assertIn("0.10", mark_texts)
        self.assertIn("0.90", mark_texts)


if __name__ == "__main__":
    unittest.main()
