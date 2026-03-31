import unittest
from unittest.mock import MagicMock
import sys

from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from probabilistic_model.gui.controller import ModelController
from probabilistic_model.gui.posterior_widget import PosteriorWidget
from random_events.variable import Continuous, Symbolic
from random_events.set import Set
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent


class TestPosteriorGUI(unittest.TestCase):

    def setUp(self):
        self.controller = ModelController()

        # Setup a simple model
        self.v1 = Continuous(name="v1")
        self.v2 = Symbolic(name="v2", domain=Set.from_iterable(["a", "b"]))

        self.model = MagicMock()
        self.model.variables = [self.v1, self.v2]
        self.model.marginal.return_value = self.model
        self.model.plot.return_value = []
        self.model.plotly_layout.return_value = {}
        self.model.truncated.return_value = (self.model, 1.0)

        self.controller.set_model(self.model)

    def test_posterior_widget_instantiation(self):
        widget = PosteriorWidget(self.controller)
        self.assertIsNotNone(widget)
        # Check if variables are populated in the list
        self.assertEqual(widget.query_vars_list.count(), 2)
        # Check initial evidence row
        self.assertEqual(len(widget.evidence_widgets), 1)

    def test_posterior_widget_add_remove_evidence_row(self):
        widget = PosteriorWidget(self.controller)
        initial_count = len(widget.evidence_widgets)

        widget.add_evidence_row()
        self.assertEqual(len(widget.evidence_widgets), initial_count + 1)

        row_widget = widget.evidence_widgets[-1].parentWidget()
        var_widget = widget.evidence_widgets[-1]
        widget.remove_evidence_row(row_widget, var_widget)
        self.assertEqual(len(widget.evidence_widgets), initial_count)

    def test_posterior_on_calculate(self):
        widget = PosteriorWidget(self.controller)
        # Select v1 as query variable
        item = widget.query_vars_list.item(0)
        item.setSelected(True)

        # Mock build_event
        evidence_event = Event.from_simple_sets(SimpleEvent.from_data())
        widget.on_calculate()

        # Since evidence is empty, it should return the model itself or call truncated (but here we check evidence_event)
        # Actually it's Event(SimpleEvent()) which is NOT empty (it has one empty simple set).
        # Wait, evidence.simple_sets is [SimpleEvent()] which is not empty list.
        # But SimpleEvent() is an empty mapping.

        # Let's check what on_calculate does with evidence rows.
        # If no evidence rows are set, widget.build_event() returns an Event with one SimpleEvent()?

        self.assertEqual(len(widget.query_vars), 1)
        self.assertEqual(widget.query_vars[0], self.v1.name)

    def test_controller_calculate_posterior(self):
        evidence = Event.from_simple_sets(SimpleEvent.from_data({self.v1: closed(0, 0.5)}))
        self.controller.calculate_posterior(evidence)
        self.model.truncated.assert_called()


if __name__ == "__main__":
    unittest.main()
