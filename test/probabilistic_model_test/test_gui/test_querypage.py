from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QLabel
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.set import Set

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.gui.query_widget import QueryWidget
from probabilistic_model.gui.variable_constraint_widget import VariableConstraintWidget

# %% variable constraint widget


def test_selecting_a_continuous_variable_constrains_it_with_the_slider(
    model, continuous_variable, application
):
    widget = VariableConstraintWidget(model.variables)
    emissions = []
    widget.changed.connect(lambda *arguments: emissions.append(arguments))

    widget.variable_combo.setCurrentIndex(widget.variable_combo.findText("v1"))
    QApplication.processEvents()
    assert widget.variable_combo.currentData() == continuous_variable
    assert len(emissions) > 0

    emissions.clear()
    widget.interval_widgets[0].slider.sliderMoved.emit((0.1, 0.9))
    assert len(emissions) > 0

    variable, constraint = widget.get_constraint()
    assert variable == continuous_variable
    assert constraint.simple_sets[0].lower == pytest.approx(0.1, abs=1e-3)
    assert constraint.simple_sets[0].upper == pytest.approx(0.9, abs=1e-3)


def test_selecting_a_symbolic_variable_constrains_it_to_a_set(model, symbolic_variable):
    widget = VariableConstraintWidget(model.variables)
    widget.variable_combo.setCurrentIndex(widget.variable_combo.findText("v2"))
    assert widget.variable_combo.currentData() == symbolic_variable

    variable, constraint = widget.get_constraint()
    assert variable == symbolic_variable
    assert isinstance(constraint, Set)


def test_constraint_widget_labels_the_range_of_the_prior(model, continuous_variable):
    priors = VariableMap(
        {
            continuous_variable: UniformDistribution(
                variable=continuous_variable, interval=closed(0.1, 0.9).simple_sets[0]
            )
        }
    )
    widget = VariableConstraintWidget(model.variables, priors)
    widget.variable_combo.setCurrentIndex(widget.variable_combo.findText("v1"))

    assert "Range:" in widget.value_label.text()
    assert "0.10" in widget.value_label.text()
    assert "0.90" in widget.value_label.text()
    mark_texts = [
        label.text()
        for label in widget.findChildren(QLabel)
        if label is not widget.value_label
    ]
    assert "0.10" in mark_texts
    assert "0.90" in mark_texts


# %% query widget


def test_query_widget_starts_with_one_query_and_one_evidence_row(controller):
    widget = QueryWidget(controller)
    assert len(widget.query_widgets) == 1
    assert len(widget.evidence_widgets) == 1


def test_query_widget_adds_and_removes_rows(controller):
    widget = QueryWidget(controller)
    initial_count = len(widget.query_widgets)

    widget.add_variable_row(
        widget.query_widgets, widget.query_widgets[0].parentWidget()
    )
    assert len(widget.query_widgets) == initial_count + 1

    variable_widget = widget.query_widgets[-1]
    widget.remove_variable_row(
        variable_widget.parentWidget(), variable_widget, widget.query_widgets
    )
    assert len(widget.query_widgets) == initial_count


# %% controller


def test_probability_of_a_query_given_evidence(
    controller, continuous_variable, symbolic_variable
):
    query = Event.from_simple_sets(
        SimpleEvent.from_data({continuous_variable: closed(0, 0.5)})
    )
    evidence = Event.from_simple_sets(
        SimpleEvent.from_data({symbolic_variable: Set.from_iterable(["a"])})
    )
    query.fill_missing_variables([symbolic_variable])
    evidence.fill_missing_variables([continuous_variable])
    assert controller.calculate_probability(query, evidence) == pytest.approx(0.5)
