from __future__ import annotations

import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set

from probabilistic_model.gui.mode_widget import ModeWidget
from probabilistic_model.gui.variable_constraint_widget import VariableConstraintWidget

# %% mode widget


def test_mode_widget_starts_with_one_evidence_row(controller):
    assert len(ModeWidget(controller).evidence_widgets) == 1


def test_mode_widget_shows_the_mode_of_the_model(controller, model):
    widget = ModeWidget(controller)
    widget.on_calculate()
    likelihood = model.mode().likelihood
    assert len(widget.modes) == 1
    assert widget.likelihood == pytest.approx(likelihood)
    assert "Result 1/1" in widget.mode_info_label.text()


def test_mode_widget_reports_impossible_evidence(
    controller, continuous_variable, monkeypatch
):
    widget = ModeWidget(controller)
    impossible = Event.from_simple_sets(
        SimpleEvent.from_data({continuous_variable: closed(2, 3)})
    )
    monkeypatch.setattr(widget, "build_evidence_event", lambda: impossible)
    widget.on_calculate()
    assert widget.mode_info_label.text() == "Unsatisfiable"


# %% variable constraint widget


def test_constraint_widget_returns_the_constraint_it_was_set_to(
    model, continuous_variable, symbolic_variable
):
    widget = VariableConstraintWidget(model.variables)

    widget.set_constraint(continuous_variable, closed(0.2, 0.8))
    variable, constraint = widget.get_constraint()
    assert variable == continuous_variable
    # the slider resolves a thousandth of the range
    assert constraint.simple_sets[0].lower == pytest.approx(0.2, abs=1e-3)
    assert constraint.simple_sets[0].upper == pytest.approx(0.8, abs=1e-3)

    widget.set_constraint(symbolic_variable, Set.from_iterable(["b"]))
    variable, constraint = widget.get_constraint()
    assert variable == symbolic_variable
    assert [str(element.element) for element in constraint.simple_sets] == ["b"]
