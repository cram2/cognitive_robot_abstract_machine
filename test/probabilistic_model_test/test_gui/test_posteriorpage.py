from __future__ import annotations

import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent

from probabilistic_model.gui.posterior_widget import PosteriorWidget

# %% posterior widget


def test_posterior_widget_lists_every_variable_and_one_evidence_row(controller, model):
    widget = PosteriorWidget(controller)
    assert widget.query_vars_list.count() == len(model.variables)
    assert len(widget.evidence_widgets) == 1


def test_posterior_widget_adds_and_removes_evidence_rows(controller):
    widget = PosteriorWidget(controller)
    initial_count = len(widget.evidence_widgets)

    widget.add_evidence_row()
    assert len(widget.evidence_widgets) == initial_count + 1

    variable_widget = widget.evidence_widgets[-1]
    widget.remove_evidence_row(variable_widget.parentWidget(), variable_widget)
    assert len(widget.evidence_widgets) == initial_count


def test_posterior_widget_queries_the_selected_variables(controller, model):
    widget = PosteriorWidget(controller)
    widget.query_vars_list.item(0).setSelected(True)
    widget.on_calculate()
    assert widget.query_vars == [model.variables[0].name]


# %% controller


def test_posterior_holds_all_mass_inside_the_evidence(
    controller, model, continuous_variable
):
    evidence = Event.from_simple_sets(
        SimpleEvent.from_data({continuous_variable: closed(0, 0.5)})
    )
    evidence.fill_missing_variables(model.variables)
    posterior = controller.calculate_posterior(evidence)
    assert posterior.probability(evidence) == pytest.approx(1.0)
