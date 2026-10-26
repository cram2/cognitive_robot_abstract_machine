from __future__ import annotations

import os

from probabilistic_model.gui.controller import ModelController
from probabilistic_model.gui.home_widget import HomeWidget

# %% controller


def test_setting_a_model_records_its_variables_and_their_priors(model):
    controller = ModelController()
    controller.set_model(model)
    assert controller.model is model
    assert controller.variable_map == {
        variable.name: variable for variable in model.variables
    }
    for variable in model.variables:
        assert list(controller.priors[variable].variables) == [variable]


# %% home widget


def test_home_widget_lists_every_variable_of_the_model(controller, model):
    widget = HomeWidget(controller)
    assert widget.variable_list_layout.count() == len(model.variables)


def test_home_widget_lists_the_variables_of_a_model_set_later(model):
    controller = ModelController()
    widget = HomeWidget(controller)
    assert widget.variable_list_layout.count() == 0

    controller.set_model(model)
    widget.refresh_variable_list()
    assert widget.variable_list_layout.count() == len(model.variables)


def test_home_widget_logo_exists():
    logo_path = HomeWidget(ModelController()).get_logo_path()
    assert os.path.exists(logo_path)
    assert logo_path.endswith("Logo.svg")
