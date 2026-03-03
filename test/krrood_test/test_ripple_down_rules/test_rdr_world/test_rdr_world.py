from __future__ import annotations

import os
from os.path import dirname

import pytest

from krrood.ripple_down_rules.utils import draw_tree, render_tree

try:
    from PyQt6.QtWidgets import QApplication
    from krrood.ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError as e:
    QApplication = None
    RDRCaseViewer = None

from ..datasets import *
from krrood.ripple_down_rules.helpers import is_matching
from krrood.ripple_down_rules import *


def test_drawer_cabinet_rdr(drawer_cabinet_rdr):
    pass


@pytest.mark.skip("Skipping test_drawer_cabinet_ai_rdr as it is not implemented yet.")
def test_drawer_cabinet_ai_rdr(drawer_cabinet_ai_rdr):
    pass


def test_save_and_load_drawer_cabinet_rdr(
    handles_and_containers_world, drawer_cabinet_rdr
):
    world = handles_and_containers_world
    filename = os.path.join(
        dirname(__file__), "..", "test_results", "world_drawer_cabinet_rdr"
    )
    model_name = drawer_cabinet_rdr.save(filename)
    loaded_rdr = GeneralRDR.load(filename, model_name=model_name)
    assert drawer_cabinet_rdr.classify(world) == loaded_rdr.classify(world)
    # assert world.bodies == loaded_rdr.start_rules[0].corner_case.bodies


def test_draw_evaluated_tree_for_drawer_cabinet_rdr(
    handles_and_containers_world, drawer_cabinet_rdr
):
    world = handles_and_containers_world
    filename = os.path.join(
        dirname(__file__), "..", "test_results", "world_drawer_cabinet_rdr"
    )
    loaded_rdr = GeneralRDR.load(filename, model_name="world_rdr")
    loaded_rdr.classify(world)
    loaded_rdr.render_evaluated_rule_tree("drawer_cabinet_rdr_evaluated_tree")
    assert True  # If no exception is raised, the test passes


def test_write_drawer_cabinet_rdr_to_python_file(
    drawer_cabinet_rdr, handles_and_containers_world
):
    rdrs_dir = os.path.join(dirname(__file__), "..", "test_generated_rdrs", "view_rdr")
    os.makedirs(rdrs_dir, exist_ok=True)
    drawer_cabinet_rdr._write_to_python(rdrs_dir)
    loaded_rdr_classifier = drawer_cabinet_rdr.get_rdr_classifier_from_python_file(
        rdrs_dir
    )
    found_views = loaded_rdr_classifier(handles_and_containers_world)
    assert len([v for v in found_views["views"] if isinstance(v, Drawer)]) == 1
    assert len([v for v in found_views["views"] if isinstance(v, Cabinet)]) == 1
    assert len(found_views["views"]) == 2


def test_drawer_rdr(correct_drawer_rdr):
    pass


def test_write_drawer_rdr_to_python_file(correct_drawer_rdr, drawer_case_queries):
    rdrs_dir = os.path.join(
        dirname(__file__), "..", "test_generated_rdrs", "drawer_rdr"
    )
    os.makedirs(rdrs_dir, exist_ok=True)
    correct_drawer_rdr._write_to_python(rdrs_dir)
    loaded_rdr_classifier = correct_drawer_rdr.get_rdr_classifier_from_python_file(
        rdrs_dir
    )
    for case_query in drawer_case_queries:
        assert is_matching(loaded_rdr_classifier, case_query)
