import os.path
import unittest

from krrood.ripple_down_rules import logger
from krrood.ripple_down_rules import app

try:
    from krrood.ripple_down_rules.user_interface.gui import RDRCaseViewer, style
except ImportError as e:
    logger.warning(f"Skipping GUI tests due to missing PyQt6: {e}")
    RDRCaseViewer = None
    style = None
from typing_extensions import List

from ..datasets import load_zoo_dataset, Species
from krrood.ripple_down_rules.datastructures.case import Case
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from ..test_object_diagram import Person, Address


@unittest.skipIf(
    app is None,
    "GUI tests need PyQt6,"
    "and they need visual inspection by a user and cannot be run automatically.",
)
class GUITestCase(unittest.TestCase):
    """Test case for the GUI components of the ripple down rules package."""

    viewer: RDRCaseViewer
    cq: CaseQuery
    cases: List[Case]
    person: Person

    @classmethod
    def setUpClass(cls):
        cls.cases, cls.targets = load_zoo_dataset(
            cache_file=f"{os.path.dirname(__file__)}/../test_results/zoo"
        )
        cls.cq = CaseQuery(
            cls.cases[0], "species", (Species,), True, _target=cls.targets[0]
        )
        cls.viewer = RDRCaseViewer()
        cls.person = Person("Ahmed", Address("Cairo"))

    def test_change_title_text(self):
        self.viewer.show()
        app.exec()
        self.viewer.title_label.setText(style("Changed Title", "o", 28, "bold"))
        self.viewer.show()
        app.exec()

    def test_update_image(self):
        self.viewer.obj_diagram_viewer.update_image(
            f"{os.path.dirname(__file__)}/../test_helpers/object_diagram_case_query.png"
        )
        self.viewer.show()
        app.exec()
        self.viewer.obj_diagram_viewer.update_image(
            f"{os.path.dirname(__file__)}/../test_helpers/object_diagram_person.png"
        )
        self.viewer.show()
        app.exec()

    def test_update_for_obj(self):
        self.viewer.update_for_object(self.cq, "CaseQuery")
        self.viewer.show()
        app.exec()
        self.viewer.update_for_object(self.person, "Person")
        self.viewer.show()
        app.exec()
