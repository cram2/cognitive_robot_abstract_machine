import os
import sys
import unittest
from typing import List, Optional
from unittest import TestCase

from PyQt5.QtWidgets import QApplication

from krrood.ripple_down_rules import CaseQuery, SingleClassRDR
from krrood.ripple_down_rules.datastructures.case import Case
from krrood.ripple_down_rules.experts import Human
try:
    from krrood.ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError as e:
    RDRCaseViewer = None

from .datasets import load_zoo_dataset, Species, load_zoo_cases

TEST_RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "test_results")
CACHE_FILE: str = os.path.join(TEST_RESULTS_DIR, "zoo_dataset.pkl")
zoo_cases, _ = load_zoo_dataset(cache_file=CACHE_FILE)

@unittest.skipIf(len(zoo_cases) == 0, "Failed to load dataset")
class TestRDR(TestCase):
    all_cases: List[Case]
    targets: List[str]
    case_queries: List[CaseQuery]
    test_results_dir: str = TEST_RESULTS_DIR
    expert_answers_dir: str = os.path.join(
        os.path.dirname(__file__), "test_expert_answers"
    )
    generated_rdrs_dir: str = os.path.join(
        os.path.dirname(__file__), "test_generated_rdrs"
    )
    cache_file: str = CACHE_FILE
    app: Optional[QApplication] = None
    viewer: Optional[RDRCaseViewer] = None
    use_gui: bool = False

    @classmethod
    def setUpClass(cls):
        # fetch dataset
        cls.all_cases, cls.targets = load_zoo_dataset(cache_file=cls.cache_file)
        cls.case_queries = [
            CaseQuery(
                case,
                "species",
                Species,
                True,
                _target=target,
            )
            for case, target in zip(cls.all_cases, cls.targets)
        ]
        for test_dir in [
            cls.test_results_dir,
            cls.expert_answers_dir,
            cls.generated_rdrs_dir,
        ]:
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
        if RDRCaseViewer is not None and QApplication is not None and cls.use_gui:
            cls.app = QApplication(sys.argv)
            cls.viewer = RDRCaseViewer()

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = os.path.join(
            self.expert_answers_dir, "scrdr_expert_answers_classify"
        )
        expert = Human(use_loaded_answers=use_loaded_answers, answers_save_path=filename)

        scrdr = SingleClassRDR()
        cat = scrdr.fit_case(
            self.case_queries[0], expert=expert
        )
        self.assertEqual(cat, self.targets[0])

        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)