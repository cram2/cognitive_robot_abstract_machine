import os
import unittest
from dataclasses import dataclass
from unittest import TestCase

from typing_extensions import List

from test.krrood_test.test_ripple_down_rules.datasets import load_zoo_dataset
from krrood.ripple_down_rules.datastructures.dataclasses import Case
from krrood.ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from krrood.ripple_down_rules.utils import (
    make_set,
    flatten_list,
    serialize_dataclass,
    deserialize_dataclass,
    render_tree,
)
from test.krrood_test.test_ripple_down_rules.test_helpers.helpers import (
    get_fit_mcrdr,
    get_fit_scrdr,
    get_fit_grdr,
)


@dataclass
class Position:
    x: float
    y: float


@dataclass
class Robot:
    name: str
    position: Position

TEST_RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "test_results")
CACHE_FILE: str = os.path.join(TEST_RESULTS_DIR, "zoo_dataset.pkl")
zoo_cases, _ = load_zoo_dataset(cache_file=CACHE_FILE)


@unittest.skipIf(len(zoo_cases) == 0, "Failed to load dataset")
class TestJSONSerialization(TestCase):
    all_cases: List[Case]
    targets: List[str]
    test_results_dir: str = TEST_RESULTS_DIR
    cache_dir: str = CACHE_FILE
    expert_answers_dir: str = os.path.join(
        os.path.dirname(__file__), "test_expert_answers"
    )

    @classmethod
    def setUpClass(cls):
        cls.all_cases, cls.targets = load_zoo_dataset(cache_file=cls.cache_dir)

    def test_scrdr_json_serialization(self):
        scrdr, _ = get_fit_scrdr(self.all_cases, self.targets)
        filename = os.path.join(self.test_results_dir, "scrdr.json")
        scrdr.to_json_file(filename)
        scrdr = SingleClassRDR.from_json_file(filename)
        for case, target in zip(self.all_cases, self.targets):
            cat = scrdr.classify(case)
            self.assertEqual(cat, target)

    def test_mcrdr_json_serialization(self):
        mcrdr_original = get_fit_mcrdr(self.all_cases, self.targets)
        filename = os.path.join(self.test_results_dir, "mcrdr.json")
        mcrdr_original.to_json_file(filename)
        mcrdr = MultiClassRDR.from_json_file(filename)
        for case, target in zip(self.all_cases, self.targets):
            cat = mcrdr.classify(case)
            cat_original = mcrdr_original.classify(case)
            render_tree(
                mcrdr_original.start_rule,
                use_dot_exporter=True,
                filename=os.path.join(self.test_results_dir, "mcrdr_before_json"),
            )
            render_tree(
                mcrdr.start_rule,
                use_dot_exporter=True,
                filename=os.path.join(self.test_results_dir, "mcrdr_after_json"),
            )
            self.assertEqual(make_set(cat), make_set(target))

    def test_grdr_json_serialization(self):
        grdr, all_targets = get_fit_grdr(self.all_cases, self.targets)
        filename = os.path.join(self.test_results_dir, "grdr.json")
        grdr.to_json_file(filename)
        grdr = GeneralRDR.from_json_file(filename)
        for case, case_targets in zip(self.all_cases[: len(all_targets)], all_targets):
            cat = grdr.classify(case)
            cat = flatten_list(cat)
            case_targets = flatten_list(case_targets)
            self.assertEqual(make_set(cat), make_set(case_targets))

    def test_serialize_dataclass(self):

        robot = Robot("Robo", Position(1.0, 2.0))

        # Serialize
        json = serialize_dataclass(robot)
        # Deserialize
        reconstructed = deserialize_dataclass(json)
        assert robot == reconstructed
