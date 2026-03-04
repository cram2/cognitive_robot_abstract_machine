from __future__ import annotations

import os
from unittest import TestCase

from typing_extensions import List, Any

from .datasets import Robot, Part, PhysicalObject
from krrood.ripple_down_rules.datastructures.case import CaseAttribute
from krrood.ripple_down_rules.datastructures.dataclasses import (
    CaseQuery,
    CallableExpression,
)
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.rdr import SingleClassRDR
from krrood.ripple_down_rules.utils import flatten_list


class RelationalRDRTestCase(TestCase):
    case: Any
    case_query: Any
    test_results_dir: str = os.path.join(os.path.dirname(__file__), "test_results")
    expert_answers_dir: str = os.path.join(
        os.path.dirname(__file__), "test_expert_answers"
    )
    robot: Robot
    part_a: Part
    part_b: Part
    part_c: Part
    part_d: Part
    part_e: Part
    part_f: Part
    target: List[PhysicalObject]

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.test_results_dir):
            os.makedirs(cls.test_results_dir)
        cls.part_a = Part(name="A")
        cls.part_b = Part(name="B")
        cls.part_c = Part(name="C")
        cls.part_d = Part(name="D")
        cls.part_e = Part(name="E")
        cls.part_f = Part(name="F")
        robot = Robot("pr2", parts=[cls.part_a, cls.part_b, cls.part_c, cls.part_d])
        cls.part_a.contained_objects = [cls.part_b, cls.part_c]
        cls.part_c.contained_objects = [cls.part_d]
        cls.part_d.contained_objects = [cls.part_e]
        cls.part_e.contained_objects = [cls.part_f]
        cls.robot: Robot = robot
        cls.case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False)
        cls.target = [cls.part_b, cls.part_c, cls.part_d, cls.part_e]

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = os.path.join(
            self.expert_answers_dir, "relational_scrdr_expert_answers_classify"
        )
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        cat = scrdr.fit_case(
            CaseQuery(self.robot, "contained_objects", (PhysicalObject,), False),
            expert=expert,
        )
        # render_tree(scrdr.start_rule, use_dot_exporter=True,
        #             filename=self.test_results_dir + "/relational_scrdr_classify")
        self.assertEqual(cat, self.target)

        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_parse_relational_conditions(self):
        user_input = "case.parts is not None and len(case.parts) > 0"
        conditions = CallableExpression(user_input, bool)
        self.assertEqual(
            conditions(self.robot),
            (self.robot.parts is not None and len(self.robot.parts) > 0),
        )

    def test_parse_relational_conclusions(self):
        user_input = "flatten_list([p.contained_objects for p in case.parts])"
        conclusion = CallableExpression(
            user_input,
            (
                CaseAttribute,
                PhysicalObject,
            ),
            scope={"flatten_list": flatten_list},
            mutually_exclusive=False,
        )
        self.assertEqual(conclusion(self.robot), self.target)
