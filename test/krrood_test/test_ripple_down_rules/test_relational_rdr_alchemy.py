from __future__ import annotations

import os
from unittest import TestCase

from sqlalchemy import ForeignKey
from sqlalchemy.orm import (
    MappedAsDataclass,
    DeclarativeBase,
    declared_attr,
    Mapped,
    mapped_column,
    relationship,
)
from typing_extensions import List, Any, Set

from krrood.ripple_down_rules.datastructures.case import CaseAttribute
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.rdr import SingleClassRDR
from krrood.ripple_down_rules.utils import render_tree


class Base(MappedAsDataclass, DeclarativeBase):

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__


class PolymorphicIdentityMixin:

    @declared_attr.directive
    def __mapper_args__(cls):
        return {
            "polymorphic_identity": cls.__tablename__,
        }


class HasPart(Base):
    left_id: Mapped[int] = mapped_column(
        ForeignKey("PhysicalObject.id"), primary_key=True, init=False, repr=False
    )
    right_id: Mapped[int] = mapped_column(
        ForeignKey("PhysicalObject.id"), primary_key=True, init=False, repr=False
    )
    left: Mapped[PhysicalObject] = relationship(
        back_populates="has_part_relations", foreign_keys=[left_id]
    )
    right: Mapped[PhysicalObject] = relationship(
        back_populates="part_of_relations", foreign_keys=[right_id]
    )

    def __hash__(self):
        return hash(id(self))


class ContainsObject(Base):
    left_id: Mapped[int] = mapped_column(
        ForeignKey("PhysicalObject.id"), primary_key=True, init=False, repr=False
    )
    right_id: Mapped[int] = mapped_column(
        ForeignKey("PhysicalObject.id"), primary_key=True, init=False, repr=False
    )
    left: Mapped[PhysicalObject] = relationship(
        back_populates="contains_objects_relations", foreign_keys=[left_id]
    )
    right: Mapped[PhysicalObject] = relationship(
        back_populates="is_contained_in_relations", foreign_keys=[right_id]
    )

    def __hash__(self):
        return hash(id(self))


class PhysicalObject(Base):
    id: Mapped[int] = mapped_column(primary_key=True, init=False, autoincrement=True)
    name: Mapped[str]
    has_part_relations: Mapped[List[HasPart]] = relationship(
        init=False,
        back_populates="left",
        foreign_keys=[HasPart.left_id],
        repr=False,
        default_factory=list,
    )
    part_of_relations: Mapped[List[HasPart]] = relationship(
        init=False,
        back_populates="right",
        foreign_keys=[HasPart.right_id],
        repr=False,
        default_factory=list,
    )
    contains_objects_relations: Mapped[List[ContainsObject]] = relationship(
        init=False,
        back_populates="left",
        foreign_keys=[ContainsObject.left_id],
        repr=False,
        default_factory=list,
    )
    is_contained_in_relations: Mapped[List[ContainsObject]] = relationship(
        init=False,
        back_populates="right",
        foreign_keys=[ContainsObject.right_id],
        repr=False,
        default_factory=list,
    )
    type: Mapped[str] = mapped_column(init=False)

    @property
    def parts(self) -> List[PhysicalObject]:
        return [has_part.right for has_part in self.has_part_relations]

    @property
    def part_of(self) -> List[PhysicalObject]:
        return [has_part.left for has_part in self.part_of_relations]

    @property
    def contained_objects(self) -> List[PhysicalObject]:
        return [cont_obj.right for cont_obj in self.contains_objects_relations]

    @property
    def is_contained_in(self) -> List[PhysicalObject]:
        return [cont_obj.left for cont_obj in self.is_contained_in_relations]

    @declared_attr.directive
    def __mapper_args__(cls):
        return {
            "polymorphic_on": "type",
            "polymorphic_identity": cls.__tablename__,
        }

    def __hash__(self):
        return hash(id(self))


class RelationalRDRTestCase(TestCase):
    case: Any
    case_query: Any
    test_results_dir: str = os.path.join(os.path.dirname(__file__), "test_results")
    expert_answers_dir: str = os.path.join(
        os.path.dirname(__file__), "test_expert_answers"
    )
    robot: PhysicalObject
    part_a: PhysicalObject
    part_b: PhysicalObject
    part_c: PhysicalObject
    part_d: PhysicalObject
    part_e: PhysicalObject
    part_f: PhysicalObject
    rob_has_parts: List[HasPart]
    containments: List[ContainsObject]
    target: List[PhysicalObject]

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.test_results_dir):
            os.makedirs(cls.test_results_dir)
        cls.part_a = PhysicalObject(name="A")
        cls.part_b = PhysicalObject(name="B")
        cls.part_c = PhysicalObject(name="C")
        cls.part_d = PhysicalObject(name="D")
        cls.part_e = PhysicalObject(name="E")
        cls.part_f = PhysicalObject(name="F")
        robot = PhysicalObject(name="pr2")
        rob_parts = [cls.part_a, cls.part_b, cls.part_c, cls.part_d]
        cls.rob_has_parts = [HasPart(left=robot, right=part) for part in rob_parts]
        cls.containments = []
        cls.containments.append(ContainsObject(left=cls.part_a, right=cls.part_b))
        cls.containments.append(ContainsObject(left=cls.part_a, right=cls.part_c))
        cls.containments.append(ContainsObject(left=cls.part_c, right=cls.part_d))
        cls.containments.append(ContainsObject(left=cls.part_d, right=cls.part_e))
        cls.containments.append(ContainsObject(left=cls.part_e, right=cls.part_f))
        cls.robot: PhysicalObject = robot
        cls.case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False)
        cls.target = [cls.part_b, cls.part_c, cls.part_d, cls.part_e]

    def test_setup(self):
        self.assertEqual(
            self.robot.parts, [self.part_a, self.part_b, self.part_c, self.part_d]
        )
        for part in self.robot.parts:
            self.assertEqual(len(part.part_of), 1)
            self.assertEqual(part.part_of[0], self.robot)
        self.assertEqual(self.robot.contained_objects, [])
        self.assertEqual(self.part_a.contained_objects, [self.part_b, self.part_c])
        self.assertEqual(self.part_c.contained_objects, [self.part_d])
        self.assertEqual(self.part_d.contained_objects, [self.part_e])
        self.assertEqual(self.part_e.contained_objects, [self.part_f])

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = self.expert_answers_dir + "/relational_scrdr_expert_answers_classify"
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
        user_input = "case.parts.contained_objects"
        conclusion = CallableExpression(
            user_input,
            (
                CaseAttribute,
                PhysicalObject,
            ),
            mutually_exclusive=False,
        )
        self.assertEqual(conclusion(self.robot), self.target)
