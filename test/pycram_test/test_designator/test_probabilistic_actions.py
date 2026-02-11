import random
import unittest

import numpy as np
import pytest
import sqlalchemy.orm
from sqlalchemy import select

from krrood.ormatic.dao import to_dao
from pycram.datastructures.enums import TaskStatus
from pycram.designators.specialized_designators.probabilistic.probabilistic_action import (
    MoveAndPickUpParameterizer,
)
from pycram.failures import PlanFailure
from pycram.language import SequentialPlan
from pycram.orm.ormatic_interface import Base, ActionNodeMappingDAO
from pycram.plan import Plan, ActionNode
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import MoveAndPickUpActionDescription, MoveAndPickUpAction


@pytest.mark.skip
def test_orm(self):
    mpa_description = MoveAndPickUpActionDescription(
        None, [self.world.get_body_by_name("milk.stl")], None, None, None
    )
    plan = SequentialPlan(self.context, mpa_description)
    mpa = MoveAndPickUpParameterizer(mpa_description, world=self.world).create_action()

    plan = Plan(
        ActionNode(designator_ref=mpa, kwargs={}, designator_type=MoveAndPickUpAction),
        self.context,
    )

    with simulated_robot:
        try:
            plan.perform()
        except PlanFailure as e:
            ...

    dao = to_dao(plan)
    self.session.add(dao)
    self.session.commit()

    result = self.session.scalars(select(ActionNodeMappingDAO)).first()
    self.assertEqual(result.status, TaskStatus.SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
