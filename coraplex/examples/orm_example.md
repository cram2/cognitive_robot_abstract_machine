---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Hands on Object Relational Mapping in CoraPlex

(orm_example)=

This tutorial will walk you through the serialization of a minimal plan in coraplex.
First we will import sqlalchemy, create an in-memory database and connect a session to it.

```python
import sqlalchemy.orm
from krrood.ormatic.utils import create_engine

engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
session = sqlalchemy.orm.Session(engine)
```

Next, we need a mapper_registry to map our classes to the database tables. We will use the default mapper_registry from sqlalchemy.

```python
import coraplex.orm.ormatic_interface
from coraplex.orm.ormatic_interface import *

coraplex.orm.ormatic_interface.Base.metadata.create_all(engine)
```

Next, we will write a simple plan where the robot raises its torso and transports a milk carton: it parks its arms,
navigates to the milk, faces and looks at it, picks it up, navigates to the target, faces and looks at it, and places
the milk there.

Only grounded actions can be stored, so every step of the transport is given here: where the robot stands, which grasp
it takes the milk by and with which arm. A step that is still an EQL query, as the ones
`TransportAction.from_graspable_by_closest_grasps` builds, is refused with a `QueryCannotBePersisted`.

```python
from coraplex.robot_plans import *
from coraplex.execution_environment import simulated_robot
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from coraplex.plans.factories import *
from coraplex.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2, TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.spatial_types.spatial_types import Pose2D

world = setup_world()
pr2_view = PR2.from_world(world)
context = Context(world, pr2_view)

milk = world.get_semantic_annotations_by_type(Milk)[0]
description = TransportAction(
    pick_up=MoveAndPickUpAction.from_standing_position(
        standing_position=Pose2D(1.63, 1.98, reference_frame=world.root),
        grasp=milk.grasp_candidates()[0],
        arm=pr2_view.left_arm,
    ),
    place=MoveAndPlaceAction.from_standing_position(
        standing_position=Pose2D(1.64, 3.24, reference_frame=world.root),
        target_location=Pose.from_xyz_quaternion(
            2.4, 3, 1.05, 0.0, 0.0, 0.0, 1.0, reference_frame=world.root
        ),
        object_designator=milk,
    ),
)
plan = sequential([MoveTorsoAction(torso_state=TorsoState.HIGH),
                   description], context=context).plan
with simulated_robot:
    plan.perform()
```

The data obtained throughout the plan execution, including robot states, poses, action descriptions and more will be
logged into the database once we insert the plan.

```python
from krrood.ormatic.data_access_objects.helper import to_dao, get_data_access_object_class

session.add(to_dao(plan))
session.commit()
```

Now we can query the database to see what we have logged. Let's say we want to see all the NavigateActions that occurred.

```python
from sqlalchemy import select
from coraplex.robot_plans.actions.core.navigation import NavigateAction

navigations = session.scalars(select(get_data_access_object_class(NavigateAction))).all()
print(*navigations, sep="\n")
```

This should print both NavigateActions of the plan: the one to where the robot picks up the milk and the one to where
it places it.

Due to the inheritance mapped in the ORM package, we can also get all executed actions with just one query.

```python
from coraplex.robot_plans.actions.base import ActionDescription

actions = session.scalars(select(get_data_access_object_class(ActionDescription))).all()
print(*actions, sep="\n")
```

This should print every action that occurred during the plan execution, including the steps the composite actions are
made of: besides the MoveTorsoAction and the TransportAction, it lists the TransportAction's ParkArmsActions, its
MoveAndPickUpAction and MoveAndPlaceAction, and the steps inside those, such as the NavigateActions, the PickUpAction and
the PlaceAction.

If you want to know more about the memory component, read the documentation of the 
[KRR component](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/intro.html).