from krrood.ormatic.data_access_objects.helper import get_dao_classfrom pycram.orm.ormatic_interface import ResolvedActionNodeMappingDAOfrom pycram.orm.model import ResolvedActionNodeMapping---
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

# Hands on Object Relational Mapping in PyCram

(orm_example)=

This tutorial will walk you through the serialization of a minimal plan in pycram.
First we will import sqlalchemy, create an in-memory database and connect a session to it.

```python
import sqlalchemy.orm
from krrood.ormatic.utils import create_engine

engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
session = sqlalchemy.orm.Session(engine)
```

Next, we need a mapper_registry to map our classes to the database tables. We will use the default mapper_registry from sqlalchemy.

```python
import pycram.orm.ormatic_interface
from pycram.orm.ormatic_interface import *

pycram.orm.ormatic_interface.Base.metadata.create_all(engine)
```

Next, we will write a simple plan where the robot parks its arms, moves somewhere, picks up an object, navigates somewhere else, and places it. 

```python
from pycram.robot_plans import *
from pycram.motion_executor import simulated_robot
from pycram.robot_plans.actions.composite.transporting import TransportAction, MoveTorsoAction
from pycram.datastructures.enums import Arms, Grasp
from pycram.plans.factories import *
from pycram.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2, TorsoState
from pycram.datastructures.dataclasses import Context


world = setup_world()
pr2_view = PR2.from_world(world)
context = Context(world, pr2_view)

description = TransportAction(world.get_body_by_name("milk.stl"),
                                         Pose.from_xyz_quaternion(3, 2.2, 0.95,
                                                                0.0, 0.0, 1.0, 0.0, reference_frame=world.root),
                                         Arms.LEFT)
plan = sequential([MoveTorsoAction(TorsoState.HIGH),
        description], context=context).plan
with simulated_robot:
    plan.perform()
```

The data obtained throughout the plan execution, including robot states, poses, action descriptions and more will be
logged into the database once we insert the plan .

```python
from krrood.ormatic.data_access_objects.helper import to_dao, get_dao_class

session.add(to_dao(plan))
session.commit()
```

Now we can query the database to see what we have logged. Let's say we want to see all the NavigateActions that occurred.

```python
from sqlalchemy import select
from pycram.robot_plans.actions.core.navigation import NavigateAction

navigations = session.scalars(select(get_dao_class(NavigateAction))).all()
print(*navigations, sep="\n")
```

This should print all the pick up actions that occurred during the plan execution, which is one.

Due to the inheritance mapped in the ORM package, we can also get all executed actions with just one query.

```python
from pycram.robot_plans.actions.base import ActionDescription

actions = session.scalars(select(get_dao_class(ActionDescription))).all()
print(*actions, sep="\n")
```

This should print all the actions that occurred during the plan execution, which is five.

If you want to know more about the memory component, read the documentation of the 
[KRR component](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/intro.html).