---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Location Designator

This example will show you what location designators are, how to use them and what they are capable of.

Location Designators are used to semantically describe locations in the world. You could, for example, create a location
designator that describes every position where a robot can be placed without colliding with the environment. Location
designator can describe locations for:

* Visibility
* Reachability
* Occupancy
* URDF Links (for example a table)

To find locations that fit the given constrains, location designator create Costmaps. Costmaps are a 2D distribution
that have a value greater than 0 for every position that fits the costmap criteria.

Location designators work similar to other designators, meaning you have to create a location designator description
which describes the location. This description can then be resolved to the actual 6D pose on runtime.

## Occupancy

We will start with a simple location designator that describes a location where the robot can be placed without
colliding with the environment. To do this we need a BulletWorld since the costmaps are mostly created from the current
state of the BulletWorld.

```python
from pycram.testing import setup_world
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.robots.pr2 import PR2


world = setup_world()
pr2_view = PR2.from_world(world)
context = Context(world, pr2_view)
```

Next up we will create the location designator description, the {meth}`~pycram.designators.location_designator.CostmapLocation` that we will be using needs a
target as a parameter. This target describes what the location designator is for, this could either be a pose or object
that the robot should be able to see or reach.

In this case we only want poses where the robot can be placed, this is the default behaviour of the location designator
which we will be extending later.

Since every designator in PyCRAM needs to be part of a plan we create a simple plan which contains our Location Designator.

```python
# from pycram.designators.location_designator import CostmapLocation
# from pycram.language import SequentialPlan
# from pycram.robot_plans import NavigateActionDescription
# 
# location_description = CostmapLocation(world.root)
# 
# location_description = SequentialPlan((world, None), pr2_view, NavigateActionDescription(location_description))
# 
# pose = location_description.resolve()
# 
# print(pose)
```

## Reachable

Next we want to have locations from where the robot can reach a specific point, like an object the robot should pick up. This
can also be done with the {meth}`~pycram.designators.location_designator.CostmapLocation` description, but this time we need to provide an additional argument.
The additional argument is the robot which should be able to reach the pose.

Since a robot is needed we will use the PR2 and use a milk as a target point for the robot to reach. The torso of the
PR2 will be set to 0.2 since otherwise the arms of the robot will be too low to reach on the countertop.

```python
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import *
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from pycram.datastructures.enums import Arms
from semantic_digital_twin.datastructures.definitions import TorsoState

with simulated_robot:
    sequential([ ParkArmsAction(Arms.BOTH),
                   MoveTorsoAction(TorsoState.HIGH)], context=context).perform()

```

```python
from pycram.locations.locations import CostmapLocation, Arms
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.motion_executor import simulated_robot

location_description = CostmapLocation(target=world.get_body_by_name("milk.stl").global_pose, reachable=True, reachable_arm=Arms.LEFT, context=context)

plan = execute_single(NavigateAction(next(iter(location_description))), context=context)

with simulated_robot:
    plan.perform()
```

As you can see we get a pose near the countertop where the robot can be placed without colliding with it. Furthermore,
we get a list of arms with which the robot can reach the given object.

## Visible

The {meth}`~pycram.designators.location_designator.CostmapLocation` can also find position from which the robot can see a given object or location. This is very
similar to how reachable locations are described, meaning we provide a object designator or a pose and a robot
designator but this time we use the ```visible_for``` parameter.

For this example we need the milk as well as the PR2, so if you did not spawn them during the previous location
designator you can spawn them with the following cell.

```python
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
location_description = CostmapLocation(target=Pose(Point3.from_iterable([-1, 0, 1.2]), reference_frame=world.root), visible=True, context=context)

plan = execute_single(NavigateAction(next(iter(location_description))), context=context)

with simulated_robot:
    plan.perform()
```

## Location Designator as Generator

Location designator descriptions implement an iter method, so they can be used as generators which generate valid poses
for the location described in the description. This can be useful if the first pose does not work for some reason.

We will see this at the example of a location designator for visibility. For this example we need the milk, if you
already have a milk spawned in you world you can ignore the following cell.

```python

location_description = CostmapLocation(target=Pose(Point3.from_iterable([-1, 0, 1.2]), reference_frame=world.root), visible=True, context=context)

for i, pose in enumerate(location_description):
    print(pose)
    if i > 3:
        break
```


## Accessing Locations

Accessing describes a location from which the robot can open a drawer. The drawer is specified by the handle that is 
used to open it.

At the moment this location designator only works in the apartment environment, so please remove the kitchen if you
spawned it in a previous example. Furthermore, we need a robot, so we also spawn the PR2 if it isn't spawned already.

```python
from pycram.locations.locations import AccessingLocation

access_location = AccessingLocation(world.get_body_by_name("handle_cab10_t"), arm=Arms.LEFT, context=context)
print(next(iter(access_location)))
```
