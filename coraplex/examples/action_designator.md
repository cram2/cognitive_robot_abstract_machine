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


# Action Designator

This example will show the different kinds of Action Designators that are available. We will see how to create Action
Designators and what they do.

Action Designators are high-level descriptions of actions which the robot should execute.

Action Designators are created from an Action Designator Description, which describes the type of action as well as the
parameter for this action. Parameter are given as a list of possible parameters.
For example, if you want to describe the robot moving to a table you would need a
{meth}`~coraplex.robot_plans.NavigateActionDescription` and a list of poses that are near the table or a 
LocationDesignator describing a pose near the table. The Action
Designator Description will then pick one of the poses and return a performable Action Designator which contains the
picked pose.

## Preface 
Action designator descriptions are able to handle a multitude of different inputs. In general, they are able to work with 
the argument directly or any iterable that generates the type of the argument. Iterables include a list of the arguments 
or another designator which generates the argument type. For example, a NavigateActionDescription takes as input a Pose 
now the possible input arguments for a NavigateActionDescription are: 

    * A Pose 
    * A list of Poses 
    * A Location Designator, since they are generating Poses  


## Navigate Action

We will start with a simple example of the {class}`~coraplex.robot_plans.actions.core.navigation.NavigateAction`.

First, we need a BulletWorld with a robot.

All plans are performed by an executor, in a world, by a robot. The robot, and how the plan samples its locations, are
given to the executor as context extensions. The `run` function below gives every plan an executor of its own.

```python
import os

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.world import World
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from coraplex.testing import setup_world

world = setup_world()
pr2 = PR2.from_world(world)

# A location samples its candidates from a costmap with the seed of the plan's
# context, so this example runs the same way twice.
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.statechart import Statechart

extensions = [RobotAccess(pr2), StatementGrounding()]
context = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, extensions)
).context


def run(plan):
    """
    Run `plan` simulated in `world`, with the robot and settings in `extensions`.
    """
    executor = KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(world, extensions)
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(plan)
    executor.compile(statechart)
    executor.execute()
    return executor
```

To move the robot we need to create a description which will be resolved to the actual designator. The description of navigation
only needs a list of possible poses. In CoraPlex **every** designator needs to be part of a plan, the plan also manages the 
world in which the designator are executed as well as the robot which executes the plan.

```python
from coraplex.robot_plans.actions.core.navigation import NavigateAction

pose = Pose.from_xyz_quaternion(1.3, 2, 0, 0, 0, 0, 1, reference_frame=world.root)

# This is the Designator Description
plan = NavigateAction(target_location=pose)
```

What we now did was: create the pose where we want to move the robot, create a description describing a navigation with
a list of possible poses (in this case the list contains only one pose) and create plan from the
description.

To execute the created plan, put it into a statechart of an executor, compile and execute it, as `run` does.

```python
run(plan)
```

Where a plan is performed is chosen by its executor: a
{class}`~coraplex.plans.executors.KinematicallySimulatedPlanExecutor` performs it on the simulated robot, a
{class}`~coraplex.plans.executors.RealRobotPlanExecutor` on the real one.

## Move Torso

This action designator moves the torso up or down, specifically it sets the torso joint to a given value.

We start again by creating the action. Afterwards, it is performed by a simulated executor.

```python
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState

torso_pose = TorsoState.HIGH

plan = MoveTorsoAction(torso_state=torso_pose)

run(plan)
```

## Set Gripper

As the name implies, this action designator is used to open or close the gripper. What the gripper is to reach is
described by a gripper configuration, which the end effector of the arm builds from one of its declared states.

The procedure is similar to the last time, but this time we will shorten it a bit.

```python
from coraplex.robot_plans.actions.core.robot_body import SetGripperAction
from semantic_digital_twin.datastructures.definitions import GripperState

configuration = pr2.left_arm.end_effector.default_configuration(GripperState.OPEN)

run(SetGripperAction(configuration=configuration))
```

## Park Arms

Park arms is used to move one or both arms into the default parking position.

```python
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

run(ParkArmsAction(arms=pr2.all_arms))
```

## Pick Up and Place

Since these two are dependent on each other, meaning you can only place something when you picked it up beforehand, they
will be shown together.

These action designators use object designators, which will not be further explained in this tutorial so please check
the example on object designators for more details.

To start we need an environment in which we can pick up and place things as well as an object to pick up.

```python
from semantic_digital_twin.datastructures.definitions import TorsoState
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk

import rclpy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from cramph.composites import Sequence

arm = pr2.right_arm
milk = world.get_semantic_annotations_by_type(Milk)[0]

run(Sequence(nodes=[ParkArmsAction(arms=pr2.all_arms),
     MoveTorsoAction(torso_state=TorsoState.HIGH),
     NavigateAction(
         target_location=Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root)
     ),
     PickUpAction(
         grasp=next(iter(milk.grasp_candidates())),
         arm=arm,
     ),
     PlaceAction(
         object_designator=milk,
         target_location=Pose.from_xyz_rpy(2.4, 2.2, 1, reference_frame=world.root),
     )]))
```

## Look At

Look at lets the robot look at a specific point, for example if it should look at an object for detecting.

```python
from coraplex.robot_plans.actions.core.navigation import LookAtAction

target_location = Pose.from_xyz_rpy(3, 2, 1, reference_frame=world.root)
run(LookAtAction(target=target_location))
```

## Detect

Detect is used to detect objects in the field of vision (FOV) of the robot. We will use the milk placed in the pick
up/place example. The robot drives in front of it, looks at it and detects it, which writes the pose the milk is seen at
into the world.

```python
from coraplex.datastructures.enums import DetectionTechnique
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk

milk = world.get_semantic_annotations_by_type(Milk)[0]

run(Sequence(nodes=[
    ParkArmsAction(arms=pr2.all_arms),
    NavigateAction(target_location=Pose.from_xyz_rpy(1.7, 2, 0, reference_frame=world.root)),
    LookAtAction(target=Pose(reference_frame=milk.root)),
    DetectAction(technique=DetectionTechnique.TYPES, object_sem_annotation=Milk),
]))
```

## Transporting

Transporting can transport an object from its current position to another target position. It is similar to the Pick and
Place plan used in the Pick-up and Place example. Since we need an Object which we can transport we spawn a milk, you
don't need to do this if you already have spawned it in a previous example.

```python
from coraplex.robot_plans import *
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from cramph.composites import Sequence

description = TransportAction.from_graspable_by_closest_grasps(
    world.get_semantic_annotations_by_type(Milk)[0],
    Pose.from_xyz_quaternion(3.0, 2.2, 1.04, 0.0, 0.0, 1.0, 0.0, reference_frame=world.root),
    pr2.left_arm,
    context,
)
run(Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH),
            description]))
```

## Opening

Opening allows the robot to open a drawer, grabbing the drawer's handle by a grasp. `OpenAction.from_handle` tries the
grasps the handle offers, the one closest to where the robot stands first.

For the moment this designator works only in the apartment environment, therefore we remove the kitchen and spawn the
apartment.

```python
from coraplex.robot_plans import *
from semantic_digital_twin.datastructures.definitions import TorsoState
from coraplex.robot_plans.actions.core.container import OpenAction
from cramph.composites import Sequence
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle

# Opening takes a handle by one of the grasps its annotation offers, and tries them until
# one works. The apartment carries no annotation, so we register one for the drawer.
with world.modify_world():
    world.add_semantic_annotation_recursively(
        handle := Handle(root=world.get_body_by_name("handle_cab10_t"))
    )

run(Sequence(nodes=[
    MoveTorsoAction(torso_state=TorsoState.HIGH),
    ParkArmsAction(arms=pr2.all_arms),
    NavigateAction(target_location=Pose.from_xyz_quaternion(1.7074915981292725, 2.6873629093170166, 0.0,
                                            -0.0, 0.0, 0.5253598267689507, -0.850880163370435,
                                            reference_frame=world.root)),
    OpenAction.from_handle(handle, pr2.right_arm)]))
```

## Closing

Closing lets the robot close an open drawer, taking its handle by a grasp just like opening does.

This action designator only works in the apartment environment for the moment, therefore we remove the kitchen and spawn
the apartment. Additionally, we open the drawer such that we can close it with the action designator.

```python
from coraplex.robot_plans.actions.core.container import CloseAction
from cramph.composites import Sequence

run(Sequence(nodes=[
    MoveTorsoAction(torso_state=TorsoState.HIGH),
    ParkArmsAction(arms=pr2.all_arms),
    NavigateAction(target_location=Pose.from_xyz_quaternion(1.72, 2.65, 0.0,
                                            -0.0, 0.0, 0.5253598267689507, -0.850880163370435,
                                            reference_frame=world.root)),
    CloseAction.from_handle(handle, pr2.right_arm)]))
```
