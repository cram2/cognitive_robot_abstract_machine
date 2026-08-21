---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(writing-a-robot-plan-exercise)=
# Writing a Robot Plan

Author: **Luca Krohm** ([LucaKro](https://github.com/LucaKro), krohm@uni-bremen.de)

In this exercise you write, step by step, the plan behind
`coraplex/demos/coraplex_kitchen_fridge_demo/demo.py`: a PR2 takes a milk out of a closed
fridge and puts it down on the kitchen island.

Every section acts on the world the section before it left behind, so by the end the whole
transport has been carried out once, and the last section wraps it into the demonstration
class the demo ships.

You will:
- Move the robot's body: park its arms, raise its torso, drive its base
- Work out where the robot has to stand to reach something
- Open and shut a fridge door
- Pick the milk up and put it down somewhere else
- Leave a plan underspecified, so it settles against the world it finds at execution time
- Derive the opening and shutting from where the milk stands, instead of writing them down

+++ {"tags": ["exercise"]}

```{note}
Stuck on a section? The [published version of this page](https://lucakro.github.io/cognitive_robot_abstract_machine/coraplex/self_assessment/exercises/0_writing_a_robot_plan.html)
carries a worked solution for every one of them.
```

+++

## 0. Setup

This section builds the kitchen with a PR2 in it, spawns the milk in the fridge, and
assembles the plan context.

```{code-cell} ipython3
:tags: [remove-input]
import logging
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import rclpy
from typing_extensions import ClassVar, List, Optional

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.execution_environment import simulated_robot
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import giskard_reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import (
    ActionLike,
    ActionNode,
    PlanNode,
    UnderspecifiedNode,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import a, an, entity, the, variable
from krrood.entity_query_language.predicate import symbolic_function
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.exceptions import ExerciseVerificationFailed
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import HasDoors, IsStorageSpace
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Fridge,
    Milk,
    ShelfLayer,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Mesh, Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

logging.disable(logging.CRITICAL)

kitchen_urdf = (
    Path(files("coraplex")).parent.parent
    / "resources"
    / "worlds"
    / "kitchen-small.urdf"
)

milk_mesh = (
    Path(files("coraplex")).parent.parent / "resources" / "objects" / "milk.stl"
)

SHELF_LAYER_NAME = "fridge_shelf"

MILK_NAME = "milk"

# How far, in meters, the base may end up from a navigation target. A motion is driven to
# the controller's own tolerance and then wound down, and the base settles a little further
# during that wind-down, so this is looser than the tolerance the navigation itself holds.
ARRIVAL_TOLERANCE = 0.1

# How far, in meters, the milk may end up from where it was placed. Nothing here is subject
# to gravity or contact forces, so a placement that goes to plan lands on the target rather
# than near it.
PLACEMENT_TOLERANCE = 0.02
```

### The kitchen

The kitchen comes from a URDF, the robot from a
[`RobotSpecification`](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/autoapi/semantic_digital_twin/api/index.html#semantic_digital_twin.api.RobotSpecification), the shelf layer the milk stands on
from a body specification, and the milk carton itself from a mesh file. The
[`WorldReasoner`](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/autoapi/semantic_digital_twin/reasoning/world_reasoner/index.html#semantic_digital_twin.reasoning.world_reasoner.WorldReasoner) turns the URDF's bodies
into a *fridge* with a *door* and a *handle*, which is what the plan below reaches for.

```{code-cell} ipython3
def build_kitchen_world() -> World:
    """
    Load the kitchen from its URDF, put the robot in it, infer what its bodies mean

    -- which is what turns the fridge's parts into a fridge with a door and a handle --
    and stand a shelf layer in the fridge.
    """
    world = WorldSpecification.from_urdf(
        str(kitchen_urdf),
        robots=[
            RobotSpecification(
                semantic_annotation_type=PR2,
                world_T_odom=HomogeneousTransformationMatrix(),
            )
        ],
    ).to_domain_object()

    with world.modify_world():
        WorldReasoner(world).reason()

    fridge = variable(Fridge, domain=world.semantic_annotations)
    fridge_annotation = the(entity(fridge)).first()
    shelf_layer = ShelfLayer.get_annotation_specification(
        SHELF_LAYER_NAME,
        BodySpecification.box(
            SHELF_LAYER_NAME,
            Scale(0.45, 0.5, 0.02),
            color=Color(0.9, 0.93, 0.95),
        ),
    ).spawn(
        world,
        parent=fridge_annotation.root,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            0.0, 0.02, 0.0, yaw=np.pi
        ),
    )
    with world.modify_world():
        fridge_annotation.add(shelf_layer)

    return world


def spawn_milk_in_world(world: World) -> None:
    """
    Spawn the milk where the demonstration starts it.

    The carton's shape is the mesh it is spawned from. That mesh's own origin lies below
    its centre, while a body is placed, and a spot on a surface sampled for it, by
    measuring equal halves out from its frame, so the geometry is centred on that frame
    here.

    :param world: The world holding the fridge.
    """
    carton = Mesh(filename=str(milk_mesh))
    specification = Milk.get_annotation_specification(
        MILK_NAME,
        BodySpecification.mesh(
            MILK_NAME,
            carton.filename,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                *-carton.mesh.bounds.mean(axis=0)
            ),
        ),
    )

    shelf_layer = variable(ShelfLayer, domain=world.semantic_annotations)
    shelf_layer_body = (
        the(entity(shelf_layer).where(shelf_layer.name.name == SHELF_LAYER_NAME))
        .first()
        .root
    )
    specification.spawn(
        world,
        parent=shelf_layer_body,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.16,
            0.0,
            (
                shelf_layer_body.collision.scale.z
                + specification.root_specification.scale.z
            )
            / 2,
        ),
    )


world = build_kitchen_world()
spawn_milk_in_world(world)
```

### Watching it in RViz

[`VizMarkerPublisher`](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/autoapi/semantic_digital_twin/adapters/ros/visualization/viz_marker/index.html#semantic_digital_twin.adapters.ros.visualization.viz_marker.VizMarkerPublisher)
publishes the world as markers, and `with_tf_publisher()` starts the tf publisher the markers
are positioned against. Every cell you run below then shows up in RViz.

To see it, add a **MarkerArray** display, set its topic to `/semworld/viz_marker`, set its
durability policy to **Transient Local**, and set the fixed frame to the tf root.
If you are doing this in the Virtual Research Lab, your RVIZ should already be configured correctly.

```{code-cell} ipython3
if not rclpy.ok():
    rclpy.init()

ros_node = rclpy.create_node("writing_a_robot_plan")
VizMarkerPublisher(_world=world, node=ros_node).with_tf_publisher()
```

### The context

A [`Context`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/datastructures/dataclasses/index.html#coraplex.datastructures.dataclasses.Context) says which world a plan acts in and
which robot carries it out. `evaluate_conditions` turns each action's pre- and post-conditions
into monitors that run alongside the motion. With it switched on, an action whose precondition 
does not hold fails *before it moves*. 
This will become relevant in Section 5.
The `_debug` flag will provide additional visual marker in RVIZ while the robot is doing geometry 
calculations. 

```{code-cell} ipython3
robot = variable(PR2, domain=world.semantic_annotations)
context = Context(
    world=world,
    robot=the(entity(robot)).first(),
    ros_node=ros_node,
    evaluate_conditions=True,
    _debug=True,
)
```

### Grasping bodies

For now, we decide that the door is opened with the **left** arm, the milk taken with the **right**.

A [`GraspDescription`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/datastructures/grasp/index.html#coraplex.datastructures.grasp.GraspDescription) says how a body is taken hold of.
The milk's is given here, with its approach direction left as `...`, which means *any member
of that enum*, so the side the gripper comes from is settled while the plan runs rather than
now.
Underspecified Statements like that were part of the previous [Chapter 3 tutorial](https://vrb.ease-crc.org/aicor-tutorial/chapter3/), but also feel free
to ask the tutors about them if you have any questions.

```{code-cell} ipython3
DOOR_ARM = Arms.LEFT
MILK_ARM = Arms.RIGHT


def get_grasp_for_milk(context: Context) -> Match[GraspDescription]:
    """
    Describe a grasp without saying which side of the object the gripper comes from.
    The approach direction is left unspecified, as it is not known until the plan is executed.
    We do not want to pick up the milk from the top in this case, so the VerticalAlignment is
    set to NoAlignment.

    :param context: The context holding the robot that reaches for it.
    :return: The grasp, still open on its approach direction.
    """
    return a(GraspDescription)(
        approach_direction=...,
        vertical_alignment=VerticalAlignment.NoAlignment,
        end_effector=ViewManager.get_end_effector_view(MILK_ARM, context.robot),
    )


fridge = the(entity(variable(Fridge, domain=world.semantic_annotations))).first()
milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()

handle_body = fridge.doors[0].handle.root
milk_body = milk.root
```

## 1. Your first plan: move the robot's own body

A plan is a tree of steps. [`sequential`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/plans/factories/index.html#coraplex.plans.factories.sequential) builds one whose
children run one after another, and `perform()` on the node it returns executes it.

Actions describing the robot's own body name a state rather than a place. 
[`ParkArmsAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/robot_body/index.html#coraplex.robot_plans.actions.core.robot_body.ParkArmsAction)
folds the specified arms into a folding position. The possible options are `Arms.LEFT`, 
`Arms.RIGHT` and `Arms.BOTH`.
[`MoveTorsoAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/robot_body/index.html#coraplex.robot_plans.actions.core.robot_body.MoveTorsoAction) moves the torso to one of the three states `TorsoState.LOW`, 
`TorsoState.MID`, and `TorsoState.HIGH`.


Execution happens inside an *execution environment*, which decides whether the motions drive
a real robot or a simulated one. `simulated_robot(collision_avoidance=True)` returns a
simulated environment that keeps every motion clear of the furniture.

Your goal:
- Build a sequential plan that parks both arms and raises the torso to a `high` state.
  Afterwards store it in a variable named `plan`, and perform it in a simulated environment with
  collision avoidance

```{code-cell} ipython3
:tags: [exercise]
# TODO: park both arms, raise the torso, and perform the plan
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the plan.")
if not isinstance(plan, PlanNode): raise ExerciseVerificationFailed("`sequential` returns the plan's root node.")
if not context.robot.get_torso().get_joint_state_by_type(TorsoState.HIGH).is_achieved(): raise ExerciseVerificationFailed("The torso should have reached its HIGH state.")
```

## 2. Driving the base

[`NavigateAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/navigation/index.html#coraplex.robot_plans.actions.core.navigation.NavigateAction) drives the base to a pose, taking that pose as its `target_location`.

The pose below is a free patch of floor between the kitchen island and the counters. Writing
standing poses out by hand works, but its tedious and not very dynamic. 
The next section lets the robot work one out instead.

Your goal:
- Navigate to `free_spot` and perform the plan

```{code-cell} ipython3
:tags: [exercise]
free_spot = Pose.from_xyz_rpy(0.4, -0.5, 0.0, reference_frame=world.root)

# TODO: drive the base to `free_spot`
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
free_spot = Pose.from_xyz_rpy(0.4, -0.5, 0.0, reference_frame=world.root)

plan = sequential([NavigateAction(target_location=free_spot)], context)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the navigation plan.")
if not np.allclose(context.robot.root.global_pose.to_position().to_np()[:2], free_spot.to_position().to_np()[:2], atol=ARRIVAL_TOLERANCE): raise ExerciseVerificationFailed("The robot should be standing at `free_spot`.")
```

## 3. Working out where to stand

[`giskard_reachability_location`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/locations/factories/index.html#coraplex.locations.factories.giskard_reachability_location) hands out poses the robot
can reach a target from, given the target, the context and the arm. `ground()` takes the first
one. A pose only comes out after the robot has driven there and reached the target, so it has
already been tried.

Your goal:
- Ground a `giskard_reachability_location` for the `handle`, reached with `DOOR_ARM`, into a
  variable named `handle_standing_pose`
- Navigate there

```{code-cell} ipython3
:tags: [exercise]
# TODO: find a pose the handle can be reached from, then drive to it
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

handle_standing_pose = ...
plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
handle_standing_pose = giskard_reachability_location(
    handle_body, context, DOOR_ARM
).ground()

plan = sequential(
    [NavigateAction(target_location=handle_standing_pose)],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if handle_standing_pose is ...: raise ExerciseVerificationFailed("Ground a location for the handle.")
if plan is ...: raise ExerciseVerificationFailed("Build the navigation plan.")
if not isinstance(handle_standing_pose, Pose): raise ExerciseVerificationFailed("A location grounds to a Pose.")
if not np.allclose(context.robot.root.global_pose.to_position().to_np()[:2], handle_standing_pose.to_position().to_np()[:2], atol=ARRIVAL_TOLERANCE): raise ExerciseVerificationFailed("The robot should be standing at the pose the location handed out.")
```

## 4. Opening the fridge

[`OpenAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/container/index.html#coraplex.robot_plans.actions.core.container.OpenAction) takes the *handle* body of the thing that opens, the arm to open it with,
and how far to swing it. Its `goal_joint_state` is in the units of the joint behind the 
handle. (radians for a hinge, meters for a drawer slider)

The hinge's own limit is about 1.5708 rad, but for noise and safety reasons, lets a be a bit
more cautious and only allow the door to open to about 1.45 rad.

Park the arms afterwards, with
[`ParkArmsAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/robot_body/index.html#coraplex.robot_plans.actions.core.robot_body.ParkArmsAction).

Your goal:
- Swing the fridge door open to `DOOR_OPENING_ANGLE` with `DOOR_ARM`, then park both arms

```{code-cell} ipython3
:tags: [exercise]
DOOR_OPENING_ANGLE = 1.45

# TODO: open the fridge door, then park the arms
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
DOOR_OPENING_ANGLE = 1.45

plan = sequential(
    [
        OpenAction(handle_body, DOOR_ARM, goal_joint_state=DOOR_OPENING_ANGLE),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the opening plan.")
door_angle = fridge.doors[0].root.parent_connection.position
if door_angle < DOOR_OPENING_ANGLE - 0.1: raise ExerciseVerificationFailed(f"The fridge door should stand open at about {DOOR_OPENING_ANGLE} rad, but it is at {door_angle:.3f}.")
```

## 5. Taking the milk, without deciding everything up front

Now the milk, with [`PickUpAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/pick_up/index.html#coraplex.robot_plans.actions.core.pick_up.PickUpAction). Where the
robot has to stand for it cannot be written down when the plan is built: a plan expands every
action before its first motion runs, and at that point the door has only just swung open and
the robot has not moved.

[`DeferredLocation`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/locations/base/index.html#coraplex.locations.base.DeferredLocation) wraps a factory that is called when the
pose is actually needed, which is once execution reaches that step.

That is expressed by leaving the action *underspecified* (recall the [Chapter 3 tutorial](https://vrb.ease-crc.org/aicor-tutorial/chapter3/)):
`a(SomeAction)(...)` describes an action rather than building one, and `variable(SomeType, domain=...)` marks 
an argument as a choice over a domain. At execution time the candidates are tried in turn, and one whose
precondition fails — a hand that cannot reach the milk, or is not free — is discarded before
it moves and the next one is taken. These preconditions are only evaluated if in the `Context` the flag
`evaluate_conditions` is set to `True`.

Your goal:
- Build a plan that navigates to a `deferred` `giskard_reachability_location` for `milk_body`, and then picks it
  up with `MILK_ARM` using the underspecified `get_grasp_for_milk(context)` we defined in the earlier section about
  `Grasping bodies`. Finally, park both arms.
- Perform it

```{code-cell} ipython3
:tags: [exercise]
# TODO: leave the standing pose to be chosen while the plan runs
# deferred_location_domain = DeferredLocation(lambda: ...)
# target_location = variable(Pose, domain=...)
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

deferred_location_domain = ...
target_location = ...
plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
plan = sequential(
    [
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(milk_body, context, MILK_ARM)
                ),
            ),
        ),
        a(PickUpAction)(
            object_designator=milk_body,
            grasp_description=get_grasp_for_milk(context),
            arm=MILK_ARM,
        ),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the pick-up plan.")
pick_up_node = next((node for node in plan.children if isinstance(node, UnderspecifiedNode) and node.underspecified_action.factory is PickUpAction), None)
if pick_up_node is None: raise ExerciseVerificationFailed("The pick-up should be left underspecified with `a(PickUpAction)`.")
navigation_node = next((node for node in plan.children if isinstance(node, UnderspecifiedNode) and node.underspecified_action.factory is NavigateAction), None)
if navigation_node is None: raise ExerciseVerificationFailed("The navigation should be left underspecified too, so its standing pose is chosen at execution time.")
tool_frame = ViewManager.get_end_effector_view(MILK_ARM, context.robot).tool_frame
if milk_body not in world.get_kinematic_structure_entities_of_branch(tool_frame): raise ExerciseVerificationFailed("The milk should be attached to the gripper that picked it up.")
```

## 6. Working out where to put it down

The milk has to go somewhere on the kitchen island. We could now predefine a pose for it like we 
did in the beginning for the navigate action. But that would be a bit boring, wouldn't it?
Let's define some helper functions to work out where to put the milk down.

The `WorldReasoner` we ran while building the world recognised the island's work surface as
a `CounterTop`, and a counter top can sample points on its own surface.
`CounterTop.sample_points_from_surface` returns points that a given body fits on, so the spot comes 
out somewhere different on every run.

There are two counter tops in this kitchen, so the island's is picked out by the name of its
body. `robot_aligned_place_pose` turns a sampled point into a pose by using the robot's own rotation.

```{code-cell} ipython3
def island_counter_top(world: World) -> CounterTop:
    """
    :param world: The world holding the kitchen.
    :return: The counter top the milk is put down on, told apart from the sink's by the
        body the kitchen names it after.
    """
    counter_top = variable(CounterTop, domain=world.semantic_annotations)
    return the(
        entity(counter_top).where(
            counter_top.root.name.name == "kitchen_island_surface"
        )
    ).first()


def place_spot_on_island(world: World, milk: Milk) -> Point3:
    """
    Pick a spot on the kitchen island to put the milk down on.

    :param world: The world holding the kitchen.
    :param milk: The carton the spot has to be big enough for.
    :return: The spot, in the world frame.
    """
    points = island_counter_top(world).sample_points_from_surface(
        body_to_sample_for=milk, amount=100
    )
    return world.transform(next(iter(points)), world.root)


def robot_aligned_place_pose(spot: Point3, context: Context) -> Pose:
    """
    :param spot: Where the object has to end up.
    :param context: The context holding the robot.
    :return: That spot, turned the way the robot is standing.
    """
    return Pose.from_xyz_rpy(
        spot.x,
        spot.y,
        spot.z,
        yaw=context.robot.root.global_pose.yaw,
        reference_frame=context.world.root,
    )


place_spot = place_spot_on_island(world, milk)
```

## 7. Putting the milk down

[`PlaceAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/placing/index.html#coraplex.robot_plans.actions.core.placing.PlaceAction) takes the body, the pose to
put it at, and the arm holding it.

The standing pose is deferred again, for the same reason as before. The navigation is given
`place_spot` itself — a location can take a point as well as a pose.

The placing pose has to be deferred too: `robot_aligned_place_pose` reads the robot's orientation, and
the robot has not driven to the island yet.

Your goal:
- Navigate to a deferred reachability location for `place_spot`, place the milk at a deferred
  `robot_aligned_place_pose` for it with `MILK_ARM`, park both arms, and perform it

```{code-cell} ipython3
:tags: [exercise]
# TODO: carry the milk to the island and put it down
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
plan = sequential(
    [
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(place_spot, context, MILK_ARM)
                ),
            ),
        ),
        a(PlaceAction)(
            object_designator=milk_body,
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: [robot_aligned_place_pose(place_spot, context)]
                ),
            ),
            arm=MILK_ARM,
        ),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the placing plan.")
placed_box = milk_body.collision.as_bounding_box_collection_in_frame(world.root).bounding_box()
surface_box = island_counter_top(world).supporting_surface.area.as_bounding_box_collection_in_frame(world.root).bounding_box()
if abs(placed_box.min_z - surface_box.max_z) > PLACEMENT_TOLERANCE: raise ExerciseVerificationFailed("The milk should be resting on the island's surface.")
if not (surface_box.min_x <= placed_box.min_x and placed_box.max_x <= surface_box.max_x): raise ExerciseVerificationFailed("The milk should stand within the island's surface, not over its edge.")
if not (surface_box.min_y <= placed_box.min_y and placed_box.max_y <= surface_box.max_y): raise ExerciseVerificationFailed("The milk should stand within the island's surface, not over its edge.")
```

## 8. Shutting the fridge again

[`CloseAction`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/robot_plans/actions/core/container/index.html#coraplex.robot_plans.actions.core.container.CloseAction) takes the same arguments as
the opening. The handle is attached the swinging door, so it is somewhere else now than it was when
you stood in front of it.

Your goal:
- Navigate to a deferred reachability location for the handle, shut the door to
  `DOOR_CLOSING_ANGLE`, park both arms, and perform it

```{code-cell} ipython3
:tags: [exercise]
DOOR_CLOSING_ANGLE = 0.0

# TODO: drive back to the handle and shut the door
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
DOOR_CLOSING_ANGLE = 0.0

plan = sequential(
    [
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(handle_body, context, DOOR_ARM)
                ),
            ),
        ),
        CloseAction(handle_body, DOOR_ARM, goal_joint_state=DOOR_CLOSING_ANGLE),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the closing plan.")
DOOR_IS_CLOSED_ANGLE = 0.02
door_angle = fridge.doors[0].root.parent_connection.position
if door_angle >= DOOR_IS_CLOSED_ANGLE: raise ExerciseVerificationFailed(f"The fridge door should be shut, but it stands at {door_angle:.3f} rad.")
```

The transport is done: the milk is on the island and the fridge is shut behind it.

## 9. Deriving the opening and the shutting

Everything so far was written down in the order it had to happen. But the plan's *subject* is
the transport — that the milk happens to stand behind a closed door is a fact about the
world, not about the task. A plan that has the opening written into it only works in a kitchen
whose fridge is shut.

So find the container instead. `InsideOf` measures what share of one body lies within another,
anything meant to hold things carries an `IsStorageSpace` annotation, and one that shuts also
has `HasDoors`. `@symbolic_function` makes a plain predicate usable inside a `.where(...)`,
evaluated per candidate as the query runs.

In this kitchen the milk is held by two storage spaces: the fridge, and the rack the fridge
stands in. Only the fridge has a door, which is what picks it out. A drawer would be found too,
but its handle hangs off the drawer rather than off a door, so only doors are followed here.

Queries like these are the subject of the
[Chapter 3 tutorial](https://vrb.ease-crc.org/aicor-tutorial/chapter3/).

The milk is on the island now, in nothing at all, so this runs against a fresh kitchen.

```{code-cell} ipython3
:tags: [remove-input]
fresh_world = build_kitchen_world()
spawn_milk_in_world(fresh_world)
fresh_fridge = the(entity(variable(Fridge, domain=fresh_world.semantic_annotations))).first()
fresh_milk = the(entity(variable(Milk, domain=fresh_world.semantic_annotations))).first().root
```

```{code-cell} ipython3
@symbolic_function
def contains(
    container: KinematicStructureEntity, body: KinematicStructureEntity
) -> bool:
    """
    :param container: The entity that may hold the body.
    :param body: The entity that may be held.
    :return: Whether nearly all of the body lies inside the container.
    """
    return InsideOf(body, container)() > 0.9


def handle_of_enclosing_container(body: Body, world: World) -> Optional[Body]:
    """
    Find what has to be pulled open before ``body`` can be taken.

    :param body: The body that may stand inside a container.
    :param world: The world holding both.
    :return: The handle of the container, or ``None`` when the body stands in the open or
        in something that does not open.
    """
    storage_space = variable(IsStorageSpace, domain=world.semantic_annotations)
    containers = an(
        entity(storage_space).where(contains(storage_space.root, body))
    ).evaluate()
    return next(
        (
            container.doors[0].handle.root
            for container in containers
            if isinstance(container, HasDoors) and container.doors
        ),
        None,
    )
```

## 10. Wrapping the transport

With the handle in hand, the opening and the shutting become something you put *around* a
list of steps rather than into it. If nothing has to be opened, the steps come back
untouched — and the plan is the transport alone.

Which hand pulls the handle can be left open, by passing `...` for the `arm`: a hand that
cannot reach it fails its precondition and the other is tried. The standing-pose search is
still asked with one named hand.

Your goal:
- Write `add_container_opening_and_closing(actions, body, context) -> List[ActionLike]` that
  returns `actions` unchanged when `body` stands in the open, and otherwise puts a navigation,
  an `OpenAction` and a park in front of it, and a navigation, a `CloseAction` and a park
  behind it, both container actions leaving their arm open

Use `handle_of_enclosing_container` to find the handle.

```{code-cell} ipython3
:tags: [exercise]
# TODO: put the container steps around the given ones
# def add_container_opening_and_closing(actions, body, context) -> List[ActionLike]:
#     ...
#     return [..., *actions, ...]

add_container_opening_and_closing = ...
```

```{code-cell} ipython3
:tags: [example-solution]
def add_container_opening_and_closing(
    actions: List[ActionLike], body: Body, context: Context
) -> List[ActionLike]:
    """
    Open the container ``body`` stands in before the given steps, and shut it again after
    them.

    Which hand does the pulling is left to the plan, and so is the side the gripper comes at
    the handle from; the standing pose is looked for with a named hand.

    :param actions: The steps that need the container open.
    :param body: The body those steps reach for.
    :param context: The context the standing poses are chosen in.
    :return: The steps with the container steps around them, and ``actions`` itself when
        the body stands in the open.
    """
    handle = handle_of_enclosing_container(body, context.world)
    if handle is None:
        return actions

    door_arm = DOOR_ARM

    return [
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(handle, context, door_arm)
                ),
            ),
        ),
        a(OpenAction)(
            object_designator=handle,
            arm=...,
            goal_joint_state=DOOR_OPENING_ANGLE,
        ),
        ParkArmsAction(Arms.BOTH),
        *actions,
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(handle, context, door_arm)
                ),
            ),
        ),
        a(CloseAction)(
            object_designator=handle,
            arm=...,
            goal_joint_state=DOOR_CLOSING_ANGLE,
        ),
        ParkArmsAction(Arms.BOTH),
    ]
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if add_container_opening_and_closing is ...: raise ExerciseVerificationFailed("Write add_container_opening_and_closing.")
fresh_robot = variable(PR2, domain=fresh_world.semantic_annotations)
fresh_context = Context(world=fresh_world, robot=the(entity(fresh_robot)).first(), evaluate_conditions=False)
transport = [ParkArmsAction(Arms.BOTH)]
wrapped = add_container_opening_and_closing(transport, fresh_milk, fresh_context)
if wrapped[3:-3] != transport: raise ExerciseVerificationFailed("The given steps should come back untouched, in the middle.")
if wrapped[1].factory is not OpenAction or wrapped[1].kwargs["object_designator"] is not fresh_fridge.doors[0].handle.root: raise ExerciseVerificationFailed("The fridge door's handle should be the one that is opened.")
if wrapped[1].kwargs["goal_joint_state"] != DOOR_OPENING_ANGLE: raise ExerciseVerificationFailed("The door should be opened to DOOR_OPENING_ANGLE.")
if wrapped[-2].factory is not CloseAction or wrapped[-2].kwargs["goal_joint_state"] != DOOR_CLOSING_ANGLE: raise ExerciseVerificationFailed("The door should be shut again to DOOR_CLOSING_ANGLE.")
if wrapped[1].kwargs["arm"] is not Ellipsis or wrapped[-2].kwargs["arm"] is not Ellipsis: raise ExerciseVerificationFailed("Both container actions should leave their arm open.")
if add_container_opening_and_closing(transport, island_counter_top(fresh_world).root, fresh_context) is not transport: raise ExerciseVerificationFailed("A body standing in the open should leave the steps exactly as they were.")
```

## 11. The whole thing, as a demonstration

[`RobotDemonstration`](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/autoapi/coraplex/demonstrations/index.html#coraplex.demonstrations.RobotDemonstration) is the scaffolding a demo runs on. It
owns the ROS session, decides whether to build a world or fetch one from a running controller,
and wraps execution in the right environment, so a demonstration writes only its own scene and
its own plan, through five methods.

Four of the five methods are given below. The plan states the transport only; what it has to
unpack to get at the milk comes out of `add_container_opening_and_closing`.

Your goal:
- Fill in `build_plan`: the four transport steps from sections 5 and 7, wrapped by
  `add_container_opening_and_closing`, behind a park and a `MoveTorsoAction`

```{code-cell} ipython3
:tags: [exercise]
@dataclass
class KitchenFridgeDemonstration(RobotDemonstration):
    """
    A robot fetches a milk out of the kitchen fridge and puts it on the kitchen island.
    """

    ros_node_name: ClassVar[str] = "kitchen_fridge_demo_node"

    def build_simulated_world(self) -> World:
        return build_kitchen_world()

    def is_scene_populated(self, world: World) -> bool:
        milk = variable(Milk, domain=world.semantic_annotations)
        return next(an(entity(milk)).evaluate(), None) is not None

    def populate_scene(self, world: World) -> None:
        spawn_milk_in_world(world)

    def build_context(self, world: World) -> Context:
        robot = variable(self.used_robot, domain=world.semantic_annotations)
        return Context(
            world=world,
            robot=the(entity(robot)).first(),
            ros_node=ros_node,
            evaluate_conditions=True,
            _debug=True,
        )

    # TODO: build the whole plan
    def build_plan(self, context: Context) -> PlanNode:
        return ...


demonstration = KitchenFridgeDemonstration(used_robot=PR2)
demonstration.run()
```

```{code-cell} ipython3
:tags: [example-solution]
@dataclass
class KitchenFridgeDemonstration(RobotDemonstration):
    """
    A robot fetches a milk out of the kitchen fridge and puts it on the kitchen island.
    """

    ros_node_name: ClassVar[str] = "kitchen_fridge_demo_node"

    def build_simulated_world(self) -> World:
        return build_kitchen_world()

    def is_scene_populated(self, world: World) -> bool:
        milk = variable(Milk, domain=world.semantic_annotations)
        return next(an(entity(milk)).evaluate(), None) is not None

    def populate_scene(self, world: World) -> None:
        spawn_milk_in_world(world)

    def build_context(self, world: World) -> Context:
        robot = variable(self.used_robot, domain=world.semantic_annotations)
        return Context(
            world=world,
            robot=the(entity(robot)).first(),
            ros_node=ros_node,
            evaluate_conditions=True,
            _debug=True,
        )

    def build_plan(self, context: Context) -> PlanNode:
        world = context.world
        milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()
        milk_body = milk.root
        place_spot = place_spot_on_island(world, milk)

        transport = [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            milk_body, context, MILK_ARM
                        )
                    ),
                ),
                ),
            a(PickUpAction)(
                object_designator=milk_body,
                grasp_description=get_grasp_for_milk(context),
                arm=MILK_ARM,
            ),
            ParkArmsAction(Arms.BOTH),
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            place_spot, context, MILK_ARM
                        )
                    ),
                ),
                ),
            a(PlaceAction)(
                object_designator=milk_body,
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: [robot_aligned_place_pose(place_spot, context)]
                    ),
                ),
                arm=MILK_ARM,
            ),
            ParkArmsAction(Arms.BOTH),
        ]

        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                *add_container_opening_and_closing(transport, milk_body, context),
            ],
            context,
        )


demonstration = KitchenFridgeDemonstration(used_robot=PR2)
demonstration.run()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
demo_world = demonstration.build_simulated_world()
if demonstration.is_scene_populated(demo_world): raise ExerciseVerificationFailed("Building the world should not already put the milk into it.")
demonstration.populate_scene(demo_world)
if not demonstration.is_scene_populated(demo_world): raise ExerciseVerificationFailed("Populating the scene should stand the milk in the fridge.")
demo_context = demonstration.build_context(demo_world)
if not demo_context.evaluate_conditions: raise ExerciseVerificationFailed("The demonstration's context should evaluate conditions.")
demo_plan = demonstration.build_plan(demo_context)
steps = [type(node.designator) if isinstance(node, ActionNode) else node.underspecified_action.factory for node in demo_plan.children]
if steps != [ParkArmsAction, MoveTorsoAction, NavigateAction, OpenAction, ParkArmsAction, NavigateAction, PickUpAction, ParkArmsAction, NavigateAction, PlaceAction, ParkArmsAction, NavigateAction, CloseAction, ParkArmsAction]: raise ExerciseVerificationFailed(f"The plan should open the fridge, transport the milk and shut the fridge again, but it reads {steps}.")
demo_place = next(node for node in demo_plan.children if isinstance(node, UnderspecifiedNode) and node.underspecified_action.factory is PlaceAction)
if not isinstance(demo_place.underspecified_action.kwargs["target_location"]._domain_.domain, DeferredLocation): raise ExerciseVerificationFailed("The placing pose reads the robot's orientation, so it has to be deferred until the robot has driven to the island.")
```

## Where to go from here

What you just wrote is, near enough, `coraplex/demos/coraplex_kitchen_fridge_demo/demo.py`.
The shipped version differs mainly in carrying the helpers you wrote here as methods on the
demonstration class, with the shelf layer's name and the carrying arm as documented fields.
Run it with:

```bash
python coraplex/demos/coraplex_kitchen_fridge_demo/demo.py
```

Then try changing something and watch the plan follow: stand the milk on the island to begin
with, and the fridge steps disappear from the plan entirely.
