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

This example shows what location designators are, how to use them and what they are capable of.

A location designator describes a region of poses the robot can be sent to, for example every position from which the
robot can reach a certain object. In CoraPlex location designators are the locations in {mod}`coraplex.locations`. A
{class}`~coraplex.locations.base.Location` is iterable: iterating it yields candidate poses, and a plan tries them in
turn until one works. CoraPlex provides locations for:

* Reachability: {class}`~coraplex.locations.locations.ReachabilityLocation`
* Visibility: {class}`~coraplex.locations.locations.VisibilityLocation`

Both sample their candidates from costmaps (see {doc}`../costmap`): 2D grids around the target whose cells are rated
above zero wherever the robot meets the criterion. Both also merge in an occupancy costmap, so every candidate is a
position where the robot can stand without colliding with its surroundings. The costmaps are built from the world as it
is when the candidates are sampled.

We start with a world holding the PR2 in the apartment. The candidates are sampled at random, so the context is given a
seed, which makes this example run the same way every time.

```python
from coraplex.testing import setup_world
from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.robots.pr2 import PR2


world = setup_world()
pr2_view = PR2.from_world(world)
context = Context(world, pr2_view, sampling_seed=0)

origin_pose = pr2_view.root.global_pose
```

## Reachable

A {class}`~coraplex.locations.locations.ReachabilityLocation` describes where the robot can stand to reach a target
pose with one arm, for example an object the robot should pick up. It needs the target and the arm that is to reach it.

We use the milk as the target. The torso of the PR2 is raised first, since otherwise its arms are too low to reach onto
the countertop.

```python
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState

with simulated_robot:
    sequential([ParkArmsAction(pr2_view.all_arms),
                MoveTorsoAction(TorsoState.HIGH)], context=context).perform()

```

```python
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.locations.locations import ReachabilityLocation
from semantic_digital_twin.spatial_types.spatial_types import Pose

location = ReachabilityLocation(
    Pose(reference_frame=world.get_body_by_name("milk.stl")),
    pr2_view.left_arm,
    context=context,
)

plan = execute_single(NavigateAction(location.ground()), context=context)

with simulated_robot:
    plan.perform()

pr2_view.root.parent_connection.origin = origin_pose.homogeneous_matrix
```

{meth}`~coraplex.locations.base.Location.ground` returns the first candidate. It is a pose near the countertop where
the robot can stand without colliding with it, facing the milk, at about half the arm's length from it. Candidates
farther from the target than the arm is long are left out, since the target cannot be reached from there.

The target is given relative to the milk, so the location follows the milk wherever it is when the candidates are
sampled.

## Accessing

How far the robot stands off the target is a fraction of the arm's length, given as a
{class}`~coraplex.datastructures.enums.ReachFraction`. The default, `GRASPING`, suits an object that stays where it is.
A container's handle is pulled towards the robot when it opens, so for a handle the robot stands further back, at
`ACCESSING`.

```python
from coraplex.datastructures.enums import ReachFraction
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle

with world.modify_world():
    world.add_semantic_annotation_recursively(
        drawer := Drawer(
            root=world.get_body_by_name("cabinet10_drawer_middle"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_m")),
        )
    )

location = ReachabilityLocation(
    Pose(reference_frame=drawer.handle.root),
    pr2_view.left_arm,
    ReachFraction.ACCESSING,
    context=context,
)

print(location.ground())
```

## Visible

A {class}`~coraplex.locations.locations.VisibilityLocation` describes where the robot can stand to see a target pose
with its default camera. It only needs the target.

```python
from coraplex.locations.locations import VisibilityLocation

location = VisibilityLocation(
    Pose(reference_frame=world.get_body_by_name("milk.stl")), context=context
)

plan = execute_single(NavigateAction(location.ground()), context=context)

with simulated_robot:
    plan.perform()

pr2_view.root.parent_connection.origin = origin_pose.homogeneous_matrix
```

## Iterating the Candidates

A location is a generator of candidates, which is useful when the first candidate does not work for some reason. A
location samples at most `number_of_samples` candidates. It draws them with its own `seed`, or with the context's
`sampling_seed` if it has none; a seed of `None` draws different candidates each time.

```python
from semantic_digital_twin.spatial_types.spatial_types import Point3

location = VisibilityLocation(
    Pose(Point3.from_iterable([-1, 0, 1.2]), reference_frame=world.root),
    context=context,
    number_of_samples=5,
)

for pose in location:
    print(pose)
```

## Locations in a Plan

A plan does not have to pick a candidate itself. Given as the domain of a variable, a location leaves the pose of an
action open, and the plan tries the candidates in turn when it gets to that action, until the action succeeds with one
of them.

```python
from krrood.entity_query_language.factories import a, variable

navigate = a(NavigateAction)(
    target_location=variable(
        Pose,
        domain=ReachabilityLocation(
            Pose(reference_frame=world.get_body_by_name("milk.stl")),
            pr2_view.left_arm,
            context=context,
        ),
    )
)

with simulated_robot:
    sequential([navigate], context=context).perform()

pr2_view.root.parent_connection.origin = origin_pose.homogeneous_matrix
```

## Tuning the Costmaps

The costmaps a location samples from are square, `map_cells` cells along each side, each cell `map_resolution` meters
wide. The defaults cover 4 m by 4 m around the target. A reachability location also takes `ring_standard_deviation`,
how far, in cells, its candidates spread around the stand-off distance.

```python
location = ReachabilityLocation(
    Pose(reference_frame=world.get_body_by_name("milk.stl")),
    pr2_view.left_arm,
    context=context,
    map_resolution=0.04,
    map_cells=100,
    ring_standard_deviation=8,
)

print(location.ground())
```
