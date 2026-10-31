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

# Plan Transformations

An action describes the statechart nodes it expands into itself. A plan transformation changes that
plan from the outside: it is applied to every action and underspecified node it matches, once that
node has been expanded, before the statechart running the plan is compiled.

That makes transformations the place for behaviour that is not part of an action's own description,
such as perceiving before a grasp or parking the arms before driving, without giving every action a
parameter for it. Transformations are given to the executor running a plan, so they hold for every
plan it runs.

# Setup a World

```python
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.testing import setup_world
from cramph.statechart import Statechart
from semantic_digital_twin.robots.pr2 import PR2

world = setup_world()

pr2 = PR2.from_world(world)

plan_transformations = []


def statechart_for(plan):
    """
    :return: An executor configured with the transformations registered so far, and a
        statechart of its context holding `plan`.
    """
    executor = KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(
            world,
            [
                RobotAccess(pr2),
                PlanRewriting(transformations=plan_transformations),
            ],
        ),
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(plan)
    return executor, statechart
```

## Looking at a Plan

Putting a plan into a statechart expands it, and preparing that statechart without compiling or
executing it lets the transformations rewrite it. A few lines are enough to print the expanded plan:

```python
def expand(plan):
    executor, statechart = statechart_for(plan)
    executor.prepare(statechart)
    return plan


def show(node, depth=0):
    print("   " * depth + type(node).__name__)
    for child in node.children:
        show(child, depth + 1)
```

## A Reach Without Transformations

`ReachAction` moves the gripper to a pre-pose and then makes its final approach onto the object.

```python
from cramph.composites import Sequence
from coraplex.robot_plans.actions.core.pick_up import ReachAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose

milk = world.get_semantic_annotations_by_type(Milk)[0]
grasp = milk.grasp_candidates()[0]

reach = expand(Sequence(nodes=[ReachAction(grasp=grasp, arm=pr2.right_arm)]))

show(reach)
```

The two Cartesian goals in the reach's body are the pre-pose and the final approach. The approach
grasps at the pose the world already holds.

## Detecting Before the Grasp

`DetectBeforeGrasp` looks at the object and detects it in front of that final approach, so the
approach acts on a freshly perceived pose. Registering it is the whole change:

```python
from coraplex.robot_plans.plan_transformations import DetectBeforeGrasp

plan_transformations.append(DetectBeforeGrasp())

reach = expand(Sequence(nodes=[ReachAction(grasp=grasp, arm=pr2.right_arm)]))

show(reach)
```

A node belongs to one statechart, so each section builds its own plan rather than expanding the
previous one again. The look and the detection now sit between the pre-pose and the approach, and
both were expanded in turn: each has a body of its own below it.

The same plan as an interactive graph:

```python
reach.statechart.visualize()
```

A transformation fires wherever its action is expanded, so the one registration also covers the
reach that `PickUpAction` builds. Nothing has to be passed down to it:

```python
from coraplex.robot_plans.actions.core.pick_up import PickUpAction

pick_up = expand(Sequence(nodes=[PickUpAction(grasp=grasp, arm=pr2.right_arm)]))

show(pick_up)
```

Running such a plan needs a perception source to answer the detection, which is why this notebook
stops at the expanded plan here. The next section performs a plan that a transformation rewrote.

## Opening What the Object Lies In

`OpenDrawerBeforePickUp` puts an opening of the drawer in front of a pick-up whose object lies in one,
followed by parking the arms and driving back to where the object can be reached from. The opening is
a move-and-open step, so where the robot stands to open the drawer is tried together with the opening
itself. To see it, the apartment needs a drawer that something lies in — a spoon in the top
drawer of cabinet 10:

```python
import os

import coraplex
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle, Spoon
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

spoon = STLParser(
    os.path.join(
        os.path.dirname(coraplex.__file__), "..", "..", "resources", "objects", "spoon.stl"
    )
).parse()

with world.modify_world():
    world.merge_world(
        spoon,
        FixedConnection(
            parent=world.get_body_by_name("cabinet10_drawer_top"),
            child=spoon.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.05, -0.05, 0
            ),
        ),
    )

with world.modify_world():
    world.add_semantic_annotation(Spoon(root=world.get_body_by_name("spoon.stl")))
    world.add_semantic_annotation_recursively(
        Drawer(
            root=world.get_body_by_name("cabinet10_drawer_top"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
        )
    )
```

The transformation is bound to `PickUpAction`, so it inserts next to the pick-up rather than inside
it. That needs the pick-up to run in a sequence, which holds the new neighbours:

```python
from coraplex.robot_plans.plan_transformations import OpenDrawerBeforePickUp

plan_transformations = [OpenDrawerBeforePickUp()]

spoon_annotation = world.get_semantic_annotations_by_type(Spoon)[0]

pick_up = expand(
    Sequence(nodes=[PickUpAction(grasp=spoon_annotation.grasp_candidates()[0], arm=pr2.right_arm)])
)

show(pick_up)
```

The opening, the parking and the drive back to the spoon now precede the pick-up. The opening and the
drive are grounded when they are run, and the parking was expanded in turn. The milk stands in the
open, so the same registration leaves its pick-up alone:

```python
milk_pick_up = expand(Sequence(nodes=[PickUpAction(grasp=grasp, arm=pr2.right_arm)]))

show(milk_pick_up)
```

## Writing a Transformation

A transformation says which nodes it applies to and how their plan changes. Which nodes it applies
to is the type it is bound to: `PlanTransformation[NavigateAction]` matches every navigation,
since an action is a statechart node. `matches_node` does that selection, and `is_applicable` says
whether the case a matched node describes needs the transformation at all, so one that is always
worth applying answers `True`.

How the plan changes is what the subclass brings. `SequentialInsertion` inserts nodes that run right
before or right after another node, and asks for that `position`, the `anchor` they run next to,
and the `nodes_to_insert`, which are built anew on every application, since a node belongs to the
one statechart it was inserted into.

```python
from dataclasses import dataclass

from typing_extensions import List

from coraplex.datastructures.enums import InsertionPosition
from coraplex.plans.plan_transformation import SequentialInsertion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.node import StatechartNode


@dataclass
class ParkArmsBeforeNavigating(SequentialInsertion[NavigateAction]):
    """
    Parks the arms before the robot drives off, so it does not carry them into the
    furniture it passes.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: NavigateAction) -> bool:
        return True

    def anchor(self, plan_node: NavigateAction) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: NavigateAction) -> List[StatechartNode]:
        return [ParkArmsAction(arms=plan_node.robot.all_arms)]
```

```python
plan_transformations = [ParkArmsBeforeNavigating()]

navigate = expand(
    Sequence(
        nodes=[NavigateAction(target_location=Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root))]
    )
)

show(navigate)
```

The parking is part of the plan like any other action, so it is performed with it:

```python
navigate = Sequence(
    nodes=[NavigateAction(target_location=Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root))]
)

executor, statechart = statechart_for(navigate)
executor.compile(statechart)
executor.execute()

print(navigate.life_cycle_state)
```

## Where the Nodes Land

A sequential insertion says when its nodes run: `BEFORE` or `AFTER` the anchor. They become steps of
the sequence that runs the anchor; where no sequence runs it, because it runs in a `Parallel` or is
the whole plan, a new sequence takes the anchor's place and runs it together with the inserted
nodes. The position is part of what the rewrite is rather than something its caller passes, so
parking once the robot has arrived is a rewrite of its own:

```python
@dataclass
class ParkArmsAfterNavigating(ParkArmsBeforeNavigating):
    """
    Parks the arms once the robot has arrived instead of before it drives off.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.AFTER


plan_transformations = [ParkArmsAfterNavigating()]

navigate = expand(
    Sequence(
        nodes=[NavigateAction(target_location=Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root))]
    )
)

show(navigate)
```

Nodes that run at the same time as the anchor come from a `ParallelInsertion` instead. They join the
`Parallel` that runs the anchor; where none runs it, a new one takes the anchor's place. Looking at
where the robot is headed while it drives is such a rewrite:

```python
from coraplex.plans.plan_transformation import ParallelInsertion
from coraplex.robot_plans.actions.core.navigation import LookAtAction


@dataclass
class LookAtTheGoalWhileNavigating(ParallelInsertion[NavigateAction]):
    """
    Looks at where the robot is headed for as long as it drives there.
    """

    def is_applicable(self, plan_node: NavigateAction) -> bool:
        return True

    def anchor(self, plan_node: NavigateAction) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: NavigateAction) -> List[StatechartNode]:
        return [LookAtAction(target=plan_node.target_location)]


plan_transformations = [LookAtTheGoalWhileNavigating()]

navigate = expand(
    Sequence(
        nodes=[NavigateAction(target_location=Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root))]
    )
)

show(navigate)
```

A transformation that is not needed in every case answers `is_applicable` with the question its
case asks, and is only asked about the nodes it matches. Here the arms are only parked before
drives that actually take the robot somewhere:

```python
@dataclass
class ParkArmsBeforeLongDrives(ParkArmsBeforeNavigating):
    """
    Parks the arms only before drives that take the robot somewhere else, not before one
    that leaves it where it already stands.
    """

    minimum_distance: float = 0.5
    """
    How far a drive has to take the robot for parking the arms to be worth it.
    """

    def is_applicable(self, plan_node: NavigateAction) -> bool:
        navigate = plan_node
        target = navigate.world.transform(navigate.target_location, navigate.world.root)
        distance = navigate.robot.root.global_pose.position.euclidean_distance(
            target.position
        )
        return float(distance) > self.minimum_distance
```

```python
plan_transformations = [ParkArmsBeforeLongDrives()]

navigate = expand(Sequence(nodes=[NavigateAction(target_location=pr2.root.global_pose)]))

show(navigate)
```

This drive leaves the robot where it already stands, so the transformation leaves its plan alone.
