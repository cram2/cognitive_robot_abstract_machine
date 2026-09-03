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

An action describes the plan it expands into itself. A plan transformation changes that plan from
the outside: it is applied to every node it matches, right after that node has been expanded and
before the nodes below it are expanded in turn.

That makes transformations the place for behaviour that is not part of an action's own description,
such as perceiving before a grasp or parking the arms before driving, without giving every action a
parameter for it. Transformations are registered on the `Context`, next to the alternative motion
mappings, so they hold for every plan built with that context.

# Setup a World

```python
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2

world = setup_world()

pr2 = PR2.from_world(world)

context = Context(world, pr2)
```

## Looking at a Plan

Every node carries the short label the plan visualization puts on it, so a few lines are enough to
print an expanded plan:

```python
def show(node, depth=0):
    print("   " * depth + node.__node_label__())
    for child in node.children:
        show(child, depth + 1)
```

## A Reach Without Transformations

`ReachAction` moves the gripper to a pre-pose and then makes its final approach onto the object. The
plan is built by `notify`, which expands the whole plan without executing it.

```python
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.pick_up import ReachAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose

milk = world.get_semantic_annotations_by_type(Milk)[0]

grasp_description = GraspDescription(
    ApproachDirection.FRONT,
    VerticalAlignment.NoAlignment,
    pr2.right_arm.end_effector,
)


reach = execute_single(
    ReachAction(
        target_pose=Pose(reference_frame=milk.root),
        arm=Arms.RIGHT,
        grasp_description=grasp_description,
        object_designator=milk,
    ), context=context)
reach.notify()

show(reach)
```

The two `MoveToolCenterPointMotion` nodes are the pre-pose and the final approach; the condition
nodes around them are the action's pre- and postcondition. The approach grasps at the pose the world
already holds.

## Detecting Before the Grasp

`DetectBeforeGrasp` looks at the object and detects it in front of that final approach, so the
approach acts on a freshly perceived pose. Registering it is the whole change:

```python
from coraplex.robot_plans.plan_transformations import DetectBeforeGrasp

context.plan_transformations.append(DetectBeforeGrasp())

reach = execute_single( ReachAction(
        target_pose=Pose(reference_frame=milk.root),
        arm=Arms.RIGHT,
        grasp_description=grasp_description,
        object_designator=milk,
    ), context=context)
reach.notify()

show(reach)
```

A node is expanded once, so each section builds its own plan rather than expanding the previous one
again. The look and the detection now sit between the pre-pose and the approach, and both were
expanded in turn: each has a plan of its own below it.

The same plan as an interactive graph:

```python
reach.plan.visualize()
```

A transformation fires wherever its action is expanded, so the one registration also covers the
reach that `PickUpAction` builds. Nothing has to be passed down to it:

```python
from coraplex.robot_plans.actions.core.pick_up import PickUpAction

pick_up = execute_single(
    PickUpAction(milk, Arms.RIGHT, grasp_description), context=context
)
pick_up.notify()

show(pick_up)
```

Running such a plan needs a perception source to answer the detection, which is why this notebook
stops at the expanded plan here. The next section performs a plan that a transformation rewrote.

## Opening What the Object Lies In

`OpenDrawerBeforePickUp` puts a drive to the handle and an opening in front of a pick-up whose object
lies in a drawer. To see it, the apartment needs a drawer that something lies in — a spoon in the top
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
it. That needs the pick-up to have a parent, which the surrounding `sequential` gives it:

```python
from coraplex.plans.factories import sequential
from coraplex.robot_plans.plan_transformations import OpenDrawerBeforePickUp

context.plan_transformations = [OpenDrawerBeforePickUp()]

spoon_annotation = world.get_semantic_annotations_by_type(Spoon)[0]

pick_up = sequential(
    [PickUpAction(spoon_annotation, Arms.RIGHT, grasp_description)], context
)
pick_up.notify()

show(pick_up)
```

The drive and the opening now precede the pick-up, and both were expanded in turn. The milk stands in
the open, so the same registration leaves its pick-up alone:

```python
milk_pick_up = sequential([PickUpAction(milk, Arms.RIGHT, grasp_description)], context)
milk_pick_up.notify()

show(milk_pick_up)
```

`TransportAction` opens drawers itself rather than relying on this transformation. It drives to the
object before it picks it up, and that drive is grounded against the world it finds — with the drawer
still shut there is no pose from which the object can be reached, so the opening has to happen before
the drive rather than before the pick-up.

## Writing a Transformation

A transformation answers two questions: which nodes it applies to, and how it rewrites their plan.

The nodes come from the type it is bound to. `ActionTransformation[NavigateAction]` applies to the
node of every navigation; `PlanTransformation[SomeNode]` applies to every node of that type.

The rewrite comes from the base class it is built on. `InsertionTransformation` inserts nodes and
asks for the `anchor` they are placed next to and the `nodes_to_insert`, which are built anew on
every application, since a node belongs to the one plan it was inserted into.

```python
from dataclasses import dataclass

from typing_extensions import List

from coraplex.plans.plan_node import ActionLike, ActionNode, MotionNode, PlanNode
from coraplex.plans.plan_transformation import ActionTransformation, InsertionTransformation
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction


@dataclass
class ParkArmsBeforeNavigating(InsertionTransformation, ActionTransformation[NavigateAction]):
    """
    Parks the arms before the robot drives off, so it does not carry them into the
    furniture it passes.
    """

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        [drive] = [
            node for node in plan_node.descendants if isinstance(node, MotionNode)
        ]
        return drive

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [ParkArmsAction(Arms.BOTH)]
```

```python
context.plan_transformations = [ParkArmsBeforeNavigating()]

navigate = execute_single(
    NavigateAction(Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root)),
    context=context,
)
navigate.notify()

show(navigate)
```

The parking is part of the plan like any other action, so it is performed with it:

```python
with simulated_robot:
    navigate.perform()

print(navigate.status)
```

## Where the Nodes Land

`InsertionTransformation` takes the position the nodes are inserted at: `BEFORE` or `AFTER` the
anchor make them its siblings, `BELOW` makes them its last children. The same transformation with
another position parks the arms once the robot has arrived instead:

```python
from coraplex.datastructures.enums import InsertionPosition

context.plan_transformations = [
    ParkArmsBeforeNavigating(position=InsertionPosition.AFTER)
]

navigate = execute_single(
    NavigateAction(Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root)),
    context=context,
)
navigate.notify()

show(navigate)
```

A transformation that has to look at more than the node type can override `applies_to`, which
decides whether the transformation rewrites a given node. Here the arms are only parked before
drives that are free to move them:

```python
@dataclass
class ParkArmsBeforeFreeDrives(ParkArmsBeforeNavigating):
    """
    Parks the arms only before drives that do not have to hold the joints where they are.
    """

    def applies_to(self, plan_node: PlanNode) -> bool:
        return (
            super().applies_to(plan_node)
            and not plan_node.designator.keep_joint_states
        )
```

```python
context.plan_transformations = [ParkArmsBeforeFreeDrives()]

navigate = execute_single(
    NavigateAction(
        Pose.from_xyz_rpy(1.5, 2.4, 0.0, reference_frame=world.root),
        keep_joint_states=True,
    ),
    context=context,
)
navigate.notify()

show(navigate)
```

This drive has to keep its joint states, so the transformation leaves its plan alone.
