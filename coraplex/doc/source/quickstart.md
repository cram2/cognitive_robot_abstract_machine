# Quickstart Guide

This guide will help you get started with CoraPlex quickly and create your first belief state and plan.

This guide assumes you have already installed CoraPlex either via pip or from source.
If you haven't done so, please refer to the installation instructions in the documentation.

## Intro to Semantic Digital Twin

CoraPlex uses the [semantic_digital_twin](https://github.com/cram2/cognitive_robot_abstract_machine/tree/main/semantic_digital_twin) package to manage the belief
state and the semantic information of it. To learn more about what the semantic digital twin is capable of refer to
its [documentation](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/intro.html).

The semantic digital twin is a belief state representation, that means it represents the robot's knowledge about itself and its environment according
to knowledge present before and sensor inputs that are taken during execution.

On a basis a semantic digital twin world is made up of three main components:

- Bodies: The physical objects in the world including the robot itself.
- Connections: The connection between bodies defining their relative poses and possible movements.
- Degree of Freedoms (DOFs): The possible movements of the connections.

If you are familiar with URDF files, you can think of bodies as links, connections as joints and DOFs as the possible movements of the joints (like prismatic, revolut, etc.).

The bodies and connections build the kinematic structure of the world. In the semantic digital twin everything in the world is
connected via connections starting from a root body. This then creates a tree structure that defines how everything is connected.

The connections define which bodies are connected but not how they move relative to each other. This is where the DOFs come into play.
A connection can have multiple DOFs defining how the child body can move relative to the parent body.

## Setting Up Your Environment

The first step is to set up the environment. This involves loading the robot, environment and object models and configuring
them as needed. For this guide we will use the PR2 in an apartment environment as well as a milk bottle.

Now to set up the environment, we will use URDF files for the robot and the environment and a STL file for the milk bottle.

```python
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser

apartment_world = URDFParser.from_file("<path-to-coraplex>/resources/worlds/apartment.urdf").parse()
milk_world = STLParser("<path-to-coraplex>/resources/objects/milk.stl").parse()
pr2_world = URDFParser.from_file("package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro").parse()
```

```{note}
The PR2 description is resolved from the `iai_pr2_description` ROS package rather than a file shipped with
CoraPlex. CoraPlex itself only bundles `resources/worlds` and `resources/objects`.
```

As you might have noticed each of the parsers returns an independent world which is not really useful. Therefore we need to
merge them into a single world.

```python
from semantic_digital_twin.world import World

world = World()

world.merge(apartment_world)
world.merge(milk_world)
```

To merge the PR2 robot with the world we need to specify the drive that the robot has since it is used to connect the root body
of the robot to the root of the other world. In this case the drive is an OmniDrive. Since we are modifiing the world we
need to use the modify_world context manager.

```python
from semantic_digital_twin.world_description.connections import OmniDrive

with world.modify_world():
    pr2_root = pr2_world.get_body_by_name("base_footprint")
    world_root = world.root
    drive_connection = OmniDrive.create_with_dofs(parent=world_root, child=pr2_root, world=world)
    world.merge_world(pr2_world, drive_connection)
```

Now we have a semantic digital twin that contains the apartment, the milk bottle and the PR2 robot. You can visualize it
using the VizMarkerPublisher, keep in mind that that uses the ROS visualization markers so you need to have ROS installed
and sourced.

```python
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
import rclpy

rclpy.init()
node = rclpy.create_node("simple_viz_node")
viz_publisher = VizMarkerPublisher(_world=world, node=node)
```

You can now open RViz2 and add a Marker display subscribing to the topic "/semworld/viz_marker" to see the world.

## Writing your First Plan

Now that we have set up the environment, we can write our first plan. A plan in CoraPlex is a structured sequence of actions
which the robot will execute to achieve a specific goal. For more details on how plans work in CoraPlex, refer to the
{ref}`statechart_header`.

The plan will consist of the following steps:

1. Park the robot's arms.
2. Raise the robot's torso.
3. Move the robot to a position near the milk bottle.
4. Pick up the milk bottle.
5. Move the robot to a position near the table.
6. Place the milk bottle on the table.

```python
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.datastructures.definitions import TorsoState
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from cramph.composites import Sequence
from cramph.statechart import Statechart

pr2 = PR2.from_world(world)
executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(pr2)])
)
milk_body = world.get_body_by_name("milk.stl")

# The pick-up is told what it is grasping, so annotate the parsed mesh as milk.
with world.modify_world():
    milk = Milk(root=milk_body)
    world.add_semantic_annotation(milk)

plan = Sequence(
    nodes=[
        ParkArmsAction(arms=pr2.all_arms),
        MoveTorsoAction(torso_state=TorsoState.HIGH),
        NavigateAction(
            target_location=Pose.from_xyz_rpy(2.0, 2.0, 0.0, reference_frame=world.root)
        ),
        PickUpAction(
            grasp=milk.grasp_candidates()[0],
            arm=pr2.right_arm,
        ),
        NavigateAction(
            target_location=Pose.from_xyz_rpy(4.0, 4.0, 0.0, reference_frame=world.root)
        ),
        PlaceAction(
            object_designator=milk,
            target_location=Pose.from_xyz_rpy(
                4.2, 4.0, 1.0, reference_frame=world.root
            ),
        ),
    ]
)
statechart = Statechart(context=executor.context)
statechart.add_node(plan)
```

What did we just do here?
We first created an executor, which runs the plan in the world, by the PR2 its context extensions name. A
`KinematicallySimulatedPlanExecutor` runs it in simulation; a `RealRobotPlanExecutor` would send it to the real robot.
Next, we retrieved the milk bottle body from the world to use it in the pick-up and place actions.
Finally, we built a plan as a {class}`~cramph.composites.Sequence`, meaning all actions will be executed one after
another in the order they are defined, and put it into a statechart of the executor's context.

To execute the plan, the executor compiles the statechart and executes it.

```python
executor.compile(statechart)
executor.execute()
```

Congratulations!🎉 You have just written and executed your first plan in CoraPlex.
