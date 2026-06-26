================
Quickstart Guide
================

This guide will help you get started with CoraPlex quickly and create your first belief state and plan.

This guide assumes you have already installed CoraPlex either via pip or from source.
If you haven't done so, please refer to the installation instructions in the documentation.


------------------------------
Intro to Semantic Digital Twin
------------------------------

CoraPlex uses the `semantic_digital_twin <https://github.com/cram2/cognitive_robot_abstract_machine/tree/main/semantic_digital_twin>`__ package to manage the belief
state and the semantic information of it. To learn more about what the semantic digital twin is capable of refer to
its `documentation <https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/intro.html>`__.

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


---------------------------
Setting Up Your Environment
---------------------------

The first step is to set up the environment. This involves loading the robot, environment and object models and configuring
them as needed. For this guide we will use the PR2 in an apartment environment as well as a milk bottle.

Now to set up the environment, we will use URDF files for the robot and the environment and a STL file for the milk bottle.

.. code-block:: python

    from semantic_digital_twin.adapters.mesh import STLParser
    from semantic_digital_twin.adapters.urdf import URDFParser

    apartment_world = URDFParser.from_file("<path-to-coraplex>/resources/worlds/apartment.urdf").parse()
    milk_world = STLParser("<path-to-coraplex>/resources/objects/milk.stl").parse()
    pr2_world = URDFParser.from_file("package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro").parse()

.. note::
    The PR2 description is resolved from the ``iai_pr2_description`` ROS package rather than a file shipped with
    CoraPlex. CoraPlex itself only bundles ``resources/worlds`` and ``resources/objects``.

As you might have noticed each of the parsers returns an independent world which is not really useful. Therefore we need to
merge them into a single world.

.. code-block:: python

    from semantic_digital_twin.world import World

    world = World()

    world.merge(apartment_world)
    world.merge(milk_world)

To merge the PR2 robot with the world we need to specify the drive that the robot has since it is used to connect the root body
of the robot to the root of the other world. In this case the drive is an OmniDrive. Since we are modifiing the world we
need to use the modify_world context manager.

.. code-block:: python

    from semantic_digital_twin.world_description.connections import OmniDrive

    with world.modify_world():
        pr2_root = pr2_world.get_body_by_name("base_footprint")
        world_root = world.root
        drive_connection = OmniDrive.create_with_dofs(parent=world_root, child=pr2_root, world=world)
        world.merge_world(pr2_world, drive_connection)

Now we have a semantic digital twin that contains the apartment, the milk bottle and the PR2 robot. You can visualize it
using the VizMarkerPublisher, keep in mind that that uses the ROS visualization markers so you need to have ROS installed
and sourced.

.. code-block:: python

    from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
    import rclpy

    rclpy.init()
    node = rclpy.create_node("simple_viz_node")
    viz_publisher = VizMarkerPublisher(_world=world, node=node)

You can now open RViz2 and add a Marker display subscribing to the topic "/semworld/viz_marker" to see the world.

-----------------------
Writing your First Plan
-----------------------

Now that we have set up the environment, we can write our first plan. A plan in CoraPlex is a structured sequence of actions
which the robot will execute to achieve a specific goal. For more details on how plans work in CoraPlex, refer to the
:ref:`plan_header`.

The plan will consist of the following steps:

1. Park the robot's arms.
2. Raise the robot's torso.
3. Move the robot to a position near the milk bottle.
4. Pick up the milk bottle.
5. Move the robot to a position near the table.
6. Place the milk bottle on the table.

.. code-block:: python

    from semantic_digital_twin.robots.pr2 import PR2
    from semantic_digital_twin.spatial_types.spatial_types import Pose
    from semantic_digital_twin.datastructures.definitions import TorsoState
    from coraplex.datastructures.dataclasses import Context
    from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
    from coraplex.datastructures.grasp import GraspDescription
    from coraplex.plans.factories import sequential
    from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
    from coraplex.robot_plans.actions.core.navigation import NavigateAction
    from coraplex.robot_plans.actions.core.pick_up import PickUpAction
    from coraplex.robot_plans.actions.core.placing import PlaceAction

    context = Context(world, PR2.from_world(world))
    milk_body = world.get_body_by_name("milk.stl")

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            NavigateAction(Pose.from_xyz_rpy(2.0, 2.0, 0.0, reference_frame=world.root)),
            PickUpAction(
                object_designator=milk_body,
                arm=Arms.RIGHT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    context.robot.right_arm.end_effector,
                ),
            ),
            NavigateAction(Pose.from_xyz_rpy(4.0, 4.0, 0.0, reference_frame=world.root)),
            PlaceAction(
                object_designator=milk_body,
                target_location=Pose.from_xyz_rpy(4.2, 4.0, 1.0, reference_frame=world.root),
                arm=Arms.RIGHT,
            ),
        ],
        context=context,
    ).plan

What did we just do here?
We first created a context which holds the world as well as the semantic description of the PR2 robot in that world.
This context is used by the plan to determine in which world and by which robot the plan should be executed.
Next, we retrieved the milk bottle body from the world to use it in the pick-up and place actions.
Finally, we built a plan from the :func:`~coraplex.plans.factories.sequential` factory, meaning all actions will be
executed one after another in the order they are defined. The factory returns a node whose ``.plan`` attribute is the
runnable plan.

To execute the plan, we need to determine if it should be run in simulation or on a real robot and then call perform.

.. code-block:: python

    from coraplex.execution_environment import simulated_robot

    with simulated_robot:
        plan.perform()

Congratulations!🎉 You have just written and executed your first plan in CoraPlex.