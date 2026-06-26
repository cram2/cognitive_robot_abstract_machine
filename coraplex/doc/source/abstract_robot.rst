Abstract Robot Overview
=======================
To define and manage semantic information about robots CoraPlex uses the ``AbstractRobot`` class of the semantic
digital twin. Specific instances of the ``AbstractRobot`` class are part of the Context that is passed to the Plan on
creation.

The ``AbstractRobot`` class defines a semantic, high-level model of a robot as it appears in a world description. Rather
than focusing on actuation details or low-level control, it organizes the robot's physical and functional structure into
coherent parts such as kinematic chains, end effectors, sensors, and the torso. This abstraction lets downstream
components reason about what the robot is, what it has, and how its parts relate, without needing to know how individual
joints or links are implemented.

The Problem It Solves
---------------------
Robot software often mixes structural knowledge (which bodies and joints form an arm, where the gripper's tool frame is,
what cameras are available) with control or task logic. That coupling complicates reuse and makes it hard to write generic
algorithms that operate across different platforms. ``AbstractRobot`` solves this by providing a uniform, semantic view of
a robot that is reconstructible from a ``World`` description. The result is a clean separation: semantic structure and
capabilities are captured once, while planners, controllers, and task logic can query that structure in a device-agnostic way.

Core Concepts and Terminology
-----------------------------
A robot is a semantic annotation with a ``root`` body typically representing its base. From this root, the robot is
composed of structured semantic annotations of its parts (all defined in
``semantic_digital_twin.robots.robot_parts``):

* A ``KinematicChain`` is a contiguous sequence of kinematic structure entities from a ``root`` body to a ``tip`` body.
* An ``EndEffector`` is the abstract base for end effectors and always defines a ``tool_frame`` as well as a
  ``front_facing_orientation`` and the derived ``front_facing_axis`` used for tasks such as approach planning. Concrete
  grippers such as ``PR2RightGripper`` and ``HSRBGripper`` extend ``EndEffector`` (together with the ``HasTwoFingers``
  mixin and one or more ``Finger`` parts).
* A ``Sensor`` is any perceptual device; ``Camera`` is a concrete sensor that adds a forward-facing axis, a field of
  view, and typical operating height bounds.
* A ``Torso`` is a special kinematic chain that connects the base to other chains, and a ``Neck`` is a specialized
  kinematic chain that connects the head and its sensors.

How Parts Fit Together
----------------------
Composition uses **specialized structures**: each robot defines only the parts it actually has, wired together with
mixins and generics, and the parts are reached as nested attributes rather than through ``add_*`` methods. For example,
the PR2 exposes its arms as ``pr2.mobile_base.torso.arms``, and an arm declares its gripper through a mixin such as
``HasEndEffector[PR2RightGripper]``. Sub-parts are instantiated automatically from the generic type hints, and the
framework handles the order in which semantic annotations are added to the world, so robot structures are valid by
construction. Calling ``validate()`` confirms that all fields are plausibly filled and that the robot can be
synchronized without issues.

To query a robot's parts regardless of its specific structure, ``AbstractRobot`` provides accessor methods:
``get_end_effectors()``, ``get_arms()``, ``get_sensors()``, ``get_torso()``, ``get_left_arm_if_specified()``,
``get_right_arm_if_specified()`` and ``get_default_camera()``.

Interaction with the World and Motion Control
---------------------------------------------
Because ``AbstractRobot`` is rooted in the world model, it can expose cross-cutting capabilities in a uniform way. The
``controlled_connections`` property returns the robot's connections that are controlled by a controller. The ``drive``
property returns the robot's ``WheeledDrive`` connection (``Optional[WheeledDrive]``) if the base is connected that way,
allowing higher-level code to discover how to command base motion without hard-coding link names.

For safety and performance tuning, the robot provides ``tighten_dof_velocity_limits_of_1dof_connections`` and
``tighten_dof_velocity_limits_proportionally`` to constrain the velocity limits of its one-degree-of-freedom active
connections.

Construction and Extensibility
------------------------------
Robots are usually created via the ``from_world`` class method, which constructs the semantic structure by looking up
bodies and connections in a ``World`` (``from_branch_in_world`` is available when several identical robots share one
world). To implement a new robot, follow three rules:

1. **Map concepts**: create a class for every distinct part of the robot defined in ``robot_parts.py``.
2. **Define the hierarchy**: use mixins and generics to declare direct parent-child relationships (for example
   ``PR2RightArm(HasEndEffector[PR2RightGripper])``).
3. **Implement the abstract methods**: most importantly ``get_ros_file_path`` and ``_get_root_body_name``.

Concrete Examples: PR2 and HSRB
-------------------------------
The ``PR2`` subclass illustrates a dual-arm mobile manipulator whose base ``root`` is ``base_footprint``. Each arm is a
kinematic chain carrying a two-fingered gripper, and the head is a ``Neck`` carrying a ``Camera`` with a field of view
and plausible operating heights. The ``HSRB`` subclass models a single-arm service robot whose arm chain carries both a
gripper and a hand-mounted ``Camera``, demonstrating how one chain can act as both a manipulator and a sensor chain. As
with PR2, the HSRB is built entirely by ``from_world`` using URDF-consistent link names so that it matches the parsed
world model.

Key takeaways
-------------
- ``AbstractRobot`` is a semantic, world-backed description of a robot's structure and capabilities.
- Parts are composed from ``KinematicChain``, ``EndEffector``, ``Sensor`` and ``Torso`` using specialized structures
  (mixins and generics), and reached as nested attributes.
- Accessor methods such as ``get_arms()`` and ``get_default_camera()`` query parts in a robot-agnostic way.
- The robot is reconstructed from a ``World`` via ``from_world``; new robots are added by subclassing and implementing
  ``get_ros_file_path`` and ``_get_root_body_name``.
