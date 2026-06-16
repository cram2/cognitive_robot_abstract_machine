---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(adding-robots)=
# Adding a New Robot

This tutorial shows how to add a new robot to `semantic_digital_twin` by writing a
semantic annotation file in `semantic_digital_twin/robots/`.
The system works for any robot morphology — wheeled manipulators, bipedal humanoids,
table-mounted dual-arm setups, and anything else. Datastructures we provide are
flexible enough to allow you to define your robot's actual structure, without being bound 
to a fixed template.

The high-level workflow is:
1. Load the URDF and inspect the kinematic tree.
2. Map bodies and joints in the tree to semantic concepts.
3. Decide where in the class hierarchy each concept belongs.
4. Implement the classes leaf-first, robot class last.
5. Instantiate and validate.

In the future, we will look to provide a GUI to help you with this process.

---

## Step 1: Load your URDF and inspect the kinematic structure

Load the robot's description file and call `visualize_world_structure` to get a
graph of the kinematic tree. Each node in the graph is a body (a URDF link), and each
edge is a connection (a URDF joint) labeled with its joint type.

```python
from semantic_digital_twin.adapters.urdf import URDFParser

# Replace this with the path to your robot's URDF or MJCF file.
parser = URDFParser.from_file("/path/to/your/robot.urdf")
world = parser.parse()
world.visualize_world_structure()
```

This graph is your primary reference for the rest of the tutorial.
Keep it open while you work and navigate from the root node downward.

---

## Step 2: Map the tree to semantic concepts

The framework provides a set of semantic concepts that you map onto the bodies and
chains you see in the graph. None of them are mandatory — use only the ones that exist
on your robot.

| Concept | What it represents | What you need from the URDF                        |
|---|---|----------------------------------------------------|
| :class:`~semantic_digital_twin.robots.robot_parts.MobileBase` | The part responsible for locomotion (wheels, legs, tracks) | A single root body. Should have collision geometry |
| :class:`~semantic_digital_twin.robots.robot_parts.Torso` | A kinematic chain that serves as a shared mounting point for arms, neck, or other sub-parts | Root body + tip body                               |
| :class:`~semantic_digital_twin.robots.robot_parts.Arm` | A kinematic chain ending at a manipulator | Root body + tip body                               |
| :class:`~semantic_digital_twin.robots.robot_parts.EndEffector` | The manipulator at the end of an arm | Root body + tool-frame body                        |
| :class:`~semantic_digital_twin.robots.robot_parts.Finger` | A kinematic sub-chain of an end-effector | Root body + tip body                               |
| :class:`~semantic_digital_twin.robots.robot_parts.Neck` | A kinematic chain that carries a sensor | Root body + tip body                               |
| :class:`~semantic_digital_twin.robots.robot_parts.Camera` | A sensor body | A single body + optical parameters                 |

:class:`~semantic_digital_twin.robots.robot_parts.MobileBase`, :class:`~semantic_digital_twin.robots.robot_parts.EndEffector`, and :class:`~semantic_digital_twin.robots.robot_parts.Camera` only require a single root body.
:class:`~semantic_digital_twin.robots.robot_parts.Torso`, :class:`~semantic_digital_twin.robots.robot_parts.Arm`, :class:`~semantic_digital_twin.robots.robot_parts.Neck`, and :class:`~semantic_digital_twin.robots.robot_parts.Finger` are :class:`~semantic_digital_twin.robots.robot_parts.KinematicChain` subclasses and need
both a **root** and a **tip** body — the framework computes all bodies and
connections between them automatically.

### Identify your robot root

The robot root is the single body from which the rest of the kinematic tree hangs.
It is the topmost node in the graph produced by `visualize_world_structure`.
Common names are `base_footprint`, `pelvis`, or `root_link`.

### Work downward from the root

Starting at the robot root, follow the tree downward and ask:
- Does this robot move through the world? → it has a :class:`~semantic_digital_twin.robots.robot_parts.MobileBase`.
- Is there a shared kinematic chain connecting the mobile base to the arms and neck? → it has a :class:`~semantic_digital_twin.robots.robot_parts.Torso`.
- Are there chains ending at a manipulator? → each is an :class:`~semantic_digital_twin.robots.robot_parts.Arm` with an :class:`~semantic_digital_twin.robots.robot_parts.EndEffector`.
- Are there dexterous sub-chains on the end-effector? → each is a :class:`~semantic_digital_twin.robots.robot_parts.Finger`.
- Is there a chain carrying a sensor? → it is a :class:`~semantic_digital_twin.robots.robot_parts.Neck` with a :class:`~semantic_digital_twin.robots.robot_parts.Camera`.
- Is the camera mounted rigidly (no moving neck joints)? → the :class:`~semantic_digital_twin.robots.robot_parts.Camera` hangs
  directly off whatever part holds it, without a :class:`~semantic_digital_twin.robots.robot_parts.Neck`.

---

## Step 3: Decide where in the hierarchy each mixin belongs

The relationship "part A directly owns sub-part B" is expressed by adding a mixin to
A's class definition. The available mixins are:

| Mixin | Field it adds | Typical placement |
|---|---|---|
| :class:`HasMobileBase[T] <semantic_digital_twin.robots.robot_part_mixins.HasMobileBase>` | `mobile_base: T` | On the robot class, or skipped entirely |
| :class:`HasTorso[T] <semantic_digital_twin.robots.robot_part_mixins.HasTorso>` | `torso: T` | On :class:`~semantic_digital_twin.robots.robot_parts.MobileBase` if the base carries the torso; on the robot directly if there is no base |
| :class:`HasOneArm[T] <semantic_digital_twin.robots.robot_part_mixins.HasOneArm>` | `arm: T` | On :class:`~semantic_digital_twin.robots.robot_parts.Torso` or directly on the robot |
| :class:`HasLeftRightArm[L, R] <semantic_digital_twin.robots.robot_part_mixins.HasLeftRightArm>` | `arms: list`, `left_arm`, `right_arm` | On :class:`~semantic_digital_twin.robots.robot_parts.Torso` or directly on the robot |
| :class:`HasArms[...] <semantic_digital_twin.robots.robot_part_mixins.HasArms>` | `arms: list` | For three or more arms |
| :class:`HasNeck[T] <semantic_digital_twin.robots.robot_part_mixins.HasNeck>` | `neck: T` | On :class:`~semantic_digital_twin.robots.robot_parts.Torso` or directly on the robot |
| :class:`HasEndEffector[T] <semantic_digital_twin.robots.robot_part_mixins.HasEndEffector>` | `end_effector: T` | Already provided by :class:`Arm[T] <semantic_digital_twin.robots.robot_parts.Arm>` |
| :class:`HasTwoFingers[L, R] <semantic_digital_twin.robots.robot_part_mixins.HasTwoFingers>` | `fingers: list` | On :class:`~semantic_digital_twin.robots.robot_parts.EndEffector` |
| :class:`HasFingers[Thumb, ...] <semantic_digital_twin.robots.robot_part_mixins.HasFingers>` | `fingers: list` | On :class:`~semantic_digital_twin.robots.robot_parts.EndEffector` for three or more fingers |
| :class:`HasSensors[...] <semantic_digital_twin.robots.robot_part_mixins.HasSensors>` | `sensors: list` | On :class:`~semantic_digital_twin.robots.robot_parts.Neck`, or directly on the robot if there is no neck |

The rule is: **a mixin goes on the class that directly owns the sub-part**.
If the torso owns both arms, :class:`~semantic_digital_twin.robots.robot_part_mixins.HasLeftRightArm` belongs on the torso class.
If the arms hang directly off the robot root with no torso in between,
:class:`~semantic_digital_twin.robots.robot_part_mixins.HasLeftRightArm` belongs on the robot class itself — as in [Tracy](https://vib.ai.uni-bremen.de/page/comingsoon/the-tracebot-laboratory/),
where `class Tracy(AbstractRobot, HasLeftRightArm[TracyLeftArm, TracyRightArm], HasSensors[TracyCamera])`.

For bipedal humanoids such as iCub3 and UnitreeG1, :class:`~semantic_digital_twin.robots.robot_parts.MobileBase` (representing the
legs) and :class:`~semantic_digital_twin.robots.robot_parts.Torso` are both direct children of the robot class, because the legs and
upper body are independent kinematic sub-systems rather than a nested chain.

There is no single correct hierarchy; **the right structure is the one that faithfully
represents your robot**.

---

## Step 4: Implement the classes

Create a new module `semantic_digital_twin/robots/my_robot.py`.
Classes that are referenced as generic type parameters must be defined before the
classes that use them, so **write the file from leaves to root** even though you
designed it from root to leaves.

### The three abstract methods on every robot part

Generally, there are three abstract methods of interest:

**:meth:`setup_default_configuration_in_world_below_robot_root(cls, robot_root) <semantic_digital_twin.robots.robot_parts.AbstractRobotPart.setup_default_configuration_in_world_below_robot_root>`**

A classmethod that locates the relevant URDF links by name and constructs the part.
Use `robot_root._world.get_body_in_branch_by_name(robot_root, "link_name")` to look
up bodies by their URDF link name, searching only within the branch of the tree rooted
at `robot_root`.

**:meth:`setup_hardware_interfaces(self) <semantic_digital_twin.robots.robot_parts.AbstractRobotPart.setup_hardware_interfaces>`**

Marks actuated joints as having a hardware interface. Call
`self._world.get_connection_by_name("joint_name").has_hardware_interface = True`
for each controlled joint, or use the convenience method
:meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobotPart._setup_hardware_interfaces_for_active_connections` to mark every active
connection in the part's subtree at once. If the part has no hardware interface,
implement this as `None`.

**:meth:`setup_joint_states(self) -> List[JointState] <semantic_digital_twin.robots.robot_parts.AbstractRobotPart.setup_joint_states>`**

Returns a list of named :class:`~semantic_digital_twin.datastructures.joint_state.JointState` objects representing notable configurations
such as park poses or open/close states. Return `[]` if the part has no
named states.

### Leaf parts: Finger and Camera

Fingers and cameras usually have no sub-parts, so you will probably start with them.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Self, List

from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import Finger
from semantic_digital_twin.world_description.world_entity import KinematicStructureEntity


# %% Fingers 
# A Finger is a KinematicChain: it needs a root body and a tip body.
# Replace the link names with the actual names from your URDF.

@dataclass(eq=False)
class MyRobotLeftFinger(Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_finger_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_finger_tip_link"
            ),
        )

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class MyRobotRightFinger(Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_finger_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_finger_tip_link"
            ),
        )

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []

```

A Camera only needs a root body plus optical parameters.
forward_facing_axis: the axis of the camera frame that points forward into the scene.
field_of_view: horizontal and vertical opening angles in radians.
default_camera=True marks this as the camera returned by robot.get_default_camera().
Exactly one camera across the entire robot must have default_camera=True.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Self, List

from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.world_entity import KinematicStructureEntity

@dataclass(eq=False)
class MyRobotCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "camera_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            minimal_height=0.8,
            maximal_height=1.5,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []
```

### End-effector

An :class:`~semantic_digital_twin.robots.robot_parts.EndEffector` needs a root body, a tool-frame body (the point the robot aligns
with objects), and a `front_facing_orientation` quaternion that describes the
forward-facing direction of the tool frame.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Self, List

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import KinematicStructureEntity


# HasTwoFingers[Left, Right] adds a `fingers` list and auto-initialises both fingers.
# Use HasFingers[Thumb, ...] for three or more fingers.

@dataclass(eq=False)
class MyRobotGripper(
    EndEffector,
    HasTwoFingers[MyRobotLeftFinger, MyRobotRightFinger],
):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "palm_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "tool_center_point"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )

    def setup_hardware_interfaces(self):
        # _setup_hardware_interfaces_for_active_connections marks every active
        # connection in the gripper's subtree (including finger joints).
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(joints, [0.0, 0.0])),
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(joints, [0.8, 0.8])),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]
```

### Arm and Neck

Both are :class:`~semantic_digital_twin.robots.robot_parts.KinematicChain` subclasses: they need a root and a tip.
:class:`Arm[T] <semantic_digital_twin.robots.robot_parts.Arm>` already inherits :class:`HasEndEffector[T] <semantic_digital_twin.robots.robot_part_mixins.HasEndEffector>`, while :class:`Neck[T] <semantic_digital_twin.robots.robot_parts.Neck>`
inherits from :class:`HasSensors[T] <semantic_digital_twin.robots.robot_part_mixins.HasSensors>`, so don't inherit from those mixins seperately here.

Multiple chains can share a root body — this is normal for humanoids where both
arms and the neck all originate from the same shoulder or chest link.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Self, List

from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Arm, Neck
from semantic_digital_twin.world_description.world_entity import KinematicStructureEntity


@dataclass(eq=False)
class MyRobotArm(Arm[MyRobotGripper]):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "shoulder_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "wrist_link"
            ),
        )

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]


@dataclass(eq=False)
class MyRobotNeck(Neck[MyRobotCamera]):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "neck_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_link"
            ),
        )

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []
```

### Torso and MobileBase

The torso is a kinematic chain that owns both the arm and the neck in this example.
The mobile base owns the torso. If your robot has no torso (for example, the arm is
mounted directly on the base), skip the :class:`~semantic_digital_twin.robots.robot_parts.Torso` class and put :class:`~semantic_digital_twin.robots.robot_part_mixins.HasOneArm` directly
on :class:`~semantic_digital_twin.robots.robot_parts.MobileBase` or the robot class.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Self, List

from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm, HasNeck, HasTorso
from semantic_digital_twin.robots.robot_parts import MobileBase, Torso
from semantic_digital_twin.world_description.world_entity import KinematicStructureEntity


# Torso: HasOneArm for a single arm, HasLeftRightArm[L, R] for two.
# HasNeck is optional — omit it if the camera is fixed.

@dataclass(eq=False)
class MyRobotTorso(Torso, HasOneArm[MyRobotArm], HasNeck[MyRobotNeck]):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        joints = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(joints, [0.0])),
            state_type=TorsoState.LOW,
        )
        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(joints, [0.3])),
            state_type=TorsoState.HIGH,
        )
        return [torso_low, torso_high]


# MobileBase: only needs a root body. HasTorso adds the torso as a child.
# If the robot does not move, omit MobileBase entirely.

@dataclass(eq=False)
class MyRobotMobileBase(MobileBase, HasTorso[MyRobotTorso]):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "base_link"
            ),
        )

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []
```

---

## Step 5: The robot class

The robot class ties everything together.
It must also implement three abstract methods.

**:meth:`get_ros_file_path(cls) -> str <semantic_digital_twin.robots.robot_parts.AbstractRobot.get_ros_file_path>`**

Returns the ROS package path to the robot's URDF or xacro file.
Our parsers can automatically resolve ROS package paths.

**:meth:`_get_root_body_name(cls) -> str <semantic_digital_twin.robots.robot_parts.AbstractRobot._get_root_body_name>`**

Returns the URDF link name of the robot root — the topmost node in your
`visualize_world_structure` graph.

**:meth:`_setup_collision_rules(self) <semantic_digital_twin.robots.robot_parts.AbstractRobot._setup_collision_rules>`**

Configures collision checking for the robot.
Load an SRDF file with :meth:`SelfCollisionMatrixRule.from_collision_srdf(srdf_path, world) <semantic_digital_twin.collision_checking.collision_rules.SelfCollisionMatrixRule.from_collision_srdf>`
to define self-collision pairs to ignore, and extend the default rules with
:class:`~semantic_digital_twin.collision_checking.collision_rules.AvoidExternalCollisions` and :class:`~semantic_digital_twin.collision_checking.collision_rules.AvoidSelfCollisions`.
If you do not yet have a collision configuration, implement this as `None`.

The optional :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot._setup_velocity_limits` override lets you tighten joint velocity limits
after the URDF values have been loaded.
:meth:`tighten_dof_velocity_limits_proportionally(maximum_velocity) <semantic_digital_twin.robots.robot_parts.AbstractRobot.tighten_dof_velocity_limits_proportionally>` scales all limits so
that the fastest joint is capped at the given value.
:meth:`tighten_dof_velocity_limits_of_1dof_connections(new_limits) <semantic_digital_twin.robots.robot_parts.AbstractRobot.tighten_dof_velocity_limits_of_1dof_connections>` lets you set limits
per joint individually.

```python
from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot


@dataclass(eq=False)
class MyRobot(AbstractRobot, HasMobileBase[MyRobotMobileBase]):
    """
    Replace this docstring with a short description of your robot.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://my_robot_description/urdf/my_robot.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def _setup_collision_rules(self):
        # Minimal implementation. Add SelfCollisionMatrixRule and
        # AvoidExternalCollisions / AvoidSelfCollisions rules here when ready.
        return None

    def _setup_velocity_limits(self):
        # Optional. tighten_dof_velocity_limits_proportionally scales all joint
        # velocity limits so the fastest joint is capped at the given value.
        self.tighten_dof_velocity_limits_proportionally(maximum_velocity=1.0)
```

---

## Step 6: Instantiate and validate

```python
from semantic_digital_twin.adapters.urdf import URDFParser

# Re-parse your URDF to get a fresh world, then annotate it with your robot class.
parser = URDFParser.from_file("/path/to/your/robot.urdf")
world = parser.parse()

robot = MyRobot.from_world(world)

# from_world calls setup_robot_part_semantic_annotations internally, which walks
# the generic type parameters you declared (e.g. HasMobileBase[MyRobotMobileBase])
# and calls setup_default_configuration_in_world_below_robot_root for each part
# automatically. You do not need to instantiate the parts manually.

# validate() checks that all parts are correctly initialised, that backreferences
# are consistent, and that exactly one camera has default_camera=True.
robot.validate()
```

If :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.validate` returns `True`, the robot is ready to use.

---

## Common pitfalls

**`@dataclass(eq=False)` is required on every class.**
:class:`~semantic_digital_twin.world_description.world_entity.SemanticAnnotation` defines a custom hash function. Without `eq=False`, Python
replaces it with the dataclass-generated one, which breaks identity tracking inside
the world.

**`default_camera=True` must be set on exactly one camera.**
:meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.get_default_camera` raises :class:`~semantic_digital_twin.exceptions.MissingDefaultCameraError` if no camera has the flag,
and returns the first match if several do. Check each :class:`~semantic_digital_twin.robots.robot_parts.Camera` subclass.

**Multiple kinematic chains may share a root body.**
A :class:`~semantic_digital_twin.robots.robot_parts.Torso` whose tip is `shoulder_link` and an :class:`~semantic_digital_twin.robots.robot_parts.Arm` whose root is also `shoulder_link`
is valid. Similarly, two arms on a humanoid may both declare `pelvis` as their root
while the torso also starts there. This is intentional — chains are defined by their
start and end, not by exclusive body ownership.

**File order: leaves before parents.**
Because Python evaluates class bodies at import time, any class used as a generic
type parameter (e.g. `Arm[MyRobotGripper]`) must already be defined earlier in the
file. Design top-down, write bottom-up.

**`None` and `return []` are always valid.**
If a part has no hardware interfaces or no named joint states, the abstract methods
still need to be present — `None` and `return []` are the correct implementations,
not omissions.

**:meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobotPart._setup_hardware_interfaces_for_active_connections` marks the entire subtree.**
This convenience method iterates over all robot parts nested below the calling part
and marks every :class:`~semantic_digital_twin.world_description.connections.ActiveConnection` as having a hardware interface. Use it when all
joints in the subtree are actuated. If only specific joints are controlled, mark them
individually with `self._world.get_connection_by_name("joint_name").has_hardware_interface = True`.
