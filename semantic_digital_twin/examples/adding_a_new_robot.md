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
# Adding a New Robot to the Semantic Digital Twin

## Step 0 - Verification of the Data You want to Integrate

> Safe yourself and your colleges time and effort and:
> 1. Verify assumptions we have about the URDF
> 	1. Please make sure the robot has a properly defined tool center point or tool frame at each of the manipulators
> 	2. Velocity limits should always be positive if set
> 	3. Any filepaths in the urdf should be defined with a `package://your_ros2_package/` prefix, so that they may be discovered by ros
> 2. Make sure the URDF and Meshes are packaged correctly into a ROS2 Package
> 	1. This includes a valid CMakeLists.txt and package.xml
>   2. You can verify this by building your ROS2 workspace with `colcon build` and sourcing the workspace
> 
> => If this is not the case, and you are in contact with the author, please consider re-requesting properly packaged data!

## Step 1 - Clone the iai_robots Repository on the Ros-jazzy Branch

```bash
git clone https://github.com/code-iai/iai_robots -b ros-jazzy
```

## Step 2 - Edit the ROS2 Package to Our Naming Schema, and Add it to the iai_robots Repo

1. Name the package `iai_{name_of_the_original_lab}_{robot_name}`.
	1. For example for the Armar7 robot of the Karlsruhe Institute for Technology (KIT) it would be `iai_kit_armar7`
2. This name also needs to be adjusted in the CMakeLists.txt, package.xml, and all the filepaths in the URDF.
	1. To safe yourself some time here, use an editor that supports the `Find & Replace` operation, and swap out the old ros2 prefix for the new one
		1. For example: swap `package://armar7/` with `package://iai_kit_armar7/`

## Step 3 - Integration into CRAM

> Disclaimer: This section will be reworked in the future when the Robot Semantic Annotation implementation has been refactored, because then @LucaKro will write a tool to create Robot Semantic Annotations easier.

1. Create a new class inheriting from AbstractRobot, and the mixins that describe your robot the best. You can find all mixins at [here](https://github.com/cram2/cognitive_robot_abstract_machine/blob/main/semantic_digital_twin/src/semantic_digital_twin/robots/robot_mixins.py)
2. Implement the abstract methods of AbstractRobot according to your robots specifications. For an example check out the implementation of the [PR2](https://github.com/cram2/cognitive_robot_abstract_machine/blob/main/semantic_digital_twin/src/semantic_digital_twin/robots/pr2.py)
	1. To get some better insight into the structure of your URDF, and to check if you have done Step 2 correctly, [edit this script and execute it](https://github.com/cram2/cognitive_robot_abstract_machine/blob/main/semantic_digital_twin/scripts/new_robot_integration_helper.py)
	2. This script will Publish your Robot to RVIZ, and create an image of the kinematic structure of your robot. This can be really helpful for figuring out what bodies are part of your robots gripper for example, without needing to search through the whole URDF.

Below you can find a starting point for your class:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_mixins import HasNeck, SpecifiesLeftRightArm, AbstractRobot
from semantic_digital_twin.world import World


@dataclass(eq=False)
class MyNewRobot(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """  
	A nice docstring for your new robot. In this case it has a neck and exactly 2 arms, which can be identified as "Left arm" and "Right arm"
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="your_robots_name", prefix=world.name),
            root=world.get_body_by_name("the_root_of_your_robot"),
            _world=world,
        )

    def _setup_collision_rules(self):
        """
        In this method you load the robot srdf you generated with the script at `giskardpy/scripts/ros2-tools/collision_matrix_tool.py`, 
        and then set up collision rules.
        """

    def _setup_semantic_annotations(self):
        """
        Here you annotate the the bodies of your robot with different semantic annotations. Usually you work from the leaves to the
        root. For example first you define the fingers, then create the gripper with the fingers, and then create the arm with the gripper
        """

    def _setup_joint_states(self):
        """
        These joint states are usually used within pycram plans to put the robot into a specific, predefined state.
        """

    def _setup_hardware_interfaces(self):
        """
        Here you specifiy which connections can be communicated with using a hardware interface on the real robot.
        """
```
