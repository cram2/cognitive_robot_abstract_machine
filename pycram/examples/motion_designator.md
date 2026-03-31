from pycram.plans.factories import execute_singlefrom pycram.plans.factories import execute_singlefrom pycram.plans.factories import execute_singlefrom pycram.plans.factories import execute_singlefrom pycram.language import SequentialPlan---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Motion Designator

Motion designators are similar to action designators, but unlike action designators, motion designators represent atomic
low-level motions. Motion designators only take the parameter that they should execute and not a list of possible
parameters, like the other designators. Like action designators, motion designators can be performed. Performing a motion
designator verifies the parameter and passes the designator to the respective process module.

Since motion designators perform a motion on the robot, we need a robot which we can use. Therefore, we will create a
BulletWorld as well as a PR2 robot.

```python
from pycram.testing import setup_world
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.robots.pr2 import PR2


world = setup_world()
pr2_view = PR2.from_world(world)

context = Context(world, pr2_view)
```

## Move

Move is used to let the robot drive to the given target pose. Motion designator are used in the same way as the other
designator, first create a description then resolve it to the actual designator and lastly, perform the resolved
designator.

```python
from pycram.robot_plans.motions import MoveMotion
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import *
from semantic_digital_twin.spatial_types.spatial_types import Pose
motion_description = MoveMotion(target=Pose.from_xyz_quaternion(pos_x=1., reference_frame=world.root))

with simulated_robot:
    execute_single(motion_description, context=context).perform()
```

## MoveTCP

MoveTCP is used to move the tool center point (TCP) of the given arm to the target position specified by the parameter.
Like any designator we start by creating a description and then resolving and performing it.

```python
from pycram.robot_plans.motions.gripper import MoveToolCenterPointMotion
from pycram.motion_executor import simulated_robot
from pycram.datastructures.enums import Arms
motion_description = MoveToolCenterPointMotion(target=Pose.from_xyz_quaternion(0.5, 0.6, 0.6, 0, 0, 0, 1, reference_frame=world.root), arm=Arms.LEFT)

with simulated_robot:
    execute_single(motion_description, context=context).perform()
```

## Looking

Looking motion designator adjusts the robot state such that the cameras point towards the target pose. Although this
motion designator takes the target as position and orientation, in reality only the position is used.

```python
from pycram.robot_plans.motions import LookingMotion
from pycram.motion_executor import simulated_robot

motion_description = LookingMotion(target=Pose.from_xyz_quaternion(1, 1, 1, 0, 0, 0, 1, reference_frame=world.root), camera=pr2_view.get_default_camera())

with simulated_robot:
    execute_single(motion_description, context=context).perform()
```

## Move Gripper

Move gripper moves the gripper of an arm to one of two states. The states can be {attr}`~pycram.datastructures.enums.GripperState.OPEN`  and {attr}`~pycram.datastructures.enums.GripperState.CLOSE`, which open
and close the gripper respectively.

```python
from pycram.robot_plans.motions import MoveGripperMotion
from pycram.motion_executor import simulated_robot
from pycram.datastructures.enums import Arms
from semantic_digital_twin.datastructures.definitions import GripperState

motion_description = MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT)

with simulated_robot:
    execute_single(motion_description, context=context).perform()
```

## Detecting

This is the motion designator implementation of detecting, if an object with the given object type is in the field of
view (FOV) this motion designator will return a list of  object designators describing the objects. It is important to specify the 
technique and state of the detection. You can also optional specify a region in which the object should be detected.


Since we need an object that we can detect, we will spawn a milk for this.

```python
# from pycram.robot_plans.motions import DetectingMotion, LookingMotion
# from pycram.process_module import simulated_robot
# from pycram.datastructures.pose import PoseStamped
# from pycram.datastructures.enums import DetectionTechnique, DetectionState
# from pycram.designators.object_designator import BelieveObject
# 
# with simulated_robot:
#     LookingMotion(target=PoseStamped.from_list([1.5, 0, 1], [0, 0, 0, 1])).perform()
# 
#     motion_description = DetectingMotion(technique=DetectionTechnique.TYPES,
#                                          state=DetectionState.START,
#                                          object_designator_description=BelieveObject(types=[Milk]).resolve(),
#                                          region=None)
# 
#     obj = motion_description.perform()
# 
#     print(obj[0])
```


## Move Joints

Move joints can move any number of joints of the robot, the designator takes two lists as parameter. The first list are
the names of all joints that should be moved and the second list are the positions to which the joints should be moved.

```python
from pycram.robot_plans.motions import MoveJointsMotion
from pycram.motion_executor import simulated_robot

with simulated_robot:
    motion_description = MoveJointsMotion(names=["torso_lift_joint", "r_shoulder_pan_joint"], positions=[0.2, -1.2])

    execute_single(motion_description, context=context).perform()
```