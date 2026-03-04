# Relational Example With Decorator Tutorial

Similar to the [Relational Example Tutorial](relational_example_tutorial.md), but using the `RDRDecorator` to enable
Ripple Down Rules classification on a method of a class (or any method).

<iframe width="560" height="315" src="https://www.youtube.com/embed/iapEdQRZTKo" frameborder="0" allowfullscreen></iframe>

### Define your Data Model With RDRDecorator

Here we define a simple data model of a robot with parts both of which are physical objects and can contain other physical objects.
We also add an RDRDecorator to the Robot class on the get_contained_objects method to enable Ripple Down Rules classification on this method.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing_extensions import List
from krrood.ripple_down_rules.rdr_decorators import RDRDecorator


@dataclass(unsafe_hash=True)
class PhysicalObject:
    """
    A physical object is an object that can be contained in a container.
    """
    name: str
    contained_objects: List[PhysicalObject] = field(default_factory=list, hash=False)

@dataclass(unsafe_hash=True)
class Part(PhysicalObject):
    ...

@dataclass(unsafe_hash=True)
class Robot(PhysicalObject):
    parts: List[Part] = field(default_factory=list, hash=False)
    containment_rdr: RDRDecorator = RDRDecorator("./", (PhysicalObject,), False,
                                                 fit=False)

    @containment_rdr.decorator
    def get_contained_objects(self) -> List[PhysicalObject]:
        """
        Returns the contained objects of the robot.
        """
        ...
```

### Create your Case Object (Object to be Queried):
In a new python script, create instances of the `Part` and `Robot` classes to represent your robot and its parts.
This will be the object that you will query with Ripple Down Rules.
```python
part_a = Part(name="A")
part_b = Part(name="B")
part_c = Part(name="C")
robot = Robot("pr2", parts=[part_a])
part_a.contained_objects = [part_b]
part_b.contained_objects = [part_c]
```

### (Optional) Enable Ripple Down Rules GUI

If you want to use the GUI for Ripple Down Rules, ensure you have PyQt6 installed:
```bash
pip install pyqt6
sudo apt-get install libxcb-cursor-dev
```

Then, you can enable the GUI in your script as follows:
```python
# Optionally Enable GUI if available
from krrood.ripple_down_rules.helpers import enable_gui
enable_gui()
```

### Directly Use the decorated method to Fit/Classify the Object
```python
robot.containment_rdr.fit = True
robot.get_contained_objects()

robot.containment_rdr.fit = False
contained_objects = robot.get_contained_objects()
assert contained_objects == [part_b]
```

When prompted to write a rule, I press edit in GUI (or type %edit in the Ipython interface if not using GUI),
I wrote the following inside the template function that the Ripple Down Rules created for me, this function takes a
`case` object as input, in this exampke the case is the `Robot` instance:

```python
contained_objects = []
for part in self_.parts:
    contained_objects.extend(part.contained_objects)
return contained_objects
```

I press the "Load" button (%load in Ipython), this loads the function I just wrote such that I can test it inside the
Ipython interface.

If I like the result, I press the "Accept" button (return func_name(**case) in Ipython), this will save the rule
permanently.

And then when asked for conditions, I wrote the following inside the template function that the Ripple Down Rules
created:

```python
return len(case.parts) > 0
```

This means that the rule will only be applied if the robot has parts.

### Additional Tip for RDRDecorator Usage:
We can let the fit mode be `True`, but give the rdr a function that tells it when to prompt for an answer.
For example, we can ask for an answer only when the robot's name is "tracy", which will result in the rdr not asking 
for an answer because the robot name is "pr2" for the current case.
The {py:attr}`ripple_down_rules.rdr_decorators.RDRDecorator.ask_now` is a user provided callable function that outputs
a boolean indicating when to ask the expert for an answer. The input to the `ask_now` function is a dictionary with the
original function arguments, while arguments like `self` and `cls` are passed as a special key `'self_'` or `'cls_'`
respectively.
```python
robot.containment_rdr.fit = True
robot.containment_rdr.ask_now = lambda case: case['self_'].name == "tracy"
robot.get_contained_objects()
```