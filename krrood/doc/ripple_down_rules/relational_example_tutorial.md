# Relational Example Tutorial

In this tutorial, we will walk through the process of fitting a Ripple Down Rules (RDR) model to a relational data model.
Where there are multiple objects that are related to each other, and we want to query them using Ripple Down Rules.

<iframe width="560" height="315" src="https://www.youtube.com/embed/Dgcj7Y7qNyI" frameborder="0" allowfullscreen></iframe>

### Define your Data Model

Here we define a simple data model of a robot with parts both of which are physical objects and can contain other physical objects.
Put this in a file called `relational_model.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing_extensions import List


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

### Define the RDR Model and the Case Query
Here create/load our RDR model, then we define a query on the `robot` object to find out which objects are contained within it.
The output type is specified as `PhysicalObject`, and there can be multiple contained objects so we set `mutually_exclusive` to `False`.

Optionally enable the GUI.

```python
from krrood.ripple_down_rules import CaseQuery, GeneralRDR

grdr = GeneralRDR(save_dir='./', model_name='part_containment_rdr')

case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), mutually_exclusive=False)
```

### Fit the Model to the Case Query by Answering the prompts.
```python
grdr.fit_case(case_query)
```

When prompted to write a rule, I press edit in GUI (or type %edit in the Ipython interface if not using GUI),
I wrote the following inside the template function that the Ripple Down Rules created for me, this function takes a
`case` object as input, in this exampke the case is the `Robot` instance:

```python
contained_objects = []
for part in case.parts:
    contained_objects.extend(part.contained_objects)
return contained_objects
```

I press the "Load" button (%load in Ipython), this loads the function I just wrote such that I can test it inside the
Ipython interface.

If I like the result, I press the "Accept" button (return func_name(case) in Ipython), this will save the rule
permanently.

And then when asked for conditions, I wrote the following inside the template function that the Ripple Down Rules
created:

```python
return len(case.parts) > 0
```

This means that the rule will only be applied if the robot has parts.

### Finally, Classify the Object and Verify the Result

```python
result = grdr.classify(robot)
assert result['contained_objects'] == {part_b}
```

If you notice, the result only contains part B, while one could say that part C is also contained in the robot, but,
the rule we wrote only returns the contained objects of the parts of the robot. To get part C, we would have to
add another rule that says that the contained objects of my contained objects are also contained in me, you can 
try that yourself and see if it works!