# Test Driven RDR Fitting Tutorial

In this tutorial, we will walk through the process of fitting a Ripple Down Rules (RDR) model to a case object using a test-driven approach.
Similar to the [RDR Fitting Tutorial](rdr_fitting_tutorial.md), but with a focus on writing tests first and then
implementing the rules, this will enable maintenance of the rules and ensure that they work as expected with changes over time,
and it will enable collaboration with other developers who can write tests for their rules and merge them into the main rule tree.

<iframe width="560" height="315" src="https://www.youtube.com/embed/g5lpQIHYIG0" frameborder="0" allowfullscreen></iframe>

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

### Create your Case Object (Object to be Queried) In a Factory Method:
In a new python test script, create instances of the `Part` and `Robot` classes to represent your robot and its parts.
This will be the object that you will query with Ripple Down Rules. Put this inside a factory method such that
the case can be recreated easily and consistently over the lifetime of the project.

```python
from krrood.ripple_down_rules import CaseQuery

def robot_factory() -> Robot:
    """
    Factory method to create a robot with parts.
    """
    part_a = Part(name="A")
    part_b = Part(name="B")
    part_c = Part(name="C")
    
    robot = Robot("pr2", parts=[part_a])
    
    # Establish containment relationships
    part_a.contained_objects = [part_b]
    part_b.contained_objects = [part_c]
    
    return robot
```

### Put The RDR model in a pytest fixture (or a normal method):
Here since this RDR model will most likely be used in multiple tests, we put it in a pytest fixture.

```python
import pytest
from krrood.ripple_down_rules import GeneralRDR


@pytest.fixture
def robot_rdr():
    """
    Fixture to create a GeneralRDR instance for testing.
    """
    return GeneralRDR(save_dir='./', model_name='robot_rdr')
```

### Write a Test for fitting contained_objects of the robot:
Here we write a test that will check if the `contained_objects` of the robot are correctly identified, and this will also
serve as the fitting process for the Ripple Down Rules model.

Notice that we added two extra inputs to the `CaseQuery`:
- `scenario`: This is a reference to the test function itself, which can be useful for debugging and understanding the
context of the case, and recreating the scenario when verifying the rules or comparing with other rules.
- `case_factory`: This is a reference to the factory method that creates the case object,
allowing the RDR model to recreate the case object when needed.

```python
def test_fit_robot_contained_objects(robot_rdr):
    """
    Test to fit the Ripple Down Rules model to the robot's contained objects.
    """
    robot = robot_factory()
    
    # Define the case query for contained objects
    case_query = CaseQuery(robot, "contained_objects", (PhysicalObject,), False,
                           scenario=test_fit_robot_contained_objects,
                           case_factory=robot_factory)
    
    # Fit the RDR model to the case query
    robot_rdr.fit_case(case_query, update_existing_rules=False)
    
    # Classify the object and verify the result
    result = robot_rdr.classify(robot)
    
    # Assert that the result contains part B as the only contained object
    assert result['contained_objects'] == {robot.parts[0].contained_objects[0]}
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

### Run the Test

You can run the test using pytest:
```bash
pytest -s test_rdr_fitting.py
```
This will execute the test, prompting you to fit the RDR model to the case object or if there is rules already fitted,
it will classify the object based on the existing rules.