# World Reasoner

The world reasoner {py:class}`semantic_digital_twin.reasoner.WorldReasoner` is a class that uses [Ripple Down Rules](https://github.com/AbdelrhmanBassiouny/ripple_down_rules/tree/main)
to classify concepts and attributes of the world. This is done using a rule based classifier that benefits from incremental
rule addition through querying the system and answering the prompts that pop up using python code.

The benefit of that is the rules of the reasoner are based on the world datastructures and are updates as the datastructures
are updated. Thus, the rules become a part of the semantic digital twin repository and are update, migrated, and versioned with it.

## How to use:

There are two ways in which the reasoner can be used, classification mode, and fitting mode, both of which are explained
bellow.

### A: Classification Mode

In classification mode, the reasoner is used as is with it's latest knowledge or rule trees to classify concepts about the
world.

For example lets say the reasoner now has rules that enable it find specific types of semantic annotations like the Drawer and the Cabinet.
The way to use the reasoner is like the following example:

```{code-cell} ipython3
from os.path import join, dirname
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.adapters.urdf import URDFParser

kitchen_world = URDFParser.from_file(join(dirname(__file__), '..', 'resources', 'urdf', 'kitchen-small.urdf')).parse()
reasoner = WorldReasoner(kitchen_world)
found_concepts = reasoner.reason()

# 1st method, access the semantic annotations directly from the reasoning result
new_semantic_annotations = found_concepts['semantic_annotations']
print(new_semantic_annotations)

# Or 2nd method, access all the semantic annotations from the world.semantic_annotations, but this will include all semantic annotations not just the new ones.
all_semantic_annotations = kitchen_world.semantic_annotations
print(all_semantic_annotations)
```

Similarly, for any other world attribute that the reasoner can infer values for, just replace the 'semantic_annotations' with the 
appropriate attribute name.

### B: Fitting Mode

In fitting mode, the reasoner can be used to improve and enlarge it's rule tree or even to widen it's application to even
more attributes of the world.

For example, let's say you want to improve an existing rule that classifies Drawers, you can do that as follows:

```{code-cell} ipython3
from os.path import join, dirname
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer


def create_kitchen_world():
    return URDFParser.from_file(join(dirname(__file__), '..', 'resources', 'urdf', 'kitchen-small.urdf')).parse()


kitchen_world = create_kitchen_world()
reasoner = WorldReasoner(kitchen_world)

reasoner.fit_attribute("semantic_annotations", [Drawer], update_existing_semantic_annotations=True,
                       world_factory=create_kitchen_world)
```

Then you will be prompted to write a rule for Drawer, and you can see the currently detected drawers shown in the Ipyton
shell. Maybe you see a mistake and not all the currently detected drawers are actual drawers, so you want to filter the
results. To start writing your rule, just type `%edit` in the Ipython terminal as shown the image bellow, or if using
the GUI just press the `Edit` button.

```{figure} _static/images/write_edit_in_ipython.png
---
width: 800px
---
Open the Template File in Editor from the Ipython Shell.
```

Now, a template file with some imports and an empty function is openned for you to write your rule inside the body of
the function as shown bellow:

```{code-cell} ipython3
from dataclasses import dataclass, field
from posixpath import dirname
from typing_extensions import Any, Callable, ClassVar, Dict, List, Optional, Type, Union
from krrood.ripple_down_rules.rdr import GeneralRDR
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer


def world_semantic_annotations_of_type_drawer(case: World) -> List[Drawer]:
    """Get possible value(s) for World.semantic_annotations  of type Drawer."""
    # Write code here
    pass
```

You can write a filter on the current semantic annotations of type Drawer as follows:

```{code-cell} ipython3
from dataclasses import dataclass, field
from posixpath import dirname
from typing_extensions import Any, Callable, ClassVar, Dict, List, Optional, Type, Union
from krrood.ripple_down_rules.rdr import GeneralRDR
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer


def world_semantic_annotations_of_type_drawer(case: World) -> List[Drawer]:
    """Get possible value(s) for World.semantic_annotations  of type Drawer."""
    known_drawers = [v for v in case.semantic_annotations if isinstance(v, Drawer)]
    good_drawers = [d for d in known_drawers if d.name.name != "bad_drawer"]
    return good_drawers
```

So the above is the generated template, and I just filled in the body of the function with my rule logic. After that
you write `%load` in the Ipython and the function you just wrote will be available to you to test it out in the Ipython
shell as shown bellow (in the GUI just pres the Load button):

```{figure} _static/images/load_rule_and_test_it.png
---
width: 1600px
---
Load the written Rule into the Ipython Shell.
```

Then if you want to change the rule, just edit the already open template file and do `load` again. Once you are happy
with your rule results just return the function output as follows (in the GUI just press the Accept button):

```{figure} _static/images/accept_rule.png
---
width: 600px
---
Accept the Rule in Ipython.
```

If you also want to contribute to the semantic digital twin package, then it's better to do that in the `test_semantic_annotations/test_semantic_annotations.py`
test file. Since there is already rules for Drawer, there would already be a test method for that. All you need to do is
set the `update_existing_semantic_annotations` to `True` like this:

```{code-cell} ipython3
def test_drawer_semantic_annotation(self):
    self.fit_rules_for_a_semantic_annotation_in_apartment(Drawer, scenario=self.test_drawer_semantic_annotation, update_existing_semantic_annotations=True)
```
then run the test from the terminal using `pytest` as follows:
```bash
cd semantic_digital_twin/test/test_semantic_annotations && pytest -s -k "test_drawer_semantic_annotation"
```
Then answer the prompt with the rule as described before. Now the rules for the Drawer semantic annotation has been updated, Nice Work!

You could also create a new test method if your world is not the apartment or if you want to add a specific test for a
specific context, more tests are always welcome :D. Just make sure you set the scenario to be the new test method name,
and set the world factory to the method that creates your world (if it doesn't exist create one and put it in the test 
file).

In addition you can fit the reasoner on a totally new concept/attribute of the world instead of `semantic_annotations`, maybe `regions`
or `predicates` , ...etc. What's great is that inside your rules you can use the semantic annotations that were classified already by
the semantic annotations rules, and vice verse, you can add semantic annotations rules that use outputs from rules on other attributes as well.
