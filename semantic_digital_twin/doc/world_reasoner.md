---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# World Reasoner

The world reasoner {py:class}`semantic_digital_twin.reasoning.world_reasoner.WorldReasoner` turns the bodies and
joints of a parsed world model into semantic annotations: the body a slider pulls out becomes a
{py:class}`semantic_digital_twin.semantic_annotations.semantic_annotations.Drawer`, the body it slides out of becomes a
{py:class}`semantic_digital_twin.semantic_annotations.semantic_annotations.Cabinet`, and so on.

It is a rule based classifier built on [Ripple Down Rules](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/ripple_down_rules/intro.html),
which live in this workspace as {py:mod}`krrood.ripple_down_rules`. The rules live in the semantic digital twin
package and are versioned and migrated with the world data structures they read, so they keep working as those data
structures change.

## Using the reasoner

Hand the reasoner a world and ask it what it can work out:

```{code-cell} ipython3
from os.path import join, dirname
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.adapters.urdf import URDFParser

kitchen_world = URDFParser.from_file(join(dirname(__file__), '..', 'resources', 'urdf', 'kitchen-small.urdf')).parse()
reasoner = WorldReasoner(kitchen_world)

# The annotations this run inferred.
new_semantic_annotations = reasoner.infer_semantic_annotations()
print(new_semantic_annotations)

# Every annotation the world now holds, including any it already had.
print(kitchen_world.semantic_annotations)
```

Inferring also wires up what belongs together: a drawer is given the handle mounted on it, a cabinet is given the
drawers and doors that open out of it, and every drawer and door is given the
{py:class}`semantic_digital_twin.semantic_annotations.semantic_annotations.Slider` or
{py:class}`semantic_digital_twin.semantic_annotations.semantic_annotations.Hinge` that already moves it.

{py:meth}`~semantic_digital_twin.reasoning.world_reasoner.WorldReasoner.reason` returns every world attribute the
reasoner has rules for, keyed by attribute name, and only re-runs when the world model has changed since the last call.

## How a body is recognised

Two things decide what a body is, in this order.

**The kinematic structure**, which settles everything that opens, never the name. The child of a prismatic
connection is a drawer and the child of a revolute connection is a door. A handle is a body of its own that nothing
hangs off, fixed to a part an active joint moves — so a drawer front's grip is a handle, while a tap's lever, which
swings on a joint of its own, is part of the mechanism rather than a handle on it. A container is whatever rigidly
mounted body has drawers or doors opening out of it.

Bodies carrying no collision geometry are skipped, because world models use them as shapeless helpers to build up a
compound motion: a door that pops out before it swings hangs off such a helper, and the rules look past it to the
container the door really belongs to. A joint whose multiplier is not `1` only repeats another joint's motion — a
URDF mimic — so the part it moves is a leaf of one mechanism rather than something that opens by itself. That is how
a front folding out of two leaves is recognised as one door opened by the single handle on its lower leaf, while a
dishwasher is not mistaken for its own door.

All of this is asked as a query over the joints, so a rule reads as the shape it is looking for:

```{code-cell} ipython3
from krrood.entity_query_language.factories import entity, inference, variable
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.connections import ActiveConnection, FixedConnection

# A handle is a body fixed to a part that an active joint moves.
mount = variable(FixedConnection, kitchen_world.connections)
joint = variable(ActiveConnection, kitchen_world.connections)
grip = mount.child
print(
    entity(inference(Handle)(root=grip))
    .where(joint.child == mount.parent, grip.has_collision())
    .tolist()
)
```

**The body's name**, but only for the things no joint gives away: an oven, a sink, a worktop, a wall, a sofa, a table.
Names are read through the annotation classes' own vocabulary — the words of the class name, plus any `_synonyms` the
class declares — so the vocabulary lives on the class where every world benefits from it, rather than as strings
spelled inside a rule. To teach the reasoner a new spelling for something, add it to that class's `_synonyms`.

Compound names mention more than one kind: `sink_area_left_upper_drawer` says *where* the body is before it says
*what* it is. Such a name is settled by the kind still being spoken about at its end, so that body is a drawer rather
than a sink. A name is also allowed to stop the structure from claiming a body — a shelf board mounted inside a
drawer looks exactly like a handle to the joints alone, and is kept apart by being called a board.

## Changing or adding a rule

The rules are ordinary, documented source in
{py:mod}`semantic_digital_twin.reasoning.world_rdr.rules` — one function per annotation type, each returning
every annotation of that type. To change what the reasoner infers, edit that function:

```{code-cell} ipython3
from krrood.entity_query_language.factories import entity, inference, variable
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from typing_extensions import List


def drawers(world: World) -> List[Drawer]:
    """
    Every body a slider pulls straight out of a container.
    """
    slider = variable(PrismaticConnection, world.connections)
    return (
        entity(inference(Drawer)(root=slider.child))
        .where(slider.child.has_collision())
        .tolist()
    )
```

Rules are written in the [Entity Query Language](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/eql/intro.html)
({py:mod}`krrood.entity_query_language`), which is what lets the reasoner explain itself afterwards (see below).
Say as much as possible as a query over the joints rather than as a Python predicate: a join is what the explanation
and the verbalization are able to read back.

Two things to keep in mind when writing one:

- **Read the annotations you depend on from the world you were handed**, as `world.semantic_annotations`. Rules run
  repeatedly until nothing new appears, so a rule can build on what earlier rules concluded, but those conclusions
  have not reached the world itself yet while the rules are still running.
- **Keep rules for the same type disjoint.** An annotation is identified by its type together with the bodies it
  refers to, so a drawer with a handle and the same drawer without one are two different annotations and the world
  would end up with both. That is why the drawer and door rules come in pairs, one for the parts that carry a handle
  and one for those that do not.

Beside it, `world_semantic_annotations_mcrdr_defs.py` holds the `conditions_`/`conclusion_` pairs the classifier
actually calls: each names the precondition of one rule and the rule that runs under it. A new rule needs a pair
added there, its type added to the `conclusion_type` of `world_semantic_annotations_mcrdr.py`, and — because the
classifier rebuilds each rule from that file's imports — the rule itself has to be importable, which is why the
rules sit in `rules.py` rather than inside the pairs.

```{note}
Rules used to be written by prompting for them in an interactive shell, through
{py:meth}`~semantic_digital_twin.reasoning.reasoner.CaseReasoner.fit_attribute`. That way of authoring them is no
longer used here: edit the rules in `world_rdr/rules.py` directly instead.
```

## Asking why

Because rules are queries rather than opaque code, the reasoner can say what made it reach a conclusion, and say it in
English:

```{code-cell} ipython3
from krrood.entity_query_language.explanation.explanation import explain_inference
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

drawer = next(a for a in new_semantic_annotations if isinstance(a, Drawer))
explanation = explain_inference(drawer)

print(explanation.get_satisfied_conditions_as_string())
print(verbalize_expression(explanation.query_root))
```
