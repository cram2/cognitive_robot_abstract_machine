"""
Native connection replacement updates the inspected transform graph structure.
"""

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.transforms import ConnectionKind, TransformGraph

from .test_live_transforms import SHELF_NAME, make_kitchen


# %% same-named native replacements
@pytest.mark.parametrize("change", ["parent", "child", "kind"])
def test_replacement_updates_same_named_connection_structure(change: str) -> None:
    """
    An old connection's name cannot pin its replacement to old graph endpoints.

    :param change: Native structural property changed while retaining the name.
    """
    world, hinge = make_kitchen()
    previous = world.get_connection_by_name(SHELF_NAME)
    graph = TransformGraph()
    before = graph.observe(world.connections, world, 100.0)
    with world.modify_world():
        world.remove_connection(previous)
        parent = hinge.child if change == "parent" else previous.parent
        child = previous.child
        if change == "child":
            world.remove_kinematic_structure_entity(child)
            child = Body(name=PrefixedName("replacement_shelf", prefix="kitchen"))
        if change == "kind":
            replacement = RevoluteConnection.create_with_dofs(
                world, parent, child, name=previous.name, axis=Vector3.Z()
            )
        else:
            replacement = FixedConnection(
                parent=parent, child=child, name=previous.name
            )
        world.add_connection(replacement)
    after = graph.observe(world.connections, world, 101.0)
    activity = next(
        entry for entry in after.activities if entry.name == str(replacement.name)
    )
    assert (activity.parent, activity.child, activity.kind) == (
        str(replacement.parent.name),
        str(replacement.child.name),
        ConnectionKind.of_connection(replacement),
    )
    assert after.signature != before.signature
