import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
)


@pytest.fixture()
def mini_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("root"))
        body2 = Body(name=PrefixedName("tip"))
        connection = RevoluteConnection.create_with_dofs(
            world=world, parent=body, child=body2, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world
