from ...datasets import World, Handle, Container, FixedConnection, PrismaticConnection
from ...conf.world.base_config import WorldConf, HandleConf, ContainerConf, FixedConnectionConf, PrismaticConnectionConf


def create_world_with_handles_and_containers(world_conf: WorldConf) -> World:
    world = World()

    for body in world_conf.bodies:
        if isinstance(body, HandleConf):
            world.bodies.append(Handle(body.name, world=world))
        elif isinstance(body, ContainerConf):
            world.bodies.append(Container(body.name, world=world))
    for connection in world_conf.connections:
        parent = next((b for b in world.bodies if b.name == connection.parent.name), None)
        child = next((b for b in world.bodies if b.name == connection.child.name), None)
        if parent and child:
            if isinstance(connection, FixedConnectionConf):
                world.connections.append(FixedConnection(parent=parent, child=child, world=world))
            elif isinstance(connection, PrismaticConnectionConf):
                world.connections.append(PrismaticConnection(parent=parent, child=child, world=world))

    return world
