from ..conf.world.base_config import WorldConf, HandleConf, ContainerConf, FixedConnectionConf, \
    PrismaticConnectionConf, RevoluteConnectionConf, DrawerConf, CabinetConf
from ...dataset.semantic_world_like_classes import World, Handle, Container, FixedConnection, PrismaticConnection, RevoluteConnection, \
    Body, Drawer, Cabinet

last_world_id = -1


def create_world(world_conf: WorldConf) -> World:
    global last_world_id
    world = World(last_world_id + 1)
    last_world_id = world.id

    for body in world_conf.bodies:
        if isinstance(body, HandleConf):
            world.bodies.append(Handle(body.name, size=body.size, world=world))
        elif isinstance(body, ContainerConf):
            world.bodies.append(Container(body.name, size=body.size, world=world))
        else:
            world.bodies.append(Body(body.name, size=body.size, world=world))
    for connection in world_conf.connections:
        parent = next((b for b in world.bodies if b.name == connection.parent.name), None)
        child = next((b for b in world.bodies if b.name == connection.child.name), None)
        if parent and child:
            if isinstance(connection, FixedConnectionConf):
                connection_cls = FixedConnection
            elif isinstance(connection, PrismaticConnectionConf):
                connection_cls = PrismaticConnection
            elif isinstance(connection, RevoluteConnectionConf):
                connection_cls = RevoluteConnection
            else:
                raise ValueError(f"Unknown connection type: {connection}")
            world.connections.append(connection_cls(parent=parent, child=child, world=world))

    for view in world_conf.views:
        if isinstance(view, DrawerConf):
            # Materialize DrawerConf into a runtime Drawer entity with real Handle/Container from world.bodies
            handle = next((b for b in world.bodies if isinstance(b, Handle) and b.name == view.handle.name), None)
            container = next((b for b in world.bodies if isinstance(b, Container) and b.name == view.container.name), None)
            if handle is None or container is None:
                raise ValueError(f"Handle or Container not found for DrawerConf: {view}")
            world.views.append(Drawer(handle=handle, container=container, world=world))
        elif isinstance(view, CabinetConf):
            # Materialize CabinetConf and its drawers into runtime entities
            cab_container = next((b for b in world.bodies if isinstance(b, Container) and b.name == view.container.name), None)
            if cab_container is None:
                raise ValueError(f"Container not found for CabinetConf: {view}")
            drawers = []
            for dconf in view.drawers:
                handle = next((b for b in world.bodies if isinstance(b, Handle) and b.name == dconf.handle.name), None)
                cont = next((b for b in world.bodies if isinstance(b, Container) and b.name == dconf.container.name), None)
                if handle is None or cont is None:
                    raise ValueError(f"Handle or Container not found for Drawer in CabinetConf: {dconf}")
                drawers.append(Drawer(handle=handle, container=cont, world=world))
            world.views.append(Cabinet(container=cab_container, drawers=drawers, world=world))
        else:
            raise ValueError(f"Unknown view type: {view}")

    return world
