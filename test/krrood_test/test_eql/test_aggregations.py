from krrood.entity_query_language.entity import (
    flatten,
    entity,
    not_,
    in_,
    for_all,
    variable,
    exists,
)
from krrood.entity_query_language.entity_result_processors import an, the
from ..dataset.example_classes import VectorsWithProperty
from ..dataset.semantic_world_like_classes import View, Drawer, Container, Cabinet


# Make a simple View-like container with an iterable attribute `drawers` to be flattened
class CabinetLike(View):
    def __init__(self, drawers, world):
        super().__init__(world=world)
        self.drawers = list(drawers)


def test_flatten_iterable_attribute(handles_and_containers_world):
    world = handles_and_containers_world

    views = variable(Cabinet, world.views)
    drawers = flatten(views.drawers)
    query = an(entity(drawers))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 3
    assert {row.handle.name for row in results} == {"Handle1", "Handle2", "Handle3"}


def test_flatten_iterable_attribute_and_use_not_equal(handles_and_containers_world):
    world = handles_and_containers_world

    cabinets = variable(Cabinet, world.views)
    drawer_1_var = variable(Drawer, world.views)
    drawer_1 = an(entity(drawer_1_var).where(drawer_1_var.handle.name == "Handle1"))
    drawers = flatten(cabinets.drawers)
    query = an(entity(drawers).where(drawer_1 != drawers))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 2
    assert {row.handle.name for row in results} == {"Handle2", "Handle3"}


def test_exists_and_for_all(handles_and_containers_world):
    world = handles_and_containers_world

    cabinets = variable(Cabinet, world.views)
    drawer_var = variable(Drawer, world.views)
    my_drawers = an(entity(drawer_var).where(drawer_var.handle.name == "Handle1"))
    cabinet_drawers = cabinets.drawers
    query = an(
        entity(my_drawers).where(
            for_all(cabinet_drawers, not_(in_(my_drawers, cabinet_drawers))),
        )
    )

    results = list(query.evaluate())

    assert len(results) == 0

    cabinets = variable(Cabinet, world.views)
    drawer_var_2 = variable(Drawer, world.views)
    my_drawers = an(entity(drawer_var_2).where(drawer_var_2.handle.name == "Handle1"))
    drawers = cabinets.drawers
    query = an(entity(my_drawers).where(exists(drawers, in_(my_drawers, drawers))))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 1
    assert results[0].handle.name == "Handle1"


def test_for_all(handles_and_containers_world):
    world = handles_and_containers_world

    cabinets = variable(Cabinet, world.views)
    container_var = variable(Container, world.bodies)
    the_cabinet_container = the(
        entity(container_var).where(container_var.name == "Container2")
    )
    query = an(
        entity(the_cabinet_container).where(
            for_all(cabinets.container, the_cabinet_container == cabinets.container),
        )
    )

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 1
    assert results[0].name == "Container2"

    cabinets = variable(Cabinet, world.views)
    container_var_2 = variable(Container, world.bodies)
    the_cabinet_container = the(
        entity(container_var_2).where(container_var_2.name == "Container2")
    )
    query = an(
        entity(the_cabinet_container).where(
            for_all(cabinets.container, the_cabinet_container != cabinets.container),
        )
    )

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 0


def test_property_selection():
    """
    Test that properties can be selected from entities in a query.
    """
    v = variable(VectorsWithProperty, None)
    q = an(entity(v).where(v.vectors[0].x == 1))
