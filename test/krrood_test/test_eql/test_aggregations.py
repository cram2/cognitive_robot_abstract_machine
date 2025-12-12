from krrood.entity_query_language.core_entity import (
    flatten,
    _entity,
    not_,
    in_,
    for_all,
    let,
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

    views = let(Cabinet, world.views)
    drawers = flatten(views.drawers)
    query = an(_entity(drawers))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 3
    assert {row.handle.name for row in results} == {"Handle1", "Handle2", "Handle3"}


def test_flatten_iterable_attribute_and_use_not_equal(handles_and_containers_world):
    world = handles_and_containers_world

    cabinets = let(Cabinet, world.views)
    drawer_1 = an(_entity(d := let(Drawer, world.views), d.handle.name == "Handle1"))
    drawers = flatten(cabinets.drawers)
    query = an(_entity(drawers, drawer_1 != drawers))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 2
    assert {row.handle.name for row in results} == {"Handle2", "Handle3"}


def test_exists_and_for_all(handles_and_containers_world):
    world = handles_and_containers_world

    cabinets = let(Cabinet, world.views)
    my_drawers = an(_entity(d := let(Drawer, world.views), d.handle.name == "Handle1"))
    cabinet_drawers = cabinets.drawers
    query = an(
        _entity(
            my_drawers,
            for_all(cabinet_drawers, not_(in_(my_drawers, cabinet_drawers))),
        )
    )

    results = list(query.evaluate())

    assert len(results) == 0

    cabinets = let(Cabinet, world.views)
    my_drawers = an(_entity(d := let(Drawer, world.views), d.handle.name == "Handle1"))
    drawers = cabinets.drawers
    query = an(_entity(my_drawers, exists(drawers, in_(my_drawers, drawers))))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 1
    assert results[0].handle.name == "Handle1"


def test_for_all(handles_and_containers_world):
    world = handles_and_containers_world

    cabinets = let(Cabinet, world.views)
    the_cabinet_container = the(
        _entity(c := let(Container, world.bodies), c.name == "Container2")
    )
    query = an(
        _entity(
            the_cabinet_container,
            for_all(cabinets.container, the_cabinet_container == cabinets.container),
        )
    )

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 1
    assert results[0].name == "Container2"

    cabinets = let(Cabinet, world.views)
    the_cabinet_container = the(
        _entity(c := let(Container, world.bodies), c.name == "Container2")
    )
    query = an(
        _entity(
            the_cabinet_container,
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
    q = an(_entity(v := let(VectorsWithProperty, None), v.vectors[0].x == 1))
