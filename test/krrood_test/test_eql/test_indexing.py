from dataclasses import dataclass
from typing_extensions import Dict, List

from krrood.entity_query_language.entity import entity, let
from krrood.entity_query_language.quantify_entity import an
from krrood.entity_query_language.predicate import Symbol
from krrood.entity_query_language.symbol_graph import SymbolGraph


def test_indexing_on_dict_field():

    @dataclass
    class Item(Symbol):
        name: str
        attrs: Dict[str, int]

        def __hash__(self):
            return hash(self.name)

    @dataclass(eq=False)
    class World(Symbol):
        items: List[Item]

        def __hash__(self):
            return hash(id(self))

    SymbolGraph().clear()
    SymbolGraph()

    world = World(
        [
            Item("A", {"score": 1}),
            Item("B", {"score": 2}),
            Item("C", {"score": 2}),
        ]
    )

    i = let(type_=Item, domain=world.items)
    q = an(entity(i, i.attrs["score"] == 2))
    res = list(q.evaluate())
    assert {x.name for x in res} == {"B", "C"}


def test_indexing_2():
    @dataclass(unsafe_hash=True)
    class Shape(Symbol):
        name: str
        color: str

    @dataclass
    class Body(Symbol):
        shapes: List[Shape]

        def __hash__(self):
            return hash(id(self))

    SymbolGraph().clear()
    SymbolGraph()

    world_bodies = [
        Body(shapes=[Shape("shape1", color="red"), Shape("shape2", color="blue")]),
        Body(shapes=[Shape("shape1", color="green"), Shape("shape2", color="black")]),
    ]

    body = let(Body, world_bodies)
    body_tha_has_red_shape = an(entity(body, body.shapes[0].color == "red")).evaluate()
    body_tha_has_red_shape = list(body_tha_has_red_shape)
    assert len(body_tha_has_red_shape) == 1
    assert body_tha_has_red_shape[0].shapes[0].color == "red"
