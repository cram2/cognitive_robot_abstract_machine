import os

import pytest

try:
    from rustworkx_utils import GraphVisualizer
except ImportError:
    GraphVisualizer = None

from ..dataset.semantic_world_like_classes import (
    Drawer,
    Handle,
    FixedConnection,
    Body,
    Container,
    PrismaticConnection,
    RevoluteConnection,
    View,
    Door,
    Wardrobe,
)
from krrood.entity_query_language.entity import (
    entity,
    variable,
    inference,
    and_,
)
from krrood.entity_query_language.entity_result_processors import an
from krrood.entity_query_language.conclusion import Add

from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.rule import alternative


@pytest.mark.skipif(GraphVisualizer is None, reason="requires rustworkx_utils")
def test_render_rx_graph_as_igraph_simple(handles_and_containers_world):
    world = handles_and_containers_world

    fixed_connection = variable(FixedConnection, world.connections)
    container = fixed_connection.parent
    handle = fixed_connection.child
    rule = an(
        entity(
            inference(Drawer)(handle=handle, container=container, world=world)
        ).where(
            HasType(handle, Handle),
        )
    )
    drawers = list(rule.evaluate())
    if os.path.exists("pdf_graph.pdf"):
        os.remove("pdf_graph.pdf")
    rule.visualize()
    assert os.path.exists("pdf_graph.pdf")
    os.remove("pdf_graph.pdf")


@pytest.mark.skipif(GraphVisualizer is None, reason="requires rustworkx_utils")
def test_render_rx_graph_as_igraph_complex(doors_and_drawers_world):
    world = doors_and_drawers_world

    body = variable(Body, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    container = variable(Container, domain=world.bodies)

    fixed_connection = variable(FixedConnection, domain=world.connections)
    fixed_connection_condition = and_(
        fixed_connection.parent == body, fixed_connection.child == handle
    )
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    revolute_connection = variable(RevoluteConnection, domain=world.connections)
    views = variable(View, domain=None, inferred=True)
    rule = an(
        entity(views).where(
            fixed_connection_condition,
            prismatic_connection.child == body,
        )
    )

    with rule:
        Add(views, inference(Drawer)(handle=handle, container=body, world=world))
        with alternative(
            revolute_connection.parent == body, revolute_connection.child == handle
        ):
            Add(views, inference(Door)(handle=handle, body=body, world=world))
        with alternative(
            fixed_connection_condition,
            body == revolute_connection.child,
            container == revolute_connection.parent,
            revolute_connection.world == world,
        ):
            Add(
                views,
                inference(Wardrobe)(
                    handle=handle, body=body, container=container, world=world
                ),
            )
    results = list(rule.evaluate())
    if os.path.exists("pdf_graph.pdf"):
        os.remove("pdf_graph.pdf")
    rule.visualize()
    assert os.path.exists("pdf_graph.pdf")
    os.remove("pdf_graph.pdf")
