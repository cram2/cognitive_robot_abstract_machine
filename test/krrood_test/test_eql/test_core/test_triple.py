from __future__ import annotations

import pytest

from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.predicate import ObjectType, SubjectType

from ...dataset.semantic_world_like_classes import (
    Container,
    Drawer,
    Handle,
    TripleDeclaringItsObjectFirst,
)


@pytest.fixture
def drawer() -> Drawer:
    """
    A drawer opened by its own handle.
    """
    return Drawer(handle=Handle("handle"), container=Container("container"))


# %% building a triple from its subject and object


def test_a_triple_takes_its_subject_and_object_whatever_order_its_fields_are_declared_in(
    drawer: Drawer,
):
    relation = TripleDeclaringItsObjectFirst.from_subject_object(drawer.handle, drawer)

    assert relation.subject is drawer.handle
    assert relation.object is drawer


def test_a_triple_built_from_its_subject_and_object_answers_whether_it_holds(
    drawer: Drawer,
):
    assert TripleDeclaringItsObjectFirst.from_subject_object(drawer.handle, drawer)()


def test_a_triple_built_from_a_symbolic_subject_is_symbolic(drawer: Drawer):
    handles = variable(Handle, [drawer.handle])

    relation = TripleDeclaringItsObjectFirst.from_subject_object(handles, drawer)

    assert isinstance(relation, InstantiatedVariable)


# %% the types a triple relates


@pytest.mark.parametrize(
    "parameter, related_type", [(SubjectType, Handle), (ObjectType, Drawer)]
)
def test_a_triple_reads_back_the_types_it_relates(parameter, related_type):
    assert (
        TripleDeclaringItsObjectFirst.get_type_of_generic_parameter(parameter)
        is related_type
    )
