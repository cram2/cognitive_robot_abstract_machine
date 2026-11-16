"""
ORMatic stores and restores dataclass fields typed with
:class:`typing_extensions.Annotated` as fields of the annotated type.
"""

from __future__ import annotations

from sqlalchemy import select

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.data_access_objects.helper import to_dao
from ..dataset.annotated_fields import AnnotatedFieldMember, AnnotatedFieldOwner
from ..dataset.ormatic_interface import AnnotatedFieldOwnerDAO


# %% type resolution


def test_annotated_field_resolves_to_the_annotated_type():
    """
    A field typed ``Annotated[T, ...]`` resolves to ``T``.
    """
    class_diagram = ClassDiagram(classes=[AnnotatedFieldOwner, AnnotatedFieldMember])
    owner = class_diagram.get_wrapped_class(clazz=AnnotatedFieldOwner)

    resolved_types = {
        wrapped_field.name: wrapped_field.resolved_type
        for wrapped_field in owner.fields
    }

    assert resolved_types == {"mass": float, "member": AnnotatedFieldMember}


# %% persistence


def test_annotated_fields_round_trip_through_the_database(session, database):
    """
    An annotated column and an annotated relationship are stored and restored.
    """
    original = AnnotatedFieldOwner(mass=2.5, member=AnnotatedFieldMember(label="a"))
    session.add(to_dao(original))
    session.commit()

    restored = session.scalars(select(AnnotatedFieldOwnerDAO)).one().from_dao()

    assert restored == original
