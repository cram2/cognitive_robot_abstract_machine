"""
Dataclasses whose fields are typed with :class:`typing_extensions.Annotated`, to verify
that the metadata an annotation carries does not hide the field's type from ORMatic.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Annotated


# %% annotated fields


@dataclass(kw_only=True)
class AnnotatedFieldMember:
    """
    An object another object refers to through an annotated field.
    """

    label: str
    """
    The label that tells members apart.
    """


@dataclass(kw_only=True)
class AnnotatedFieldOwner:
    """
    An object with an annotated column and an annotated relationship.
    """

    mass: Annotated[float, "kilograms"]
    """
    The mass of the owner, annotated with its unit.
    """

    member: Annotated[AnnotatedFieldMember, "the member this owner refers to"]
    """
    The member this owner refers to.
    """
