from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any

from krrood.adapters.json_serializer import to_json
from krrood.patterns.identifier_and_name import HasID, HasName

# %% objects a serialized object refers to


@dataclass
class ObjectReference(ABC):
    """
    How a serialized object points at an object it does not contain: by the object's
    :class:`~krrood.patterns.identifier_and_name.HasID` identifier and, where it has
    one, its :class:`~krrood.patterns.identifier_and_name.HasName` name, which says
    which object was meant where it cannot be found.

    The name is never looked up with, so two objects sharing one name stay apart.
    """

    subject: str
    """
    What the referring object calls the object it refers to, for example ``parent``.
    """

    @property
    def id_key(self) -> str:
        """
        Where the identifier of the object sits in the serialized object.
        """
        return f"{self.subject}_id"

    @property
    def name_key(self) -> str:
        """
        Where the name of the object sits in the serialized object, if it has one.
        """
        return f"{self.subject}_name"

    def write(self, data: dict[str, Any], identified_object: HasID) -> None:
        """
        Put the reference to an object into a serialized object.

        :param data: The json of the object that refers to it.
        :param identified_object: The object it refers to.
        """
        data[self.id_key] = to_json(identified_object.id)
        if not isinstance(identified_object, HasName):
            return
        data[self.name_key] = to_json(identified_object.name)

    @abstractmethod
    def resolve(self, data: dict[str, Any], **kwargs) -> HasID:
        """
        The object a serialized object refers to.

        :param data: The json of the object that refers to it.
        :param kwargs: The kwargs of the ``_from_json`` that is reading it.
        """
