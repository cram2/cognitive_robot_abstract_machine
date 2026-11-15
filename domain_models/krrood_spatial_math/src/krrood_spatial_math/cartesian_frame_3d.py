from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from uuid import UUID

from typing_extensions import Any, TypeVar

from krrood.adapters.json_serializer import from_json
from krrood.adapters.object_reference import ObjectReference
from krrood.patterns.identifier_and_name import HasID, HasName, NameT
from krrood.patterns.interface import Interface
from krrood_spatial_math.exceptions import MissingCartesianFrame3DResolver

# %% frames that positions and directions are expressed in


class CartesianFrame3D(HasName[NameT], HasID, Interface):
    """
    A right-handed Cartesian frame in three dimensions, in metres, such as the frame of
    a robot's body.
    """

    @abstractmethod
    def express_in_own_frame(
        self, value: ValueInCartesianFrame3DT
    ) -> ValueInCartesianFrame3DT:
        """
        :param value: A value in the same space as this frame.
        :return: The same value, given in this frame.
        """


class ValueInCartesianFrame3D(Interface):
    """
    A value, such as a point or a pose, given in a :class:`CartesianFrame3D`.
    """

    reference_frame: CartesianFrame3D | None
    """
    The frame the value is given in, or *None* where the value is given in no frame.
    """


ValueInCartesianFrame3DT = TypeVar(
    "ValueInCartesianFrame3DT", bound=ValueInCartesianFrame3D
)
"""
The kind of value a Cartesian frame expresses.
"""


class CartesianFrame3DResolver(Interface):
    """
    Finds the Cartesian frames serialized spatial types refer to, while one JSON
    document is deserialized.

    It travels through nested ``from_json`` calls in their keyword arguments.
    """

    @abstractmethod
    def get(self, key: UUID, name: Any = None) -> CartesianFrame3D:
        """
        :param key: The identifier the document refers to the frame with.
        :param name: The name the reference to the frame went by, which says which frame
            was meant when it cannot be found.
        :return: The frame with that identifier.
        """


# %% referring to a frame in a serialized spatial type


@dataclass
class CartesianFrame3DReference(ObjectReference):
    """
    How a serialized spatial type points at a :class:`CartesianFrame3D`.
    """

    def resolve(self, *, data: dict[str, Any], **kwargs) -> CartesianFrame3D:
        """
        :param data: The serialized spatial type that refers to the frame.
        :param kwargs: The keyword arguments of the ``from_json`` call, which carry the
            :class:`CartesianFrame3DResolver` that finds the frame.
        :return: The frame the serialized spatial type refers to.
        :raises MissingCartesianFrame3DResolver: If the kwargs carry no
            :class:`CartesianFrame3DResolver`.
        """
        resolvers = [
            value
            for value in kwargs.values()
            if isinstance(value, CartesianFrame3DResolver)
        ]
        if not resolvers:
            raise MissingCartesianFrame3DResolver(frame_id_key=self.id_key)
        return resolvers[0].get(
            key=from_json(data[self.id_key]), name=from_json(data.get(self.name_key))
        )
