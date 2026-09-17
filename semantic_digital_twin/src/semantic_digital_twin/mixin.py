from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import List

from typing_extensions import Any, Dict, Optional, Type, TypeVar

from semantic_digital_twin.exceptions import DuplicateSimulatorPropertyError


class FieldMetadata(StrEnum):
    """
    The keys a simulator property's fields carry in their ``metadata``.
    """

    SIMULATOR_ATTRIBUTE_NAME = "simulator_attribute_name"
    """
    The name the simulator's own attribute for the field carries, where it differs from
    the field's name.
    """


@dataclass
class SimulatorAdditionalProperty:
    """
    Class representing an additional property for a simulator.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: The fields as a dictionary, each under the simulator's own name for it
            if the field declares one (see :attr:`FieldMetadata.SIMULATOR_ATTRIBUTE_NAME`), else
            under its own.
        """
        return {
            declared_field.metadata.get(
                FieldMetadata.SIMULATOR_ATTRIBUTE_NAME, declared_field.name
            ): getattr(self, declared_field.name)
            for declared_field in fields(self)
        }


@dataclass
class UniqueSimulatorProperty(SimulatorAdditionalProperty):
    """
    A simulator property an entity carries at most one of, such as the physical
    settings of one body or one geometry; a simulator reads exactly one and would
    silently ignore the rest.

    Properties an entity may carry several of, such as cameras or lights, are plain
    :class:`SimulatorAdditionalProperty`.
    """

    ...


TUniqueSimulatorProperty = TypeVar(
    "TUniqueSimulatorProperty", bound=UniqueSimulatorProperty
)
"""
The concrete kind of property a lookup asks for, so that the lookup returns that kind
rather than the base class.
"""


@dataclass(eq=False)
class HasSimulatorProperties:
    """
    Mixin class to add simulator additional properties to a data class.
    """

    simulator_additional_properties: List[SimulatorAdditionalProperty] = field(
        default_factory=list, kw_only=True, repr=False
    )
    """
    A list of additional properties for the simulator, it can contain properties of
    multiple simulators. Extend it with :meth:`add_simulator_property`, which keeps a
    :class:`UniqueSimulatorProperty` from being attached twice.
    """

    def add_simulator_property(
        self, simulator_property: SimulatorAdditionalProperty
    ) -> None:
        """
        Attach a property to this entity.

        :param simulator_property: The property to attach.
        :raises DuplicateSimulatorPropertyError: If the property is a
            :class:`UniqueSimulatorProperty` and one of its type is attached already.
        """
        if isinstance(simulator_property, UniqueSimulatorProperty):
            property_type = type(simulator_property)
            if self.get_simulator_property_of_type(property_type) is not None:
                raise DuplicateSimulatorPropertyError(property_type, 2)
        self.simulator_additional_properties.append(simulator_property)

    def get_simulator_property_of_type(
        self, property_type: Type[TUniqueSimulatorProperty]
    ) -> Optional[TUniqueSimulatorProperty]:
        """
        The one property of ``property_type`` this entity carries.

        :param property_type: The type of property to look up.
        :return: The property, or ``None`` if none of that type is attached.
        :raises DuplicateSimulatorPropertyError: If more than one is attached, which
            :meth:`add_simulator_property` prevents but a list handed in whole does not.
        """
        matches = [
            simulator_property
            for simulator_property in self.simulator_additional_properties
            if isinstance(simulator_property, property_type)
        ]
        if len(matches) > 1:
            raise DuplicateSimulatorPropertyError(property_type, len(matches))
        if not matches:
            return None
        return matches[0]
