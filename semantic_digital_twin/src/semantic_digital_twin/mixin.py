from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List

from krrood.patterns.field_metadata import FieldMetadata
from typing_extensions import Any, Dict, Optional, Type, TypeVar

from semantic_digital_twin.exceptions import DuplicateSimulatorPropertyError


@dataclass
class SimulatorAttributeName(FieldMetadata):
    """
    Marks a field of a simulator property whose attribute the simulator spells
    differently than the field is named.
    """

    attribute_name: str
    """
    The simulator's own name for the attribute.
    """


@dataclass
class SimulatorAdditionalProperty:
    """
    Class representing an additional property for a simulator.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: The fields as a dictionary, each under the simulator's own name for it
            if the field declares one (see :class:`SimulatorAttributeName`), else under
            its own.
        """
        return {
            self._attribute_name(declared_field.name): getattr(
                self, declared_field.name
            )
            for declared_field in fields(self)
        }

    @classmethod
    def _attribute_name(cls, field_name: str) -> str:
        """
        :param field_name: One of this property's fields.
        :return: The simulator's name for it.
        """
        renamed = SimulatorAttributeName.of_field(cls, field_name)
        return field_name if renamed is None else renamed.attribute_name


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
                raise DuplicateSimulatorPropertyError(property_type)
        self.simulator_additional_properties.append(simulator_property)

    def get_simulator_property_of_type(
        self, property_type: Type[TUniqueSimulatorProperty]
    ) -> Optional[TUniqueSimulatorProperty]:
        """
        The property of ``property_type`` this entity carries.

        :param property_type: The type of property to look up.
        :return: The property, or ``None`` if none of that type is attached.
        """
        return next(
            (
                simulator_property
                for simulator_property in self.simulator_additional_properties
                if isinstance(simulator_property, property_type)
            ),
            None,
        )
