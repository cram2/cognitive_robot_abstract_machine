# krrood/entity_query_language/diagram_introspection.py
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing_extensions import List, Type

from ...class_diagrams.attribute_introspector import (
    AttributeIntrospector,
    DiscoveredAttribute,
)


@dataclass
class DescriptorAwareIntrospector(AttributeIntrospector):
    """Discover dataclass fields plus EQL descriptor-backed attributes.

    Public attributes that implement the descriptor protocol (`__get__` and `__set__`)
    and expose an `attr_name` are mapped to their hidden backing dataclass field,
    but are presented under the public attribute name.
    """

    def discover(self, owner_cls: Type) -> List[DiscoveredAttribute]:
        from .property_descriptor import PropertyDescriptor

        # Index all dataclass fields by name
        all_dc_fields = {f.name: f for f in dc_fields(owner_cls)}

        discovered: list[DiscoveredAttribute] = []

        # 1) Normal public dataclass fields
        for f in all_dc_fields.values():
            if not f.name.startswith("_"):
                discovered.append(DiscoveredAttribute(public_name=f.name, field=f))

        # 2) Descriptor-backed attributes via duck typing
        for public_name in dir(owner_cls):
            if public_name.startswith("_"):
                continue
            property_descriptor = getattr(owner_cls, public_name)
            if not isinstance(property_descriptor, PropertyDescriptor):
                continue
            discovered.append(
                DiscoveredAttribute(
                    public_name=public_name,
                    field=property_descriptor.wrapped_field.field,
                    property_descriptor=property_descriptor,
                )
            )

        return discovered
