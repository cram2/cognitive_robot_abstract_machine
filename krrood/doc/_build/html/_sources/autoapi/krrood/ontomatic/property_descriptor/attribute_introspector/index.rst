krrood.ontomatic.property_descriptor.attribute_introspector
===========================================================

.. py:module:: krrood.ontomatic.property_descriptor.attribute_introspector


Classes
-------

.. autoapisummary::

   krrood.ontomatic.property_descriptor.attribute_introspector.DescriptorAwareIntrospector


Module Contents
---------------

.. py:class:: DescriptorAwareIntrospector

   Bases: :py:obj:`krrood.class_diagrams.attribute_introspector.AttributeIntrospector`


   Discover dataclass fields plus EQL descriptor-backed attributes.

   Public attributes that implement the descriptor protocol (`__get__` and `__set__`)
   and expose an `attr_name` are mapped to their hidden backing dataclass field,
   but are presented under the public attribute name.


   .. py:method:: discover(owner_cls: typing_extensions.Type) -> typing_extensions.List[krrood.class_diagrams.attribute_introspector.DiscoveredAttribute]

      Return discovered attributes for `owner_cls`.

      The `field` of each result must be a dataclass `Field` belonging to
      `owner_cls`, while `public_name` is how it should be addressed and displayed.



