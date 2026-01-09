krrood.class_diagrams.attribute_introspector
============================================

.. py:module:: krrood.class_diagrams.attribute_introspector


Classes
-------

.. autoapisummary::

   krrood.class_diagrams.attribute_introspector.DiscoveredAttribute
   krrood.class_diagrams.attribute_introspector.AttributeIntrospector
   krrood.class_diagrams.attribute_introspector.DataclassOnlyIntrospector


Module Contents
---------------

.. py:class:: DiscoveredAttribute

   Attribute discovered on a class.


   .. py:attribute:: field
      :type:  dataclasses.Field

      The dataclass field object that is wrapped.



   .. py:attribute:: public_name
      :type:  typing_extensions.Optional[str]
      :value: None


      The public name of the field.



   .. py:attribute:: property_descriptor
      :type:  typing_extensions.Optional[krrood.ontomatic.property_descriptor.PropertyDescriptor]
      :value: None


      The property descriptor instance that manages the field.



.. py:class:: AttributeIntrospector

   Bases: :py:obj:`abc.ABC`


   Strategy that discovers class attributes for diagramming.

   Implementations return the set of dataclass-backed attributes that
   should appear on a class diagram, including their public names.


   .. py:method:: discover(owner_cls: typing_extensions.Type) -> typing_extensions.List[DiscoveredAttribute]
      :abstractmethod:


      Return discovered attributes for `owner_cls`.

      The `field` of each result must be a dataclass `Field` belonging to
      `owner_cls`, while `public_name` is how it should be addressed and displayed.



.. py:class:: DataclassOnlyIntrospector

   Bases: :py:obj:`AttributeIntrospector`


   Discover only public dataclass fields (no leading underscore).


   .. py:method:: discover(owner_cls: typing_extensions.Type) -> typing_extensions.List[DiscoveredAttribute]

      Return discovered attributes for `owner_cls`.

      The `field` of each result must be a dataclass `Field` belonging to
      `owner_cls`, while `public_name` is how it should be addressed and displayed.



