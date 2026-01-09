krrood.ontomatic.property_descriptor.property_descriptor
========================================================

.. py:module:: krrood.ontomatic.property_descriptor.property_descriptor


Attributes
----------

.. autoapisummary::

   krrood.ontomatic.property_descriptor.property_descriptor.SymbolType
   krrood.ontomatic.property_descriptor.property_descriptor.DomainRangeMap


Classes
-------

.. autoapisummary::

   krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor


Module Contents
---------------

.. py:data:: SymbolType

   Type alias for symbol types.


.. py:data:: DomainRangeMap

   Type alias for the domain-range map.


.. py:class:: PropertyDescriptor

   Bases: :py:obj:`krrood.entity_query_language.predicate.Symbol`


   Descriptor managing a data class field while giving it metadata like superproperties,
   sub-properties, inverse, transitivity, ...etc.

   The descriptor injects a hidden dataclass-managed attribute (backing storage) into the owner class
   and collects domain and range types for introspection.

   The way this should be used is after defining your dataclasses you declare either in the same file or in a separate
   file the descriptors for each field that is considered a relation between two symbol types.

   Example:
       >>> from krrood.ontomatic.property_descriptor.mixins import HasInverseProperty
       >>> from dataclasses import dataclass
       >>> from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
       >>> @dataclass
       ... class Company(Symbol):
       ...     name: str
       ...     members: Set[Person] = field(default_factory=set)
       ...
       >>> @dataclass
       ... class Person(Symbol):
       ...     name: str
       ...     works_for: Set[Company] = field(default_factory=set)
       ...
       >>> @dataclass
       >>> class Member(PropertyDescriptor):
       ...     pass
       ...
       >>> @dataclass
       ... class MemberOf(PropertyDescriptor, HasInverseProperty):
       ...     @classmethod
       ...     def get_inverse(cls) -> Type[PropertyDescriptor]:
       ...         return Member
       ...
       >>> @dataclass
       >>> class WorksFor(MemberOf):
       ...     pass
       ...
       >>> Person.works_for = WorksFor(Person, "works_for")
       >>> Company.members = Member(Company, "members")


   .. py:attribute:: domain
      :type:  SymbolType

      The domain type for this descriptor instance.



   .. py:attribute:: field_name
      :type:  str

      The name of the field on the domain type that this descriptor instance manages.



   .. py:attribute:: wrapped_field
      :type:  krrood.class_diagrams.wrapped_field.WrappedField

      The wrapped field instance that this descriptor instance manages.



   .. py:attribute:: domain_range_map
      :type:  typing_extensions.ClassVar[typing_extensions.DefaultDict[typing_extensions.Type[PropertyDescriptor], DomainRangeMap]]

      A mapping from descriptor class to the mapping from domain types to range types for that descriptor class.



   .. py:attribute:: all_domains
      :type:  typing_extensions.ClassVar[typing_extensions.Dict[typing_extensions.Type[PropertyDescriptor], typing_extensions.Set[SymbolType]]]

      A set of all domain types for this descriptor class.



   .. py:attribute:: all_ranges
      :type:  typing_extensions.ClassVar[typing_extensions.Dict[typing_extensions.Type[PropertyDescriptor], typing_extensions.Set[SymbolType]]]

      A set of all range types for this descriptor class.



   .. py:property:: private_attr_name
      :type: str


      The name of the private attribute that stores the values on the owner instance.



   .. py:property:: is_iterable

      Whether the field is iterable or not



   .. py:property:: range
      :type: SymbolType


      The range type for this descriptor instance.



   .. py:method:: add_relation_to_the_graph(domain_value: krrood.entity_query_language.predicate.Symbol, range_value: krrood.entity_query_language.predicate.Symbol, inferred: bool = False) -> None

      Add the relation between the domain_value and the range_value to the symbol graph.

      :param domain_value: The domain value (i.e., the instance that this descriptor is attached to).
      :param range_value: The range value (i.e., the value to set on the managed attribute, and is the target of the
       relation).
      :param inferred: Whether the relation is inferred or not.



   .. py:method:: update_value(domain_value: krrood.entity_query_language.predicate.Symbol, range_value: krrood.entity_query_language.predicate.Symbol) -> bool

      Update the value of the managed attribute

      :param domain_value: The domain value to update (i.e., the instance that this descriptor is attached to).
      :param range_value: The range value to update (i.e., the value to set on the managed attribute).



   .. py:method:: get_associated_field_of_domain_type(domain_type: typing_extensions.Union[typing_extensions.Type[krrood.entity_query_language.predicate.Symbol], krrood.class_diagrams.class_diagram.WrappedClass]) -> typing_extensions.Optional[krrood.class_diagrams.wrapped_field.WrappedField]
      :classmethod:


      Get the field of the domain type that is associated with this descriptor class.

      :param domain_type: The domain type that has an associated field with this descriptor class.



   .. py:method:: get_fields_of_superproperties_in_role_taker_of_class(domain_type: typing_extensions.Union[SymbolType, krrood.class_diagrams.class_diagram.WrappedClass]) -> typing_extensions.Tuple[typing_extensions.Optional[krrood.class_diagrams.wrapped_field.WrappedField], typing_extensions.List[krrood.class_diagrams.wrapped_field.WrappedField]]
      :classmethod:


      Return the role-taker field and all associated fields that are superproperties of this descriptor class.

      :param domain_type: The domain type that has a role-taker, where the role-taker has associated fields with the
       super properties of this descriptor class.



   .. py:method:: get_fields_of_superproperties(domain_type: typing_extensions.Union[SymbolType, krrood.class_diagrams.class_diagram.WrappedClass]) -> typing_extensions.Tuple[krrood.class_diagrams.wrapped_field.WrappedField, Ellipsis]
      :classmethod:


      Get the fields of the domain type that are associated with the super classes of this descriptor class.

      :param domain_type: The domain type that has an associated field with the super classes of this descriptor class.



