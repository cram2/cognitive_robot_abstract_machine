krrood.ontomatic.property_descriptor.monitored_container
========================================================

.. py:module:: krrood.ontomatic.property_descriptor.monitored_container


Attributes
----------

.. autoapisummary::

   krrood.ontomatic.property_descriptor.monitored_container.monitored_type_map
   krrood.ontomatic.property_descriptor.monitored_container.T


Classes
-------

.. autoapisummary::

   krrood.ontomatic.property_descriptor.monitored_container.MonitoredContainer
   krrood.ontomatic.property_descriptor.monitored_container.MonitoredList
   krrood.ontomatic.property_descriptor.monitored_container.MonitoredSet


Module Contents
---------------

.. py:data:: monitored_type_map
   :type:  typing_extensions.Dict[typing_extensions.Type, typing_extensions.Type[MonitoredContainer]]

   A mapping of container types to their monitored container types.


.. py:data:: T

.. py:class:: MonitoredContainer(*args, descriptor: krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor, **kwargs)

   Bases: :py:obj:`typing_extensions.Generic`\ [\ :py:obj:`T`\ ], :py:obj:`abc.ABC`


   A container abstract class to be inherited from for specific container types to invoke the on-add
   callback of the descriptor. This is used by the
   :py:class:`krrood.ontomatic.property_descriptor.PropertyDescriptor` to apply
   implicit inferences.

   For example like here, the Set[Person] will be internally replaced with a MonitoredSet[Person] by the
   descriptor, this allows for catching additions/insertions/removals to the Set and applying implicit inferences:
   >>> from dataclasses import dataclass, field
   >>> from typing_extensions import Set
   >>> from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
   >>> from krrood.entity_query_language.predicate import Symbol
   ...
   >>> @dataclass
   >>> class Person(Symbol):
   >>>     name: str
   ...
   >>> @dataclass
   >>> class Company(Symbol):
   >>>     name: str
   >>>     members: Set[Person] = field(default_factory=set)
   ...
   >>> @dataclass
   >>> class Member(PropertyDescriptor):
   >>>     pass
   ...
   >>> Company.members = Member(Company, "members")
   >>> company = Company("Company")
   >>> person = Person("Person")
   >>> company.members.add(person)
   >>> assert isinstance(company.members, MonitoredSet)


.. py:class:: MonitoredList(*args, descriptor: krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor, **kwargs)

   Bases: :py:obj:`MonitoredContainer`, :py:obj:`list`


   A list that invokes the descriptor on_add for further implicit inferences.


   .. py:method:: extend(items)

      Extend list by appending elements from the iterable.



   .. py:method:: append(item)

      Append object to the end of the list.



   .. py:method:: insert(idx, item)

      Insert object before index.



.. py:class:: MonitoredSet(*args, descriptor: krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor, **kwargs)

   Bases: :py:obj:`MonitoredContainer`, :py:obj:`set`


   A set that invokes the descriptor on_add for further implicit inferences.


   .. py:method:: add(value)

      Add an element to a set.

      This has no effect if the element is already present.



   .. py:method:: update(values)

      Update a set with the union of itself and others.



