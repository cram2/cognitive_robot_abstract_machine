krrood.ontomatic.failures
=========================

.. py:module:: krrood.ontomatic.failures


Exceptions
----------

.. autoapisummary::

   krrood.ontomatic.failures.UnMonitoredContainerTypeForDescriptor


Module Contents
---------------

.. py:exception:: UnMonitoredContainerTypeForDescriptor

   Bases: :py:obj:`Exception`


   Raised when a descriptor is used on a field with a container type that is not monitored (i.e., is not a subclass of
   MonitoredContainer). This happens when your type hint of the field is using a container type that is not supported.


   .. py:attribute:: clazz
      :type:  type


   .. py:attribute:: field_name
      :type:  str


   .. py:attribute:: container_type
      :type:  type


