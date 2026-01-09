krrood.ontomatic.property_descriptor.mixins
===========================================

.. py:module:: krrood.ontomatic.property_descriptor.mixins


Classes
-------

.. autoapisummary::

   krrood.ontomatic.property_descriptor.mixins.TransitiveProperty
   krrood.ontomatic.property_descriptor.mixins.HasInverseProperty


Module Contents
---------------

.. py:class:: TransitiveProperty

   A mixin for descriptors that are transitive.


.. py:class:: HasInverseProperty

   Bases: :py:obj:`abc.ABC`


   A mixin for descriptors that have an inverse property.


   .. py:method:: get_inverse() -> typing_extensions.Type[krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor]
      :classmethod:

      :abstractmethod:


      The inverse of the property.



