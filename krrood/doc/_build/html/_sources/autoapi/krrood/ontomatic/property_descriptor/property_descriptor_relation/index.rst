krrood.ontomatic.property_descriptor.property_descriptor_relation
=================================================================

.. py:module:: krrood.ontomatic.property_descriptor.property_descriptor_relation


Classes
-------

.. autoapisummary::

   krrood.ontomatic.property_descriptor.property_descriptor_relation.PropertyDescriptorRelation


Module Contents
---------------

.. py:class:: PropertyDescriptorRelation

   Bases: :py:obj:`krrood.entity_query_language.symbol_graph.PredicateClassRelation`


   Edge data representing a relation between two wrapped instances that is represented structurally by a property
   descriptor attached to the source instance.


   .. py:property:: transitive
      :type: bool


      If the relation is transitive or not.



   .. py:property:: inverse_of
      :type: typing_extensions.Optional[typing_extensions.Type[krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor]]


      The inverse of the relation if it exists.



   .. py:method:: add_to_graph()

      Add the relation to the graph and infer additional relations if possible. In addition, update the value of
       the wrapped field in the source instance if this relation is an inferred relation.



   .. py:method:: update_source_wrapped_field_value()

      Update the wrapped field value for the source instance.



   .. py:method:: infer_super_relations()

      Infer all super relations of this relation.



   .. py:method:: infer_inverse_relation()

      Infer the inverse relation if it exists.



   .. py:property:: super_relations
      :type: typing_extensions.Iterable[typing_extensions.Tuple[krrood.entity_query_language.symbol_graph.WrappedInstance, krrood.class_diagrams.wrapped_field.WrappedField]]


      Find neighboring symbols connected by super edges.

      This method identifies neighboring symbols that are connected
      through edge with relation types that are superclasses of the current relation type.

      Also, it looks for role taker super relations of the source if it exists.

      :return: An iterator over neighboring symbols and relations that are super relations.



   .. py:property:: direct_super_relations

      Return the direct super relations of the source.



   .. py:property:: role_taker_super_relations

      Return the source role taker super relations.



   .. py:property:: role_taker_fields
      :type: typing_extensions.List[krrood.class_diagrams.wrapped_field.WrappedField]


      Return the role taker fields of the source role taker association.



   .. py:property:: source_role_taker_association
      :type: typing_extensions.Optional[krrood.class_diagrams.class_diagram.Association]


      Return the source role taker association of the relation.



   .. py:property:: inverse_domain_and_field
      :type: typing_extensions.Tuple[krrood.entity_query_language.symbol_graph.WrappedInstance, krrood.class_diagrams.wrapped_field.WrappedField]


      Get the inverse of the property descriptor.

      :return: The inverse domain instance and property descriptor field.



   .. py:property:: target_role_taker
      :type: typing_extensions.Optional[krrood.entity_query_language.symbol_graph.WrappedInstance]


      Return the role taker of the target if it exists.



   .. py:property:: inverse_field_from_target_role_taker
      :type: typing_extensions.Optional[krrood.class_diagrams.wrapped_field.WrappedField]


      Return the inverse field of this relation field that is stored in the role taker of the target.



   .. py:property:: target_role_taker_association
      :type: typing_extensions.Optional[krrood.class_diagrams.class_diagram.Association]


      Return role taker association of the target if it exists.



   .. py:property:: inverse_field
      :type: typing_extensions.Optional[krrood.class_diagrams.wrapped_field.WrappedField]


      Return the inverse field (if it exists) stored in the target of this relation.



   .. py:method:: infer_transitive_relations()

      Add all transitive relations of this relation type that results from adding this relation to the graph.



   .. py:method:: infer_transitive_relations_outgoing_from_source()

      Infer transitive relations outgoing from the source.



   .. py:method:: infer_transitive_relations_incoming_to_target()

      Infer transitive relations incoming to the target.



   .. py:property:: target_outgoing_relations_with_same_descriptor_type
      :type: typing_extensions.Iterable[krrood.entity_query_language.symbol_graph.PredicateClassRelation]


      Get the outgoing relations from the target that have the same property descriptor type as this relation.



   .. py:property:: source_incoming_relations_with_same_descriptor_type
      :type: typing_extensions.Iterable[krrood.entity_query_language.symbol_graph.PredicateClassRelation]


      Get the incoming relations from the source that have the same property descriptor type as this relation.



   .. py:property:: property_descriptor_cls
      :type: typing_extensions.Type[krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor]


      Return the property descriptor class of the relation.



