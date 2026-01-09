krrood.entity_query_language.orm.model
======================================

.. py:module:: krrood.entity_query_language.orm.model


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.orm.model.SymbolGraphMapping
   krrood.entity_query_language.orm.model.WrappedInstanceMapping


Module Contents
---------------

.. py:class:: SymbolGraphMapping

   Bases: :py:obj:`krrood.ormatic.dao.AlternativeMapping`\ [\ :py:obj:`krrood.entity_query_language.symbol_graph.SymbolGraph`\ ]


   Mapping specific for SymbolGraph.
   Import this class when you want to persist SymbolGraph.


   .. py:attribute:: instances
      :type:  typing_extensions.List[krrood.entity_query_language.symbol_graph.WrappedInstance]


   .. py:attribute:: predicate_relations
      :type:  typing_extensions.List[krrood.entity_query_language.symbol_graph.PredicateClassRelation]


   .. py:method:: from_domain_object(obj: krrood.entity_query_language.symbol_graph.SymbolGraph)
      :classmethod:


      Create this from a domain object.
      Do not create any DAOs here but the target DAO of `T`.
      The rest of the `to_dao` algorithm will process the fields of the created instance.

      :param obj: The source object.
      :return: A new instance of this mapping class.



   .. py:method:: to_domain_object() -> krrood.ormatic.dao.T

      Create a domain object from this instance.

      :return: The constructed domain object.



.. py:class:: WrappedInstanceMapping

   Bases: :py:obj:`krrood.ormatic.dao.AlternativeMapping`\ [\ :py:obj:`krrood.entity_query_language.symbol_graph.WrappedInstance`\ ]


   Base class for alternative mapping implementations.


   .. py:attribute:: instance
      :type:  typing_extensions.Optional[krrood.entity_query_language.predicate.Symbol]


   .. py:method:: from_domain_object(obj: krrood.entity_query_language.symbol_graph.WrappedInstance)
      :classmethod:


      Create this from a domain object.
      Do not create any DAOs here but the target DAO of `T`.
      The rest of the `to_dao` algorithm will process the fields of the created instance.

      :param obj: The source object.
      :return: A new instance of this mapping class.



   .. py:method:: to_domain_object() -> krrood.ormatic.dao.T

      Create a domain object from this instance.

      :return: The constructed domain object.



