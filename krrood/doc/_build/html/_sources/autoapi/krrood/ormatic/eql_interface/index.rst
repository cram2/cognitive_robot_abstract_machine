krrood.ormatic.eql_interface
============================

.. py:module:: krrood.ormatic.eql_interface


Exceptions
----------

.. autoapisummary::

   krrood.ormatic.eql_interface.EQLTranslationError
   krrood.ormatic.eql_interface.UnsupportedQueryTypeError
   krrood.ormatic.eql_interface.UnsupportedOperatorError
   krrood.ormatic.eql_interface.UnsupportedQuantifierError
   krrood.ormatic.eql_interface.AttributeResolutionError
   krrood.ormatic.eql_interface.MissingDAOError
   krrood.ormatic.eql_interface.DomainExtractionError


Classes
-------

.. autoapisummary::

   krrood.ormatic.eql_interface.VariableTypeExtractor
   krrood.ormatic.eql_interface.AttributeChainResolver
   krrood.ormatic.eql_interface.RelationshipResolver
   krrood.ormatic.eql_interface.OperatorMapper
   krrood.ormatic.eql_interface.DomainValueExtractor
   krrood.ormatic.eql_interface.JoinManager
   krrood.ormatic.eql_interface.EQLTranslator


Functions
---------

.. autoapisummary::

   krrood.ormatic.eql_interface.eql_to_sql


Module Contents
---------------

.. py:exception:: EQLTranslationError

   Bases: :py:obj:`Exception`


   Raised when an EQL expression cannot be translated into SQLAlchemy.


.. py:exception:: UnsupportedQueryTypeError

   Bases: :py:obj:`EQLTranslationError`


   Raised when an unsupported query type is encountered.


.. py:exception:: UnsupportedOperatorError

   Bases: :py:obj:`EQLTranslationError`


   Raised when an unsupported operator is encountered.


.. py:exception:: UnsupportedQuantifierError

   Bases: :py:obj:`EQLTranslationError`


   Raised when an unsupported quantifier is encountered.


.. py:exception:: AttributeResolutionError

   Bases: :py:obj:`EQLTranslationError`


   Raised when an attribute cannot be resolved.


.. py:exception:: MissingDAOError

   Bases: :py:obj:`EQLTranslationError`


   Raised when a DAO class cannot be found for a type.


.. py:exception:: DomainExtractionError

   Bases: :py:obj:`EQLTranslationError`


   Raised when a value cannot be extracted from a domain.


.. py:class:: VariableTypeExtractor

   Extracts underlying Variable and its python type from a leaf-like node.


   .. py:method:: extract(node: typing_extensions.Any) -> tuple[typing_extensions.Optional[krrood.entity_query_language.symbolic.Variable], typing_extensions.Optional[type]]

      Extract variable and type from a node.

      :param node: The node to extract from
      :return: Tuple of (variable, type)



.. py:class:: AttributeChainResolver

   Resolves attribute chains for EQL Attribute expressions.


   .. py:method:: extract_leaf_variable(attribute: krrood.entity_query_language.symbolic.Attribute) -> typing_extensions.Any

      Extract the leaf variable from an attribute chain.

      :param attribute: The attribute to extract from
      :return: The leaf variable or node



   .. py:method:: extract_base_dao(attribute: krrood.entity_query_language.symbolic.Attribute) -> typing_extensions.Optional[type]

      Extract the base DAO class from an attribute chain.

      :param attribute: The attribute to extract from
      :return: The DAO class or None



.. py:class:: RelationshipResolver

   Resolves relationships and foreign keys for DAO classes.


   .. py:method:: resolve_relationship_and_foreign_key(dao_class: type, attribute_name: str) -> tuple[typing_extensions.Any, typing_extensions.Any]

      Resolve the relationship and foreign key column for a DAO attribute.

      :param dao_class: The DAO class
      :param attribute_name: The attribute name
      :return: Tuple of (relationship, foreign_key_column)



.. py:class:: OperatorMapper

   Maps EQL operators to SQLAlchemy expressions.


   .. py:method:: map_comparison_operator(operation: typing_extensions.Any, left: typing_extensions.Any, right: typing_extensions.Any) -> typing_extensions.Any

      Map a comparison operator to a SQLAlchemy expression.

      :param operation: The operator
      :param left: Left operand
      :param right: Right operand
      :return: SQLAlchemy expression



   .. py:method:: map_contains_operator(operation: typing_extensions.Any, left: typing_extensions.Any, right: typing_extensions.Any) -> typing_extensions.Any

      Map a contains operator to a SQLAlchemy expression.

      :param operation: The operator
      :param left: Left operand
      :param right: Right operand
      :return: SQLAlchemy expression



.. py:class:: DomainValueExtractor

   Extracts values from EQL Variable/Literal domains.


   .. py:attribute:: session
      :type:  sqlalchemy.orm.Session


   .. py:method:: extract_from_literal(literal_node: krrood.entity_query_language.symbolic.Literal) -> typing_extensions.Any

      Extract values from a Literal node.

      :param literal_node: The Literal node
      :return: The extracted value(s)



   .. py:method:: extract_from_variable(variable: krrood.entity_query_language.symbolic.Variable) -> typing_extensions.Any

      Extract a value from a Variable domain.

      :param variable: The Variable
      :return: The extracted value



.. py:class:: JoinManager

   Manages JOIN operations for the EQL translator.

   Tracks both which relationship paths have been joined and the SQLAlchemy
   alias used for each path so that downstream column references can bind to
   the correct FROM element without triggering implicit joins.


   .. py:attribute:: aliases_by_path
      :type:  dict[tuple[type, str], typing_extensions.Any]


   .. py:attribute:: joined_tables
      :type:  set[type]


   .. py:method:: add_path_join(dao_class: type, attribute_name: str, alias: typing_extensions.Any) -> None

      Register a path-based JOIN and its alias.

      :param dao_class: The DAO class
      :param attribute_name: The attribute name
      :param alias: The SQLAlchemy aliased entity used for the join



   .. py:method:: is_path_joined(dao_class: type, attribute_name: str) -> bool

      Check if a path has already been joined.

      :param dao_class: The DAO class
      :param attribute_name: The attribute name
      :return: True if already joined



   .. py:method:: get_alias_for_path(dao_class: type, attribute_name: str) -> typing_extensions.Any

      Get the alias associated with a previously joined path.



   .. py:method:: add_table_join(dao_class: type) -> None

      Register a table-level JOIN.

      :param dao_class: The DAO class



   .. py:method:: is_table_joined(dao_class: type) -> bool

      Check if a table has already been joined.

      :param dao_class: The DAO class
      :return: True if already joined



.. py:class:: EQLTranslator

   Translate an EQL query into an SQLAlchemy query.

   This assumes the query has a structure like:
   - quantifier (an/the)
       - select like (entity, setof)
           - Root Condition
               - child 1
               - child 2
               - ...



   .. py:attribute:: eql_query
      :type:  krrood.entity_query_language.symbolic.SymbolicExpression


   .. py:attribute:: session
      :type:  sqlalchemy.orm.Session


   .. py:attribute:: sql_query
      :type:  typing_extensions.Optional[sqlalchemy.Select]
      :value: None



   .. py:attribute:: join_manager
      :type:  JoinManager


   .. py:property:: quantifier
      :type: krrood.entity_query_language.symbolic.SymbolicExpression


      Get the quantifier from the query.



   .. py:property:: select_like
      :type: typing_extensions.Any


      Get the select-like expression from the query.



   .. py:property:: root_condition
      :type: krrood.entity_query_language.symbolic.SymbolicExpression


      Get the root condition from the query.



   .. py:method:: translate() -> None

      Translate the EQL query to SQL.



   .. py:method:: evaluate() -> typing_extensions.List[typing_extensions.Any]

      Evaluate the translated SQL query.

      :return: Query results



   .. py:method:: translate_query(query: krrood.entity_query_language.symbolic.SymbolicExpression) -> typing_extensions.Optional[typing_extensions.Any]

      Translate an EQL query expression to SQL.

      :param query: The EQL query expression
      :return: SQLAlchemy expression or None



   .. py:method:: translate_and(query: krrood.entity_query_language.symbolic.AND) -> typing_extensions.Optional[typing_extensions.Any]

      Translate an eql.AND query into an sql.AND.

      :param query: EQL query
      :return: SQL expression or None if all parts are handled via JOINs.



   .. py:method:: translate_or(query: krrood.entity_query_language.symbolic.OR) -> typing_extensions.Optional[typing_extensions.Any]

      Translate an eql.OR query into an sql.OR.

      :param query: EQL query
      :return: SQL expression or None if all parts are handled via JOINs.



   .. py:method:: translate_comparator(query: krrood.entity_query_language.symbolic.Comparator) -> typing_extensions.Optional[typing_extensions.Any]

      Translate an eql.Comparator query into a SQLAlchemy expression.

      :param query: The comparator query
      :return: SQLAlchemy expression or None if handled via JOIN



   .. py:method:: translate_attribute(query: krrood.entity_query_language.symbolic.Attribute) -> typing_extensions.Any

      Translate an eql.Attribute query into a SQLAlchemy column.

      :param query: The attribute query
      :return: SQLAlchemy column expression



.. py:function:: eql_to_sql(query: krrood.entity_query_language.symbolic.SymbolicExpression, session: sqlalchemy.orm.Session) -> EQLTranslator

   Translate an EQL query to SQL.

   :param query: The EQL query
   :param session: The SQLAlchemy session
   :return: The translator instance


