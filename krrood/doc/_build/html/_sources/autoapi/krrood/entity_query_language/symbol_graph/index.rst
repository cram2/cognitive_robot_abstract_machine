krrood.entity_query_language.symbol_graph
=========================================

.. py:module:: krrood.entity_query_language.symbol_graph


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.symbol_graph.PredicateClassRelation
   krrood.entity_query_language.symbol_graph.WrappedInstance
   krrood.entity_query_language.symbol_graph.SymbolGraph


Module Contents
---------------

.. py:class:: PredicateClassRelation

   Edge data representing a predicate-based relation between two wrapped instances.

   The relation carries a flag indicating whether it was inferred or added directly.


   .. py:attribute:: source
      :type:  WrappedInstance

      The source of the predicate



   .. py:attribute:: target
      :type:  WrappedInstance

      The target of the predicate



   .. py:attribute:: wrapped_field
      :type:  krrood.class_diagrams.wrapped_field.WrappedField

      The dataclass field in the source class that represents this relation with the target.



   .. py:attribute:: inferred
      :type:  bool
      :value: False


      Whether it was inferred or not.



   .. py:method:: add_to_graph() -> bool

      Add the relation to the graph.

      :return: True if the relation was newly added, False if it already existed.



   .. py:property:: color
      :type: str



.. py:class:: WrappedInstance

   A node wrapper around a concrete Symbol instance used in the instance graph.


   .. py:property:: instance
      :type: typing_extensions.Optional[krrood.entity_query_language.predicate.Symbol]


      :return: The symbol that is referenced to. Can return None if this symbol is garbage collected already.



   .. py:attribute:: instance_reference
      :type:  weakref.ReferenceType[krrood.entity_query_language.predicate.Symbol]
      :value: None


      A weak reference to the symbol instance this wraps.



   .. py:attribute:: index
      :type:  typing_extensions.Optional[int]
      :value: None


      Index in the instance graph of the symbol graph that manages this object.



   .. py:attribute:: inferred
      :type:  bool
      :value: False


      Rather is instance was inferred or constructed.



   .. py:attribute:: instance_type
      :type:  typing_extensions.Type[krrood.entity_query_language.predicate.Symbol]
      :value: None


      The type of the instance.
      This is needed to clean it up from the cache after the instance reference died.



   .. py:property:: name

      Return a unique display name composed of class name and node index.



   .. py:property:: color
      :type: str



.. py:class:: SymbolGraph

   A singleton combination of a class and instance diagram.
   This class tracks the life cycles `Symbol` instance created in the python process.
   Furthermore, relations between instances are also tracked.

   Relations are represented as edges where each edge has a relation object attached to it. The relation object
   contains also the Predicate object that represents the relation.

   The construction of this object will do nothing if a singleton instance of this already exists.
   Make sure to call `clear()` before constructing this object if you want a new one.


   .. py:property:: class_diagram
      :type: krrood.class_diagrams.ClassDiagram



   .. py:method:: add_node(wrapped_instance: WrappedInstance)

      Add a wrapped instance to the cache.

      :param wrapped_instance: The instance to add.



   .. py:method:: remove_node(wrapped_instance: WrappedInstance)

      Remove a wrapped instance from the cache.

      :param wrapped_instance: The instance to remove.



   .. py:method:: remove_dead_instances()


   .. py:method:: get_instances_of_type(type_: typing_extensions.Type[krrood.entity_query_language.predicate.Symbol]) -> typing_extensions.Iterable[krrood.entity_query_language.predicate.Symbol]

      Get all wrapped instances of the given type and all its subclasses.

      :param type_: The symbol type to look for
      :return: All wrapped instances that refer to an instance of the given type.



   .. py:method:: get_wrapped_instance(instance: typing_extensions.Any) -> typing_extensions.Optional[WrappedInstance]


   .. py:method:: ensure_wrapped_instance(instance: typing_extensions.Any) -> WrappedInstance

      Ensures that the given instance is wrapped into a `WrappedInstance`. If the
      instance is not already wrapped, creates a new `WrappedInstance` object and
      adds it as a node. Returns the wrapped instance.

      :param instance: The object to be checked and wrapped if necessary.:
      :return: WrappedInstance: The wrapped object.



   .. py:method:: clear() -> None


   .. py:method:: add_instance(wrapped_instance: WrappedInstance) -> None

      Add a wrapped instance to the graph.

      This is an adapter that delegates to add_node to keep API compatibility with
      SymbolGraphMapping.create_from_dao.



   .. py:method:: add_relation(relation: PredicateClassRelation) -> bool

      Add a relation edge to the instance graph.



   .. py:method:: relation_exists(relation: PredicateClassRelation) -> bool


   .. py:method:: relations() -> typing_extensions.Iterable[PredicateClassRelation]


   .. py:property:: wrapped_instances
      :type: typing_extensions.List[WrappedInstance]



   .. py:method:: get_incoming_relations_with_type(wrapped_instance: WrappedInstance, relation_type: typing_extensions.Type[PredicateClassRelation]) -> typing_extensions.Iterable[PredicateClassRelation]

      Get all relations with the given type that are incoming to the given wrapped instance.

      :param wrapped_instance: The wrapped instance to get the relations from.
      :param relation_type: The type of the relation to filter for.



   .. py:method:: get_incoming_relations_with_condition(wrapped_instance: WrappedInstance, edge_condition: typing_extensions.Callable[[PredicateClassRelation], bool]) -> typing_extensions.Iterable[PredicateClassRelation]

      Get all relations with the given condition that are incoming to the given wrapped instance.

      :param wrapped_instance: The wrapped instance to get the relations from.
      :param edge_condition: The condition to filter for.



   .. py:method:: get_incoming_relations(wrapped_instance: WrappedInstance) -> typing_extensions.Iterable[PredicateClassRelation]

      Get all relations incoming to the given wrapped instance.

      :param wrapped_instance: The wrapped instance to get the relations from.



   .. py:method:: get_outgoing_relations_with_type(wrapped_instance: WrappedInstance, relation_type: typing_extensions.Type[PredicateClassRelation]) -> typing_extensions.Iterable[PredicateClassRelation]

      Get all relations with the given type that are outgoing from the given wrapped instance.

      :param wrapped_instance: The wrapped instance to get the relations from.
      :param relation_type: The type of the relation to filter for.



   .. py:method:: get_outgoing_relations_with_condition(wrapped_instance: WrappedInstance, edge_condition: typing_extensions.Callable[[PredicateClassRelation], bool]) -> typing_extensions.Iterable[PredicateClassRelation]

      Get all relations with the given condition that are outgoing from the given wrapped instance.

      :param wrapped_instance: The wrapped instance to get the relations from.
      :param edge_condition: The condition to filter for.



   .. py:method:: get_outgoing_relations(wrapped_instance: WrappedInstance) -> typing_extensions.Iterable[PredicateClassRelation]

      Get all relations outgoing from the given wrapped instance.

      :param wrapped_instance: The wrapped instance to get the relations from.



   .. py:method:: to_dot(filepath: str, format_='svg', graph_type='instance', without_inherited_associations: bool = True) -> None

      Generate a dot file from the instance graph, requires graphviz and pydot libraries.

      :param filepath: The path to the dot file.
      :param format_: The format of the dot file (svg, png, ...).
      :param graph_type: The type of the graph to generate (instance, type).
      :param without_inherited_associations: Whether to include inherited associations in the graph.



