krrood.class_diagrams.class_diagram
===================================

.. py:module:: krrood.class_diagrams.class_diagram


Attributes
----------

.. autoapisummary::

   krrood.class_diagrams.class_diagram.RWXNode


Exceptions
----------

.. autoapisummary::

   krrood.class_diagrams.class_diagram.ParseError


Classes
-------

.. autoapisummary::

   krrood.class_diagrams.class_diagram.ClassRelation
   krrood.class_diagrams.class_diagram.Inheritance
   krrood.class_diagrams.class_diagram.Association
   krrood.class_diagrams.class_diagram.HasRoleTaker
   krrood.class_diagrams.class_diagram.WrappedClass
   krrood.class_diagrams.class_diagram.ClassDiagram


Module Contents
---------------

.. py:data:: RWXNode
   :value: None


.. py:class:: ClassRelation

   Bases: :py:obj:`abc.ABC`


   Abstract base class representing a relationship between two classes in a UML class diagram.


   .. py:attribute:: source
      :type:  WrappedClass

      The source class in the relation.



   .. py:attribute:: target
      :type:  WrappedClass

      The target class in the relation.



   .. py:property:: color
      :type: str


      Default edge color used when visualizing the relation.



.. py:class:: Inheritance

   Bases: :py:obj:`ClassRelation`


   Represents an inheritance (generalization) relationship in UML.

   This is an "is-a" relationship where the source class inherits from the target class.
   In UML notation, this is represented by a solid line with a hollow triangle pointing to the parent class.


.. py:class:: Association

   Bases: :py:obj:`ClassRelation`


   Represents a general association relationship between two classes.

   This is the most general form of relationship, indicating that instances of one class
   are connected to instances of another class. In UML notation, this is shown as a solid line.


   .. py:attribute:: field
      :type:  krrood.class_diagrams.wrapped_field.WrappedField

      The field in the source class that creates this association with the target class.



   .. py:property:: one_to_many
      :type: bool


      Whether the association is one-to-many (True) or many-to-one (False).



   .. py:method:: get_key(include_field_name: bool = False) -> tuple

      A tuple representing the key of the association.



.. py:class:: HasRoleTaker

   Bases: :py:obj:`Association`


   This is an association between a role and a role taker where the role class contains a role taker field.


.. py:exception:: ParseError

   Bases: :py:obj:`TypeError`


   Error that will be raised when the parser encounters something that can/should not be parsed.

   For instance, Union types


.. py:class:: WrappedClass

   A node wrapper around a Python class used in the class diagram graph.


   .. py:attribute:: index
      :type:  typing_extensions.Optional[int]
      :value: None



   .. py:attribute:: clazz
      :type:  typing_extensions.Type


   .. py:property:: fields
      :type: typing_extensions.List[krrood.class_diagrams.wrapped_field.WrappedField]


      Return wrapped fields discovered by the diagram’s attribute introspector.

      Public names from the introspector are used to index `_wrapped_field_name_map_`.



   .. py:property:: name

      Return a unique display name composed of class name and node index.



.. py:class:: ClassDiagram

   A graph of classes and their relations discovered via attribute introspection.


   .. py:attribute:: classes
      :type:  dataclasses.InitVar[typing_extensions.List[typing_extensions.Type]]


   .. py:attribute:: introspector
      :type:  krrood.class_diagrams.attribute_introspector.AttributeIntrospector


   .. py:method:: get_associations_with_condition(clazz: typing_extensions.Union[typing_extensions.Type, WrappedClass], condition: typing_extensions.Callable[[Association], bool]) -> typing_extensions.Iterable[Association]

      Get all associations that match the condition.

      :param clazz: The source class or wrapped class for which outgoing edges are to be retrieved.
      :param condition: The condition to filter relations by.



   .. py:method:: get_outgoing_relations(clazz: typing_extensions.Union[typing_extensions.Type, WrappedClass]) -> typing_extensions.Iterable[ClassRelation]

      Get all outgoing edge relations of the given class.

      :param clazz: The source class or wrapped class for which outgoing edges are to be retrieved.



   .. py:method:: get_common_role_taker_associations(cls1: typing_extensions.Union[typing_extensions.Type, WrappedClass], cls2: typing_extensions.Union[typing_extensions.Type, WrappedClass]) -> typing_extensions.Tuple[typing_extensions.Optional[HasRoleTaker], typing_extensions.Optional[HasRoleTaker]]

      Return pair of role-taker associations if both classes point to the same target.

      The method checks whether both classes have a HasRoleTaker association to the
      same target class and returns the matching associations, otherwise ``(None, None)``.



   .. py:method:: get_role_taker_associations_of_cls(cls: typing_extensions.Union[typing_extensions.Type, WrappedClass]) -> typing_extensions.Optional[HasRoleTaker]

      Return the role-taker association of a class if present.

      A role taker is a field that is a one-to-one relationship and is not optional.



   .. py:method:: get_neighbors_with_relation_type(cls: typing_extensions.Union[typing_extensions.Type, WrappedClass], relation_type: typing_extensions.Type[ClassRelation]) -> typing_extensions.Tuple[WrappedClass, Ellipsis]

      Return all neighbors of a class whose connecting edge matches the relation type.

      :param cls: The class or wrapped class for which neighbors are to be found.
      :param relation_type: The type of the relation to filter edges by.
      :return: A tuple containing the neighbors of the class, filtered by the specified relation type.



   .. py:method:: get_outgoing_neighbors_with_relation_type(cls: typing_extensions.Union[typing_extensions.Type, WrappedClass], relation_type: typing_extensions.Type[ClassRelation]) -> typing_extensions.Tuple[WrappedClass, Ellipsis]

      Caches and retrieves the outgoing neighbors of a given class with a specific relation type
      using the dependency graph.

      :param cls: The class or wrapped class for which outgoing neighbors are to be found.
          relation_type: The type of the relation to filter edges by.
      :return: A tuple containing the outgoing neighbors of the class, filtered by the specified relation type.
      :raises: Any exceptions raised internally by `find_successors_by_edge` or during class wrapping.



   .. py:method:: get_incoming_neighbors_with_relation_type(cls: typing_extensions.Union[typing_extensions.Type, WrappedClass], relation_type: typing_extensions.Type[ClassRelation]) -> typing_extensions.Tuple[WrappedClass, Ellipsis]


   .. py:method:: get_out_edges(cls: typing_extensions.Union[typing_extensions.Type, WrappedClass]) -> typing_extensions.Tuple[ClassRelation, Ellipsis]

      Caches and retrieves the outgoing edges (relations) for the provided class in a
      dependency graph.

      :param cls: The class or wrapped class for which outgoing edges are to be retrieved.
      :return: A tuple of outgoing edges (relations) associated with the provided class.



   .. py:property:: parent_map

      Build parent map from inheritance edges: child_idx -> set(parent_idx)



   .. py:method:: all_ancestors(node_idx: int) -> set[int]

      DFS to compute all ancestors for each node index



   .. py:method:: get_assoc_keys_by_source(include_field_name: bool = False) -> dict[int, set[tuple]]

      Fetches association keys grouped by their source from the internal dependency graph.

      This method traverses the edges of the dependency graph, identifies associations,
      and groups their keys by their source nodes. Optionally includes the field name
      of associations in the resulting keys.

      :include_field_name: Optional; If True, includes the field name in the
              association keys. Defaults to False.

      :return: A dictionary where the keys are source node identifiers (int), and the
          values are sets of tuples representing association keys.



   .. py:method:: to_subdiagram_without_inherited_associations(include_field_name: bool = False) -> ClassDiagram

      Return a new class diagram where association edges that are present on any
      ancestor of the source class are removed from descendants.

      Inheritance edges are preserved.



   .. py:method:: remove_edges(edges)

      Remove edges from the dependency graph



   .. py:property:: wrapped_classes

      Return all wrapped classes present in the diagram.



   .. py:property:: associations
      :type: typing_extensions.List[Association]


      Return all association relations present in the diagram.



   .. py:property:: inheritance_relations
      :type: typing_extensions.List[Inheritance]


      Return all inheritance relations present in the diagram.



   .. py:method:: get_wrapped_class(clazz: typing_extensions.Type) -> typing_extensions.Optional[WrappedClass]

      Gets the wrapped class corresponding to the provided class type.

      If the class type is already a WrappedClass, it will be returned as is. Otherwise, the
      method checks if the class type has an associated WrappedClass in the internal mapping
      and returns it if found.

      :param clazz : The class type to check or retrieve the associated WrappedClass.
      :return: The associated WrappedClass if it exists, None otherwise.



   .. py:method:: add_node(clazz: typing_extensions.Union[typing_extensions.Type, WrappedClass])

      Adds a new node to the dependency graph for the specified wrapped class.

      The method sets the position of the given wrapped class in the dependency graph,
      links it with the current class diagram, and updates the mapping of the underlying
      class to the wrapped class.

      :param clazz: The wrapped class object to be added to the dependency graph.



   .. py:method:: add_relation(relation: ClassRelation)

      Adds a relation to the internal dependency graph.

      The method establishes a directed edge in the graph between the source and
      target indices of the provided relation. This function is used to model
      dependencies among entities represented within the graph.

      :relation: The relation object that contains the source and target entities and
      encapsulates the relationship between them.



   .. py:method:: visualize(filename: str = 'class_diagram.pdf', title: str = 'Class Diagram', figsize: tuple = (35, 30), node_size: int = 7000, font_size: int = 25, layout: str = 'layered', edge_style: str = 'straight', **kwargs)

      Visualize the class diagram using rustworkx_utils.

      Creates a visual representation of the class diagram showing classes and their relationships.
      The diagram is saved as a PDF file.

      :param filename: Output filename for the visualization
      :param title: Title for the diagram
      :param figsize: Figure size as (width, height) tuple
      :param node_size: Size of the nodes in the visualization
      :param font_size: Font size for labels
      :param kwargs: Additional keyword arguments passed to RWXNode.visualize()



   .. py:method:: clear()


