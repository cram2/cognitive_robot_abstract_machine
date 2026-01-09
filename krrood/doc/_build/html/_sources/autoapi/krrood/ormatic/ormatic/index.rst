krrood.ormatic.ormatic
======================

.. py:module:: krrood.ormatic.ormatic


Attributes
----------

.. autoapisummary::

   krrood.ormatic.ormatic.logger
   krrood.ormatic.ormatic.TypeMappingsType


Classes
-------

.. autoapisummary::

   krrood.ormatic.ormatic.AlternativelyMaps
   krrood.ormatic.ormatic.ORMatic


Module Contents
---------------

.. py:data:: logger

.. py:data:: TypeMappingsType

.. py:class:: AlternativelyMaps

   Bases: :py:obj:`krrood.class_diagrams.class_diagram.ClassRelation`


   Edge type that says that the source alternativly maps the target, e. g.
   `AlternativeMaps(source=PointMapping, target=Point)` means that PointMapping is the mapping for Point.


.. py:class:: ORMatic

   ORMatic is a tool for generating SQLAlchemy ORM models from a set of dataclasses.


   .. py:attribute:: class_dependency_graph
      :type:  krrood.class_diagrams.class_diagram.ClassDiagram

      The class diagram to add the orm for.



   .. py:attribute:: alternative_mappings
      :type:  typing_extensions.List[typing_extensions.Type[krrood.ormatic.dao.AlternativeMapping]]
      :value: []


      List of alternative mappings that should be used to map classes.



   .. py:attribute:: type_mappings
      :type:  TypeMappingsType

      A dict that maps classes to custom types that should be used to save the classes.
      They keys of the type mappings must be disjoint with the classes given..



   .. py:attribute:: inheritance_strategy
      :type:  krrood.ormatic.utils.InheritanceStrategy

      The inheritance strategy to use.



   .. py:attribute:: foreign_key_postfix
      :value: '_id'


      The postfix that will be added to foreign key columns (not the relationships).



   .. py:attribute:: imported_modules
      :type:  sortedcontainers.SortedSet[str]

      A set of modules that need to be imported.



   .. py:attribute:: type_annotation_map
      :type:  typing_extensions.Dict[str, str]

      The string version of type mappings that is used in jinja.



   .. py:attribute:: inheritance_graph
      :type:  rustworkx.PyDiGraph[int]
      :value: None


      A graph that represents the inheritance structure of the classes. Extracted from the class dependency graph.



   .. py:attribute:: wrapped_tables
      :type:  typing_extensions.Dict[krrood.class_diagrams.class_diagram.WrappedClass, krrood.ormatic.wrapped_table.WrappedTable]

      The wrapped tables instances for the SQLAlchemy conversion.



   .. py:attribute:: association_tables
      :type:  typing_extensions.List[krrood.ormatic.wrapped_table.AssociationTable]
      :value: []


      List of association tables for many-to-many relationships.



   .. py:property:: alternatively_maps_relations
      :type: typing_extensions.List[AlternativelyMaps]



   .. py:method:: get_alternative_mapping(wrapped_class: krrood.class_diagrams.class_diagram.WrappedClass) -> typing_extensions.Optional[krrood.class_diagrams.class_diagram.WrappedClass]

      Finds and returns an alternative mapping for the given wrapped class,
      if one exists, based on the relations specified in
      `alternatively_maps_relations`.

      :param wrapped_class: The wrapped class for which an alternative
          mapping is to be searched.
      :return: An alternate mapping of the type WrappedClass if found,
          otherwise None.



   .. py:method:: create_type_annotations_map()


   .. py:property:: wrapped_classes_in_topological_order
      :type: typing_extensions.List[krrood.class_diagrams.class_diagram.WrappedClass]


      :return: List of all tables in topological order.



   .. py:property:: mapped_classes
      :type: typing_extensions.List[typing_extensions.Type]



   .. py:method:: make_all_tables()


   .. py:method:: foreign_key_name(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField) -> str

      :return: A foreign key name for the given field.



   .. py:method:: to_sqlalchemy_file(file: typing_extensions.TextIO)

      Generate a Python file with SQLAlchemy declarative mappings from the ORMatic models.

      :param file: The file to write to



