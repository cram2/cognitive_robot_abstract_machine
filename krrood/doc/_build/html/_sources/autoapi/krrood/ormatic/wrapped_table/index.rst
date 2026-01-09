krrood.ormatic.wrapped_table
============================

.. py:module:: krrood.ormatic.wrapped_table


Attributes
----------

.. autoapisummary::

   krrood.ormatic.wrapped_table.logger


Exceptions
----------

.. autoapisummary::

   krrood.ormatic.wrapped_table.WrappedTableNotFound


Classes
-------

.. autoapisummary::

   krrood.ormatic.wrapped_table.ColumnConstructor
   krrood.ormatic.wrapped_table.AssociationTable
   krrood.ormatic.wrapped_table.WrappedTable


Module Contents
---------------

.. py:data:: logger

.. py:exception:: WrappedTableNotFound

   Bases: :py:obj:`KeyError`


   Mapping key not found.


   .. py:attribute:: type_
      :type:  typing_extensions.Type


   .. py:attribute:: wrapped_field
      :type:  krrood.class_diagrams.wrapped_field.WrappedField


.. py:class:: ColumnConstructor

   Represents a column constructor that can be used to create a column in SQLAlchemy.


   .. py:attribute:: name
      :type:  str

      The name of the column.



   .. py:attribute:: type
      :type:  str

      The type of the column.
      Needs to be like "Mapped[<type>]".



   .. py:attribute:: constructor
      :type:  typing_extensions.Optional[str]
      :value: None


      The constructor call for sqlalchemy of the column.



.. py:class:: AssociationTable

   Represents an association table for many-to-many relationships in SQLAlchemy.


   .. py:attribute:: name
      :type:  str

      The name of the association table.



   .. py:attribute:: left_table_name
      :type:  str

      The name of the left (source) table.



   .. py:attribute:: left_foreign_key
      :type:  str

      The foreign key column name for the left table.



   .. py:attribute:: left_primary_key
      :type:  str

      The full primary key reference for the left table (e.g., 'TableName.primary_key').



   .. py:attribute:: right_table_name
      :type:  str

      The name of the right (target) table.



   .. py:attribute:: right_foreign_key
      :type:  str

      The foreign key column name for the right table.



   .. py:attribute:: right_primary_key
      :type:  str

      The full primary key reference for the right table (e.g., 'TableName.primary_key').



.. py:class:: WrappedTable

   A class that wraps a dataclass and contains all the information needed to create a SQLAlchemy table from it.


   .. py:attribute:: wrapped_clazz
      :type:  krrood.class_diagrams.class_diagram.WrappedClass

      The wrapped class that this table wraps.



   .. py:attribute:: ormatic
      :type:  krrood.ormatic.ormatic.ORMatic

      Reference to the ORMatic instance that created this WrappedTable.



   .. py:attribute:: builtin_columns
      :type:  typing_extensions.List[ColumnConstructor]
      :value: []


      List of columns that can be directly mapped using builtin types



   .. py:attribute:: custom_columns
      :type:  typing_extensions.List[ColumnConstructor]
      :value: []


      List for custom columns that need to by fully qualified as triple of (name, type, constructor)



   .. py:attribute:: foreign_keys
      :type:  typing_extensions.List[ColumnConstructor]
      :value: []


      List of columns that represent foreign keys as triple of (name, type, constructor)



   .. py:attribute:: relationships
      :type:  typing_extensions.List[ColumnConstructor]
      :value: []


      List of relationships that should be added to the table.



   .. py:attribute:: mapper_args
      :type:  typing_extensions.Dict[str, str]


   .. py:attribute:: primary_key_name
      :type:  str
      :value: 'database_id'


      The name of the primary key column.



   .. py:attribute:: polymorphic_on_name
      :type:  str
      :value: 'polymorphic_type'


      The name of the column that will be used to identify polymorphic identities if any present.



   .. py:attribute:: skip_fields
      :type:  typing_extensions.List[krrood.class_diagrams.wrapped_field.WrappedField]
      :value: []


      A list of fields that should be skipped when processing the dataclass.



   .. py:property:: primary_key


   .. py:property:: child_tables
      :type: typing_extensions.List[WrappedTable]



   .. py:property:: has_children
      :type: bool


      Indicate whether this table has subclasses in the generated DAO model.

      The check is performed in two simple steps:
      - Use the inheritance graph to determine direct children of this wrapped class.
      - Additionally, scan existing wrapped tables for any table that resolves this
        instance as its ``parent_table`` (covers alternative-mapping hierarchies).



   .. py:method:: create_mapper_args()


   .. py:property:: full_primary_key_name


   .. py:property:: tablename


   .. py:property:: parent_table
      :type: typing_extensions.Optional[WrappedTable]


      Resolve the parent DAO table for this table.

      This first tries to use a direct inheritance relation. If that is not
      available and this table is an alternative mapping, it resolves the
      parent through the original classes' inheritance and maps back to the
      correct DAO table.

      :return: The parent ``WrappedTable`` or ``None`` if there is no parent.



   .. py:property:: is_alternatively_mapped


   .. py:property:: fields
      :type: typing_extensions.List[krrood.class_diagrams.wrapped_field.WrappedField]


      :return: The list of fields specified only in this associated dataclass that should be mapped.



   .. py:method:: parse_fields()


   .. py:method:: parse_field(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField)

      Parses a given `WrappedField` and determines its type or relationship to create the
      appropriate column or define relationships in an ORM context.
      The method processes several
      types of fields, such as type types, built-in types, enumerations, one-to-one relationships,
      custom types, JSON containers, and one-to-many relationships.

      This creates the right information in the right place in the table definition to be read later by the jinja
      template.

      :param wrapped_field: An instance of `WrappedField` that contains metadata about the field
          such as its data type, whether it represents a built-in or user-defined type, or if it has
          specific ORM container properties.



   .. py:method:: create_builtin_column(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField)

      Creates a built-in column mapping for the given wrapped field. Depending on the
      properties of the `wrapped_field`, this function determines whether it's an enum,
      a built-in type, or requires additional imports. It then constructs appropriate
      column definitions and adds them to the respective list of database mappings.

      :param wrapped_field: The WrappedField instance representing the field
          to create a built-in column for.



   .. py:method:: create_type_type_column(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField)

      Create a column for a field of type `Type`.
      :param wrapped_field: The field to extract type information from.
      :return:



   .. py:method:: get_table_of_wrapped_field(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField) -> WrappedTable

      :param wrapped_field: The wrapped field to get the table for.
      :return: The wrapped table for the given wrapped field.



   .. py:method:: create_one_to_one_relationship(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField)

      Create a one-to-one relationship with using the given field.
      This adds a foreign key and a relationship to this table.

      :param wrapped_field: The field to get the information from.



   .. py:method:: create_one_to_many_relationship(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField)

      Creates a many-to-many relationship mapping for the given wrapped field using an association table.
      This allows multiple instances of the source table to reference the same instances of the target table.

      :param wrapped_field: The field for the many-to-many relationship.



   .. py:method:: create_json_column(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField)

      Create a column for a list-like of built-in values.

      :param wrapped_field: The field to extract the information from.



   .. py:method:: create_custom_type(wrapped_field: krrood.class_diagrams.wrapped_field.WrappedField)


   .. py:property:: base_class_name


