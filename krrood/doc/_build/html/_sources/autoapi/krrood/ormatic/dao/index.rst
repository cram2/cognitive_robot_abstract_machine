krrood.ormatic.dao
==================

.. py:module:: krrood.ormatic.dao


Attributes
----------

.. autoapisummary::

   krrood.ormatic.dao.logger
   krrood.ormatic.dao.T
   krrood.ormatic.dao.WorkItemT
   krrood.ormatic.dao.InstanceDict
   krrood.ormatic.dao.InProgressDict


Classes
-------

.. autoapisummary::

   krrood.ormatic.dao.DataAccessObjectWorkItem
   krrood.ormatic.dao.DataAccessObjectState
   krrood.ormatic.dao.ToDataAccessObjectWorkItem
   krrood.ormatic.dao.ToDataAccessObjectState
   krrood.ormatic.dao.FromDataAccessObjectWorkItem
   krrood.ormatic.dao.FromDataAccessObjectState
   krrood.ormatic.dao.HasGeneric
   krrood.ormatic.dao.DataAccessObject
   krrood.ormatic.dao.AlternativeMapping


Functions
---------

.. autoapisummary::

   krrood.ormatic.dao.is_data_column
   krrood.ormatic.dao.get_dao_class
   krrood.ormatic.dao.get_alternative_mapping
   krrood.ormatic.dao.to_dao


Module Contents
---------------

.. py:data:: logger

.. py:data:: T

.. py:data:: WorkItemT

.. py:data:: InstanceDict

.. py:data:: InProgressDict

.. py:function:: is_data_column(column: sqlalchemy.Column) -> bool

   Check if a column contains data.

   :param column: The SQLAlchemy column to check.
   :return: True if it is a data column.


.. py:class:: DataAccessObjectWorkItem

   Bases: :py:obj:`abc.ABC`


   Abstract base class for conversion work items.


   .. py:attribute:: dao_instance
      :type:  DataAccessObject


.. py:class:: DataAccessObjectState

   Bases: :py:obj:`typing_extensions.Generic`\ [\ :py:obj:`WorkItemT`\ ], :py:obj:`abc.ABC`


   Abstract base class for conversion states.


   .. py:attribute:: memo
      :type:  InstanceDict

      Cache for converted objects to prevent duplicates and handle circular references.



   .. py:attribute:: work_items
      :type:  collections.deque[WorkItemT]

      Deque of work items to be processed.



   .. py:method:: push_work_item(*args: typing_extensions.Any, **kwargs: typing_extensions.Any) -> None
      :abstractmethod:


      Add a new work item to the processing queue.

      :param args: Positional arguments for the work item.
      :param kwargs: Keyword arguments for the work item.



   .. py:method:: has(source: typing_extensions.Any) -> bool

      Check if the given source object has already been converted.

      :param source: The object to check.
      :return: True if already converted.



   .. py:method:: get(source: typing_extensions.Any) -> typing_extensions.Optional[typing_extensions.Any]

      Get the converted object for the given source object.

      :param source: The source object.
      :return: The converted object if it exists.



   .. py:method:: register(source: typing_extensions.Any, target: typing_extensions.Any) -> None

      Register a conversion result in the memoization store.

      :param source: The source object.
      :param target: The conversion result.



   .. py:method:: pop(source: typing_extensions.Any) -> typing_extensions.Optional[typing_extensions.Any]

      Remove and return the conversion result for the given source object.

      :param source: The source object.
      :return: The conversion result if it existed.



.. py:class:: ToDataAccessObjectWorkItem

   Bases: :py:obj:`DataAccessObjectWorkItem`


   Work item for converting an object to a Data Access Object.


   .. py:attribute:: source_object
      :type:  typing_extensions.Any


   .. py:attribute:: alternative_base
      :type:  typing_extensions.Optional[typing_extensions.Type[DataAccessObject]]
      :value: None



.. py:class:: ToDataAccessObjectState

   Bases: :py:obj:`DataAccessObjectState`\ [\ :py:obj:`ToDataAccessObjectWorkItem`\ ]


   State for converting objects to Data Access Objects.


   .. py:attribute:: keep_alive
      :type:  InstanceDict

      Dictionary that prevents objects from being garbage collected.



   .. py:method:: push_work_item(source_object: typing_extensions.Any, dao_instance: DataAccessObject, alternative_base: typing_extensions.Optional[typing_extensions.Type[DataAccessObject]] = None)

      Add a new work item to the processing queue.

      :param source_object: The object being converted.
      :param dao_instance: The DAO instance being populated.
      :param alternative_base: Base class for alternative mapping, if any.



   .. py:method:: apply_alternative_mapping_if_needed(dao_clazz: typing_extensions.Type[DataAccessObject], source_object: typing_extensions.Any) -> typing_extensions.Any

      Apply an alternative mapping if the DAO class requires it.

      :param dao_clazz: The DAO class to check.
      :param source_object: The object being converted.
      :return: The source object or the result of alternative mapping.



   .. py:method:: register(source_object: typing_extensions.Any, dao_instance: DataAccessObject) -> None

      Register a partially built DAO in the memoization stores.

      :param source_object: The object being converted.
      :param dao_instance: The partially built DAO.



.. py:class:: FromDataAccessObjectWorkItem

   Bases: :py:obj:`DataAccessObjectWorkItem`


   Work item for converting a Data Access Object back to a domain object.


   .. py:attribute:: domain_object
      :type:  typing_extensions.Any


.. py:class:: FromDataAccessObjectState

   Bases: :py:obj:`DataAccessObjectState`\ [\ :py:obj:`FromDataAccessObjectWorkItem`\ ]


   State for converting Data Access Objects back to domain objects.


   .. py:attribute:: discovery_mode
      :type:  bool
      :value: False


      Whether the state is currently in discovery mode.



   .. py:attribute:: initialized_ids
      :type:  typing_extensions.Set[int]

      Set of DAO ids that have been fully initialized.



   .. py:attribute:: is_processing
      :type:  bool
      :value: False


      Whether the state is currently in the processing loop.



   .. py:attribute:: synthetic_parent_daos
      :type:  typing_extensions.Dict[typing_extensions.Tuple[int, typing_extensions.Type[DataAccessObject]], DataAccessObject]

      Cache for synthetic parent DAOs to maintain identity across discovery and filling phases.



   .. py:method:: is_initialized(dao_instance: DataAccessObject) -> bool

      Check if the given DAO instance has been fully initialized.

      :param dao_instance: The DAO instance to check.
      :return: True if fully initialized.



   .. py:method:: mark_initialized(dao_instance: DataAccessObject)

      Mark the given DAO instance as fully initialized.

      :param dao_instance: The DAO instance to mark.



   .. py:method:: push_work_item(dao_instance: DataAccessObject, domain_object: typing_extensions.Any)

      Add a new work item to the processing queue.

      :param dao_instance: The DAO instance being converted.
      :param domain_object: The domain object being populated.



   .. py:method:: allocate_and_memoize(dao_instance: DataAccessObject, original_clazz: typing_extensions.Type) -> typing_extensions.Any

      Allocate a new instance and store it in the memoization dictionary.

      :param dao_instance: The DAO instance to register.
      :param original_clazz: The domain class to instantiate.
      :return: The uninitialized domain object instance.



   .. py:method:: apply_circular_fixes(domain_object: typing_extensions.Any, circular_references: typing_extensions.Dict[str, typing_extensions.Any]) -> None

      Resolve circular references in the domain object.

      :param domain_object: The object to fix.
      :param circular_references: Mapping of attribute names to circular reference identifiers.



.. py:class:: HasGeneric

   Bases: :py:obj:`typing_extensions.Generic`\ [\ :py:obj:`T`\ ]


   Base class for classes that carry a generic type argument.


   .. py:method:: original_class() -> T
      :classmethod:


      Get the concrete generic argument.

      :return: The generic type argument.
      :raises NoGenericError: If no generic argument is found.



.. py:class:: DataAccessObject(*args, **kwargs)

   Bases: :py:obj:`HasGeneric`\ [\ :py:obj:`T`\ ]


   Base class for Data Access Objects (DAOs) providing bidirectional conversion between
   domain objects and SQLAlchemy models.

   This class automates the mapping between complex domain object graphs and relational
   database schemas using SQLAlchemy. It supports inheritance, circular references,
   and custom mappings via :class:`AlternativeMapping`.

   Conversion Directions
   ---------------------

   1. **Domain to DAO (to_dao)**:
      Converts a domain object into its DAO representation. It uses an iterative
      BFS approach with a queue of work items to traverse the object graph. New work items
      for nested relationships are added to the queue during processing, ensuring all
      reachable objects are converted while maintaining the BFS order.

   2. **DAO to Domain (from_dao)**:
      Converts a DAO back into a domain object. To handle the strict initialization
      requirements of ``dataclasses`` and the potential for circular references,
      it uses a Two-Pass Iterative Approach:

      - Phase 1: Discovery (DFS):
        Traverses the DAO graph to identify all reachable DAOs. For each DAO, it
        allocates an uninitialized domain object (using ``__new__``) and records
        the discovery order.
      - Phase 2: Filling (Bottom-Up):
        Processes the discovered objects in reverse order. By moving from leaves
        to roots, it ensures that child dependencies are fully initialized before
        they are passed to a parent's constructor (``__init__``).

   Handling Circular References
   ----------------------------

   Circular references are handled by separating object allocation from initialization.
   If a circular dependency is detected during Phase 2 of ``from_dao`` (i.e., a
   required dependency is not yet initialized), the converter identifies the
   cycle and applies a fix-up step after the parent's ``__init__`` has been called,
   using :meth:`DataAccessObject._apply_circular_fixes`.

   Alternative Mappings
   --------------------

   For domain objects that do not map 1:1 to a single DAO (e.g., those requiring
   special constructor logic or representing a view of multiple tables),
   :class:`AlternativeMapping` can be used. The converter recognizes these and
   delegates the creation of the domain object to the mapping's ``create_from_dao``
   method during the Filling Phase.



   .. py:method:: to_dao(source_object: T, state: typing_extensions.Optional[ToDataAccessObjectState] = None, register: bool = True) -> _DAO
      :classmethod:


      Convert an object to its Data Access Object.

      :param source_object: The object to convert.
      :param state: The conversion state.
      :param register: Whether to register the result in the memo.
      :return: The converted DAO instance.



   .. py:method:: uses_alternative_mapping(class_to_check: typing_extensions.Type) -> bool
      :classmethod:


      Check if a class uses an alternative mapping.

      :param class_to_check: The class to check.
      :return: True if alternative mapping is used.



   .. py:method:: fill_dao_default(source_object: T, state: ToDataAccessObjectState) -> None

      Populate the DAO instance from a source object.

      :param source_object: The source object.
      :param state: The conversion state.



   .. py:method:: fill_dao_if_subclass_of_alternative_mapping(source_object: T, alternative_base: typing_extensions.Type[DataAccessObject], state: ToDataAccessObjectState) -> None

      Populate the DAO instance for an alternatively mapped subclass.

      :param source_object: The source object.
      :param alternative_base: The base class using alternative mapping.
      :param state: The conversion state.



   .. py:method:: get_columns_from(source_object: typing_extensions.Any, columns: typing_extensions.Iterable[sqlalchemy.Column]) -> None

      Assign values from specified columns of a source object to the DAO.

      :param source_object: The source of column values.
      :param columns: The columns to copy.



   .. py:method:: fill_relationships_from(source_object: typing_extensions.Any, relationships: typing_extensions.Iterable[sqlalchemy.orm.RelationshipProperty], state: ToDataAccessObjectState) -> None

      Populate relationships from a source object.

      :param source_object: The source of relationship values.
      :param relationships: The relationships to process.
      :param state: The conversion state.



   .. py:method:: from_dao(state: typing_extensions.Optional[FromDataAccessObjectState] = None) -> T

      Convert the DAO back into a domain object instance.

      :param state: The conversion state.
      :return: The converted domain object.



.. py:class:: AlternativeMapping

   Bases: :py:obj:`HasGeneric`\ [\ :py:obj:`T`\ ], :py:obj:`abc.ABC`


   Base class for alternative mapping implementations.


   .. py:method:: to_dao(source_object: T, state: typing_extensions.Optional[ToDataAccessObjectState] = None) -> _DAO
      :classmethod:


      Resolve a source object to a DAO.

      :param source_object: The object to convert.
      :param state: The conversion state.
      :return: The converted DAO instance.



   .. py:method:: from_domain_object(obj: T) -> typing_extensions.Self
      :classmethod:

      :abstractmethod:


      Create this from a domain object.
      Do not create any DAOs here but the target DAO of `T`.
      The rest of the `to_dao` algorithm will process the fields of the created instance.

      :param obj: The source object.
      :return: A new instance of this mapping class.



   .. py:method:: to_domain_object() -> T
      :abstractmethod:


      Create a domain object from this instance.

      :return: The constructed domain object.



.. py:function:: get_dao_class(original_clazz: typing_extensions.Type) -> typing_extensions.Optional[typing_extensions.Type[DataAccessObject]]

   Retrieve the DAO class for a domain class.

   :param original_clazz: The domain class.
   :return: The corresponding DAO class or None.


.. py:function:: get_alternative_mapping(original_clazz: typing_extensions.Type) -> typing_extensions.Optional[typing_extensions.Type[AlternativeMapping]]

   Retrieve the alternative mapping for a domain class.

   :param original_clazz: The domain class.
   :return: The corresponding alternative mapping or None.


.. py:function:: to_dao(source_object: typing_extensions.Any, state: typing_extensions.Optional[ToDataAccessObjectState] = None) -> DataAccessObject

   Convert an object to its corresponding DAO.

   :param source_object: The object to convert.
   :param state: The conversion state.
   :return: The converted DAO instance.


