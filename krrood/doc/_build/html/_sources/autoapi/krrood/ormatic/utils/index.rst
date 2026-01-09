krrood.ormatic.utils
====================

.. py:module:: krrood.ormatic.utils


Attributes
----------

.. autoapisummary::

   krrood.ormatic.utils.T
   krrood.ormatic.utils.leaf_types


Classes
-------

.. autoapisummary::

   krrood.ormatic.utils.classproperty
   krrood.ormatic.utils.InheritanceStrategy


Functions
---------

.. autoapisummary::

   krrood.ormatic.utils.classes_of_module
   krrood.ormatic.utils.drop_database
   krrood.ormatic.utils.module_and_class_name
   krrood.ormatic.utils.is_direct_subclass
   krrood.ormatic.utils.get_classes_of_ormatic_interface
   krrood.ormatic.utils.create_engine


Module Contents
---------------

.. py:class:: classproperty(fget)

   A decorator that allows a class method to be accessed as a property.


   .. py:attribute:: fget


.. py:function:: classes_of_module(module: types.ModuleType) -> typing_extensions.List[typing_extensions.Type]

   Get all classes of a given module.

   :param module: The module to inspect.
   :return: All classes of the given module.


.. py:data:: T

.. py:data:: leaf_types

.. py:function:: drop_database(engine: sqlalchemy.Engine) -> None

    Drops all tables in the given database engine. This function removes foreign key
    constraints and tables in reverse dependency order to ensure that proper
    dropping of objects occurs without conflict. For MySQL/MariaDB, foreign key
   checks are disabled temporarily during the process.

    This method differs from sqlalchemy `MetaData.drop_all <https://docs.sqlalchemy.org/en/20/core/metadata.html#sqlalchemy.schema.MetaData.drop_all>`_\ such that databases containing cyclic
    backreferences are also droppable.

    :param engine: The SQLAlchemy Engine instance connected to the target database
        where tables will be dropped.
    :type engine: Engine
    :return: None


.. py:class:: InheritanceStrategy(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access:

     >>> Color.RED
     <Color.RED: 1>

   - value lookup:

     >>> Color(1)
     <Color.RED: 1>

   - name lookup:

     >>> Color['RED']
     <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.


   .. py:attribute:: JOINED
      :value: 'joined'



   .. py:attribute:: SINGLE
      :value: 'single'



.. py:function:: module_and_class_name(t: typing_extensions.Union[typing_extensions.Type, typing_extensions._SpecialForm]) -> str

.. py:function:: is_direct_subclass(cls: typing_extensions.Type, *bases: typing_extensions.Type) -> bool

   :param cls: The class to check.
   :param bases: The base classes to check against.

   :return: Whether 'cls' is directly derived from any of the given base classes or is the same class.


.. py:function:: get_classes_of_ormatic_interface(interface: types.ModuleType) -> typing_extensions.Tuple[typing_extensions.List[typing_extensions.Type], typing_extensions.List[typing_extensions.Type[krrood.ormatic.dao.AlternativeMapping]], typing_extensions.Dict]

   Get all classes and alternative mappings of an existing ormatic interface.

   :param interface: The ormatic interface to extract the information from.
   :return: A list of classes and a list of alternative mappings used in the interface.


.. py:function:: create_engine(url: typing_extensions.Union[str, sqlalchemy.URL], **kwargs: typing_extensions.Any) -> sqlalchemy.Engine

   Check https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine for more information.

   :param url: The database URL.
   :return: An SQLAlchemy engine that uses the JSON (de)serializer from KRROOD.


