krrood.ormatic.custom_types
===========================

.. py:module:: krrood.ormatic.custom_types


Classes
-------

.. autoapisummary::

   krrood.ormatic.custom_types.TypeType


Module Contents
---------------

.. py:class:: TypeType(*args: Any, **kwargs: Any)

   Bases: :py:obj:`sqlalchemy.TypeDecorator`


   Type that casts fields that are of type `type` to their class name on serialization and converts the name
   to the class itself through the globals on load.


   .. py:attribute:: impl


   .. py:method:: process_bind_param(value: typing_extensions.Type, dialect)

      Receive a bound parameter value to be converted.

      Custom subclasses of :class:`_types.TypeDecorator` should override
      this method to provide custom behaviors for incoming data values.
      This method is called at **statement execution time** and is passed
      the literal Python data value which is to be associated with a bound
      parameter in the statement.

      The operation could be anything desired to perform custom
      behavior, such as transforming or serializing data.
      This could also be used as a hook for validating logic.

      :param value: Data to operate upon, of any type expected by
       this method in the subclass.  Can be ``None``.
      :param dialect: the :class:`.Dialect` in use.

      .. seealso::

          :ref:`types_typedecorator`

          :meth:`_types.TypeDecorator.process_result_value`




   .. py:method:: process_result_value(value: impl, dialect) -> typing_extensions.Optional[typing_extensions.Type]

      Receive a result-row column value to be converted.

      Custom subclasses of :class:`_types.TypeDecorator` should override
      this method to provide custom behaviors for data values
      being received in result rows coming from the database.
      This method is called at **result fetching time** and is passed
      the literal Python data value that's extracted from a database result
      row.

      The operation could be anything desired to perform custom
      behavior, such as transforming or deserializing data.

      :param value: Data to operate upon, of any type expected by
       this method in the subclass.  Can be ``None``.
      :param dialect: the :class:`.Dialect` in use.

      .. seealso::

          :ref:`types_typedecorator`

          :meth:`_types.TypeDecorator.process_bind_param`





