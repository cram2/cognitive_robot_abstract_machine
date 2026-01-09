krrood.ormatic.exceptions
=========================

.. py:module:: krrood.ormatic.exceptions


Exceptions
----------

.. autoapisummary::

   krrood.ormatic.exceptions.NoGenericError
   krrood.ormatic.exceptions.NoDAOFoundError
   krrood.ormatic.exceptions.NoDAOFoundDuringParsingError
   krrood.ormatic.exceptions.UnsupportedRelationshipError


Module Contents
---------------

.. py:exception:: NoGenericError

   Bases: :py:obj:`krrood.utils.DataclassException`, :py:obj:`TypeError`


   Exception raised when the original class for a DataAccessObject subclass cannot
   be determined.

   This exception is typically raised when a DataAccessObject subclass has not
   been parameterized properly, which prevents identifying the original class
   associated with it.


   .. py:attribute:: clazz
      :type:  typing_extensions.Type


.. py:exception:: NoDAOFoundError

   Bases: :py:obj:`krrood.utils.DataclassException`, :py:obj:`TypeError`


   Represents an error raised when no DAO (Data Access Object) class is found for a given class.

   This exception is typically used when an attempt to convert a class into a corresponding DAO fails.
   It provides information about the class and the DAO involved.


   .. py:attribute:: obj
      :type:  typing_extensions.Any

      The class that no dao was found for



.. py:exception:: NoDAOFoundDuringParsingError(obj: typing_extensions.Any, dao: typing_extensions.Type, relationship: sqlalchemy.orm.RelationshipProperty = None)

   Bases: :py:obj:`NoDAOFoundError`


   Represents an error raised when no DAO (Data Access Object) class is found for a given class.

   This exception is typically used when an attempt to convert a class into a corresponding DAO fails.
   It provides information about the class and the DAO involved.


   .. py:attribute:: dao
      :type:  typing_extensions.Type

      The DAO class that tried to convert the cls to a DAO if any.



   .. py:attribute:: relationship
      :type:  sqlalchemy.orm.RelationshipProperty

      The relationship that tried to create the DAO.



   .. py:attribute:: message
      :value: 'Class Uninferable does not have a DAO. This happened when trying to create a dao for...



.. py:exception:: UnsupportedRelationshipError

   Bases: :py:obj:`krrood.utils.DataclassException`, :py:obj:`ValueError`


   Raised when a relationship direction is not supported by the ORM mapping.

   This error indicates that the relationship configuration could not be
   interpreted into a domain mapping.


   .. py:attribute:: relationship
      :type:  sqlalchemy.orm.RelationshipProperty


