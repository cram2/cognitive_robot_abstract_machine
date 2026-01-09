krrood.singleton
================

.. py:module:: krrood.singleton


Classes
-------

.. autoapisummary::

   krrood.singleton.SingletonMeta


Module Contents
---------------

.. py:class:: SingletonMeta

   Bases: :py:obj:`type`


   A metaclass for creating singleton classes.


   .. py:method:: clear_instance()

      Removes the single, stored instance of this class, allowing a new one
      to be created on the next call.



