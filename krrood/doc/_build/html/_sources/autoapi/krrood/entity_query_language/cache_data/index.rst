krrood.entity_query_language.cache_data
=======================================

.. py:module:: krrood.entity_query_language.cache_data


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.cache_data.SeenSet
   krrood.entity_query_language.cache_data.ReEnterableLazyIterable


Module Contents
---------------

.. py:class:: SeenSet

   Coverage index for previously seen partial assignments.

   This replaces the linear scan with a trie-based index using a fixed key order.
   An assignment A is considered covered if there exists a stored constraint C
   such that C.items() is a subset of A.items().


   .. py:attribute:: keys
      :type:  tuple
      :value: ()



   .. py:attribute:: all_seen
      :type:  bool
      :value: False



   .. py:attribute:: constraints
      :type:  list
      :value: []



   .. py:attribute:: exact
      :type:  set


   .. py:method:: add(assignment: typing_extensions.Dict) -> None

      Add a constraint (partial assignment) to the coverage index.



   .. py:method:: check(assignment: typing_extensions.Dict) -> bool

      Return True if any stored constraint is a subset of the given assignment.
      Mirrors previous semantics: encountering an empty assignment flips all_seen
      but returns False the first time to allow population.



   .. py:method:: exact_contains(assignment: typing_extensions.Dict) -> bool

      Return True if the assignment contains all cache keys and the exact key tuple
      exists in the cache. This is an O(1) membership test and does not consult
      the coverage trie.



   .. py:method:: clear()


.. py:class:: ReEnterableLazyIterable

   Bases: :py:obj:`typing_extensions.Generic`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   A wrapper for an iterable that allows multiple iterations over its elements,
   materializing values as they are iterated over.


   .. py:attribute:: iterable
      :type:  typing_extensions.Iterable[krrood.entity_query_language.utils.T]
      :value: []


      The iterable to wrap.



   .. py:attribute:: materialized_values
      :type:  typing_extensions.List[krrood.entity_query_language.utils.T]
      :value: []


      The materialized values of the iterable.



   .. py:method:: set_iterable(iterable)

      Set the iterable and wrap it in a generator.

      This is needed because of the weakref data we get from SymbolGraph. If we do `self.iterable = iterable` and
      weakref instances die, the iterable would have None values for them. But if we wrap it in a generator,
      they are actually removed, and the generator doesn't find them, which is the wanted behavior.



