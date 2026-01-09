krrood.entity_query_language.utils
==================================

.. py:module:: krrood.entity_query_language.utils


Attributes
----------

.. autoapisummary::

   krrood.entity_query_language.utils.six
   krrood.entity_query_language.utils.Source
   krrood.entity_query_language.utils.T
   krrood.entity_query_language.utils.Binding
   krrood.entity_query_language.utils.Stage


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.utils.IDGenerator


Functions
---------

.. autoapisummary::

   krrood.entity_query_language.utils.lazy_iterate_dicts
   krrood.entity_query_language.utils.generate_combinations
   krrood.entity_query_language.utils.generate_bindings
   krrood.entity_query_language.utils.filter_data
   krrood.entity_query_language.utils.make_list
   krrood.entity_query_language.utils.is_iterable
   krrood.entity_query_language.utils.make_tuple
   krrood.entity_query_language.utils.make_set
   krrood.entity_query_language.utils.chain_stages


Module Contents
---------------

.. py:data:: six
   :value: None


.. py:data:: Source
   :value: None


.. py:class:: IDGenerator

   A class that generates incrementing, unique IDs and caches them for every object this is called on.


.. py:function:: lazy_iterate_dicts(dict_of_iterables)

   Generator that yields dicts with one value from each iterable


.. py:function:: generate_combinations(generators_dict)

   Yield all combinations of generator values as keyword arguments


.. py:function:: generate_bindings(child_vars_items, sources)

   Yield keyword-argument dictionaries for child variables using a depth‑first
   backtracking strategy with early pruning.

   The input mirrors Variable._child_vars_.items(): a sequence of (name, var)
   pairs. Each yielded item is a mapping: name -> {var_id: value}.

   The function evaluates each child variable against the current partial
   binding "sources" so constraints can prune the search space early.
   A simple heuristic chooses an evaluation order that prefers already bound,
   indexed, or kwargs‑constrained variables first.


.. py:function:: filter_data(data, selected_indices)

.. py:function:: make_list(value: typing_extensions.Any) -> typing_extensions.List

   Make a list from a value.

   :param value: The value to make a list from.


.. py:function:: is_iterable(obj: typing_extensions.Any) -> bool

   Check if an object is iterable.

   :param obj: The object to check.


.. py:function:: make_tuple(value: typing_extensions.Any) -> typing_extensions.Any

   Make a tuple from a value.


.. py:function:: make_set(value: typing_extensions.Any) -> typing_extensions.Set

   Make a set from a value.

   :param value: The value to make a set from.


.. py:data:: T

.. py:data:: Binding

   A dictionary mapping variable IDs to values.


.. py:data:: Stage

   A function that accepts a binding and returns an iterator of bindings.


.. py:function:: chain_stages(stages: typing_extensions.List[Stage], initial: Binding) -> typing_extensions.Iterator[Binding]

   Chains a sequence of stages into a single pipeline.

   This function takes a list of computational stages and an initial binding, passing the
   result of each computation stage to the next one. It produces an iterator of bindings
   by applying each stage in sequence to the current binding.

   :param stages: List[Stage]: A list of stages where each stage is a callable that accepts
       a Binding and produces an iterator of bindings.
   :param initial: Binding: The initial binding to start the computation with.

   :return: Iterator[Binding]: An iterator over the bindings resulting from applying all
       stages in sequence.


