krrood.entity_query_language.enums
==================================

.. py:module:: krrood.entity_query_language.enums


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.enums.RDREdge
   krrood.entity_query_language.enums.InferMode
   krrood.entity_query_language.enums.EQLMode
   krrood.entity_query_language.enums.PredicateType


Module Contents
---------------

.. py:class:: RDREdge(*args, **kwds)

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


   .. py:attribute:: Refinement
      :value: 'except if'


      Refinement edge, the edge that represents the refinement of an incorrectly fired rule.



   .. py:attribute:: Alternative
      :value: 'else if'


      Alternative edge, the edge that represents the alternative to the rule that has not fired.



   .. py:attribute:: Next
      :value: 'also if'


      Next edge, the edge that represents the next rule to be evaluated.



   .. py:attribute:: Then
      :value: 'then'


      Then edge, the edge that represents the connection to the conclusion.



.. py:class:: InferMode(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   The infer mode of a predicate, whether to infer new relations or retrieve current relations.


   .. py:attribute:: Auto

      Inference is done automatically depending on the world state.



   .. py:attribute:: Always

      Inference is always performed.



   .. py:attribute:: Never

      Inference is never performed.



.. py:class:: EQLMode(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   The modes of an entity query.


   .. py:attribute:: Rule

      Means this is a Rule that infers new relations/instances.



   .. py:attribute:: Query

      Means this is a Query that searches for matches



.. py:class:: PredicateType(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   The type of a predicate.


   .. py:attribute:: SubClassOfPredicate

      The predicate is an instance of Predicate class.



   .. py:attribute:: DecoratedMethod

      The predicate is a method decorated with @predicate decorator.



