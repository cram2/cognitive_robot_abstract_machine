Probabilistic Circuit Structural Learning
=========================================


Overview
--------

This page documents the structural learning functionality implemented
for the RX probabilistic circuit module.

The implementation follows the pruning and growing approach described in:

Dang, M., Liu, A., and Van den Broeck, G.
"Sparse Probabilistic Circuits via Pruning and Growing."
NeurIPS 2022.


Structural learning automatically modifies the structure of a
probabilistic circuit based on observed data.

The implemented operations are:

* Edge flow calculation.
* Edge pruning based on probability contribution.
* Node growing through SumUnit duplication.
* Complete prune-grow structural learning pipeline.



Edge Flow Calculation
---------------------

The edge flow calculation is implemented by:

.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.calculate_edge_flows


The edge flow represents the contribution of each connection to the
overall probability distribution.

These values are used to identify relevant and irrelevant parts of the
circuit. Edges with very low probability flow are considered candidates
for removal during pruning.



Edge Pruning
------------

The pruning operation is implemented by:

.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.prune_edges


The procedure is:

#. Calculate the probability flow of every weighted edge.
#. Rank edges according to their contribution.
#. Remove edges with the lowest flow values.
#. Normalize the remaining SumUnit weights.


Pruning reduces unnecessary structural complexity while preserving the
most important probability paths in the circuit.



Node Growing
------------

The growing operation is implemented by:

.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.grow_nodes


Growing increases the representational capacity of the probabilistic
circuit.

Selected SumUnits are duplicated while preserving their outgoing
structure. Small perturbations are added to the duplicated weights to
avoid creating identical components.


This allows the model to discover more expressive structures without
requiring a manually designed larger circuit.



Complete Learning Pipeline
--------------------------

The complete structural learning procedure is implemented by:

.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.sparse_probabilistic_circuit_learning


The algorithm alternates between pruning and growing steps:

#. Remove low-contribution connections.
#. Increase structural capacity by growing selected nodes.
#. Repeat the process for several iterations.


The resulting circuit is a more compact and expressive probabilistic
model adapted to the observed data.



When to Use Structural Learning
-------------------------------

Structural learning is useful when:

* The initial probabilistic circuit contains redundant structure.
* The manually designed architecture is too large or inefficient.
* The model should automatically adapt its structure to data.
* A compact probabilistic circuit with preserved modelling capacity is
  desired.



Interactive Demo
----------------

A Jupyter notebook demonstrating the complete structural learning
process is included as part of this documentation.