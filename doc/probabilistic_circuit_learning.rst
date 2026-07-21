Probabilistic Circuit Structural Learning
=========================================

This page documents the structural learning functionality added to the
JAX probabilistic circuit module.

The implemented functionality follows the pruning and growing approach
from:

Dang, M., Liu, A., and Van den Broeck, G.
"Sparse Probabilistic Circuits via Pruning and Growing."
NeurIPS 2022.


Overview
--------

The structural learning implementation introduces methods to modify the
structure of probabilistic circuits by removing unnecessary connections
and increasing the model capacity.

The new functionality consists of:

* Edge flow calculation.
* EFLOW pruning.
* GROW expansion.
* A combined pruning and growing pipeline.


Edge Flow Calculation
---------------------

The function
:func:`probabilistic_model.probabilistic_circuit.jax.learning.calculate_edge_flows`
computes the average probability flow through the edges of sparse sum layers.

These values are used to identify edges with low contribution to the
probabilistic circuit structure.


EFLOW Pruning
-------------

The function
:func:`probabilistic_model.probabilistic_circuit.jax.learning.prune_circuit_eflow`
removes edges with the lowest probability flow.

The pruning process:

#. Calculates edge flows using the input data.
#. Selects the edges with the lowest contribution.
#. Removes these edges from sparse sum layers.
#. Returns a new pruned probabilistic circuit.


GROW Expansion
--------------

The function
:func:`probabilistic_model.probabilistic_circuit.jax.learning.grow_circuit`
increases the number of nodes in sparse sum layers.

Selected nodes are duplicated and small random perturbations are added to
the duplicated weights. This allows the circuit to increase its capacity
while avoiding identical duplicated parameters.


Prune and Grow Pipeline
-----------------------

The function
:func:`probabilistic_model.probabilistic_circuit.jax.learning.prune_and_grow`
combines both operations.

The pipeline first applies EFLOW pruning to remove low-flow edges and then
applies GROW expansion to increase the structural capacity of the circuit.


Testing
-------

Tests for this functionality are implemented in:

.. code-block:: text

    test/probabilistic_model_test/test_jax/test_learning.py

The tests verify:

* Edge flow computation.
* Removal of low-flow edges during pruning.
* Increase of nodes during growing.
* The complete prune and grow pipeline.