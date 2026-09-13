"""
Layered probabilistic circuits backed by numpy.

The units of a circuit are grouped into layers that hold the parameters of all of their
nodes in contiguous arrays, the way the jax implementation in
:mod:`probabilistic_model.probabilistic_circuit.jax` does, while supporting the full set
of queries of the graph based implementation in
:mod:`probabilistic_model.probabilistic_circuit.rx`.
"""
