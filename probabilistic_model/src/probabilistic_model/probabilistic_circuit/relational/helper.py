from __future__ import annotations

import rustworkx
from sortedcontainers import SortedSet

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)
from random_events.variable import Variable


def find_lowest_product_nodes_that_model_variables(
    circuit: ProbabilisticCircuit, variables: SortedSet[Variable]
) -> list[ProductUnit]:
    """Find the lowest product nodes in the circuit that model all given variables.

    Traverses the circuit layer by layer from the leaves upward.  A product node
    is included only if it models every variable in ``variables`` and none of its
    ancestors already appears in the result (avoiding duplicates at higher layers).

    These nodes serve as the attachment points where a grounded exchangeable
    distribution is connected to the class circuit during ``ground``.

    :param circuit: The circuit to search.
    :param variables: The set of variables that every returned node must model.
    :return: The lowest-level product nodes that jointly cover all of ``variables``,
        with no node being an ancestor of another in the result.
    """
    found_nodes: list[ProductUnit] = []
    ancestor_indices: set[int] = set()
    for layer in reversed(circuit.layers):
        for node in layer:
            if not isinstance(node, ProductUnit):
                continue
            if node.index in ancestor_indices:
                continue
            if not variables.issubset(node.variables):
                continue
            found_nodes.append(node)
            ancestor_indices.add(node.index)
            ancestor_indices.update(rustworkx.ancestors(circuit.graph, node.index))
    return found_nodes
