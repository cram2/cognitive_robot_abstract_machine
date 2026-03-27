"""
utils
=====
Internal utility functions for causal_circuit.py.

Covers circuit attachment, support event inspection, and SumUnit
normalisation checks.
"""

from __future__ import annotations

import copy
from typing import Any, Set

from scipy.special import logsumexp

from random_events.variable import Variable

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)


def attach_marginal_circuit(
    marginal_circuit: ProbabilisticCircuit,
    target_product: ProductUnit,
    target_circuit: ProbabilisticCircuit,
) -> None:
    """
    Attach the root of marginal_circuit as a child of target_product,
    constructing fresh nodes owned by target_circuit.

    marginal() and log_truncated_in_place() return flat circuits
    (SumUnit -> leaves, or a single leaf), so one level of recursion suffices.
    """
    root = marginal_circuit.root
    if isinstance(root, SumUnit):
        new_sum_unit = SumUnit(probabilistic_circuit=target_circuit)
        for child_log_weight, child_subcircuit in root.log_weighted_subcircuits:
            new_sum_unit.add_subcircuit(
                leaf(copy.deepcopy(child_subcircuit.distribution), target_circuit),
                child_log_weight,
            )
        target_product.add_subcircuit(new_sum_unit)
    else:
        target_product.add_subcircuit(
            leaf(copy.deepcopy(root.distribution), target_circuit)
        )


def variables_of_simple_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a SimpleEvent.

    A SimpleEvent is a VariableMap — directly iterable as a dict via .keys().
    Uses the public VariableMap.keys() API so this remains stable across
    random_events versions.
    """
    try:
        return set(support_event.keys())
    except AttributeError:
        return set()


def variables_of_composite_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a composite Event.

    A composite Event exposes .simple_sets, each of which is a SimpleEvent.
    Delegates per-SimpleEvent extraction to variables_of_simple_event.
    """
    variable_set: Set[Variable] = set()
    for simple_set in support_event.simple_sets:
        variable_set.update(variables_of_simple_event(simple_set))
    return variable_set


def variables_of_support_event(support_event: Any) -> Set[Variable]:
    """
    Dispatcher — routes to variables_of_simple_event or
    variables_of_composite_event based on the event type.
    """
    if hasattr(support_event, "simple_sets"):
        return variables_of_composite_event(support_event)
    return variables_of_simple_event(support_event)


def sum_unit_is_normalized(sum_unit: SumUnit, tolerance: float = 1e-6) -> bool:
    """
    Return True iff the SumUnit's log-weights sum to log(1) == 0.

    Uses logsumexp for numerical stability, matching SumUnit.normalize().
    """
    log_weights = sum_unit.log_weights
    if len(log_weights) == 0:
        return True
    return abs(float(logsumexp(log_weights))) < tolerance