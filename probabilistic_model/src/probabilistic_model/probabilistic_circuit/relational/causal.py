"""
Bridges ``RelationalProbabilisticCircuit`` grounding into ``CausalCircuit``
construction.

.. note::
    This module mirrors how ``rspn.py`` deliberately bridges ``probabilistic_model``
    and ``krrood``: it is the seam where relational grounding meets exact causal
    inference, kept separate from both ``rspn.py`` and ``causal_circuit.py`` so
    neither needs to depend on the other.
"""

from __future__ import annotations

import logging
import math
from typing_extensions import TYPE_CHECKING, List, Optional, Union

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    AmbiguousVariablePathError,
    VariableNotFoundError,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    GroundingMode,
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.variable import Variable

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match

logger = logging.getLogger(__name__)

DEFAULT_ADJUSTMENT_REGION_COUNT_WARNING_THRESHOLD = 1000
"""
Default value of ``adjustment_region_count_warning_threshold``.

See :func:`ground_relational_causal_circuit`.
"""


def resolve_variable(circuit: ProbabilisticCircuit, path: str) -> Variable:
    """
    Resolve a dotted access-path suffix to the Variable it names in a grounded circuit.

    Accepts either a variable's full runtime name (e.g. ``"SceneRoom.objects[0].type"``)
    or just enough of its trailing access path to be unambiguous (e.g.
    ``"objects[0].type"``, or ``"chair_count()"`` for an aggregation latent), so callers
    don't need to reconstruct the class-name prefixing convention grounding applies.

    :param circuit: The grounded circuit to resolve the path against.
    :param path: The variable's full name, or an unambiguous suffix of it.
    :return: The matching Variable.
    :raises VariableNotFoundError: If no variable's name matches.
    :raises AmbiguousVariablePathError: If more than one variable's name matches.
    """
    matches = [
        variable
        for variable in circuit.variables
        if variable.name == path or variable.name.endswith(f".{path}")
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise VariableNotFoundError(path, list(circuit.variables))
    raise AmbiguousVariablePathError(path, matches)


def _resolve_variables(
    circuit: ProbabilisticCircuit, variables: List[Union[Variable, str]]
) -> List[Variable]:
    """
    Resolve a mixed list of Variables and dotted access-path strings.

    :param circuit: The grounded circuit to resolve any path strings against.
    :param variables: Variables and/or dotted access-path strings.
    :return: The resolved Variables, in input order.
    """
    return [
        (
            variable
            if isinstance(variable, Variable)
            else resolve_variable(circuit, variable)
        )
        for variable in variables
    ]


def ground_relational_causal_circuit(
    relational_probabilistic_circuit: RelationalProbabilisticCircuit,
    query: Match,
    causal_variables: List[Union[Variable, str]],
    effect_variables: List[Union[Variable, str]],
    adjustment_variables: Optional[List[Union[Variable, str]]] = None,
    grounding_mode: GroundingMode = GroundingMode.CAUSAL_SAMPLED,
    adjustment_region_count_warning_threshold: int = DEFAULT_ADJUSTMENT_REGION_COUNT_WARNING_THRESHOLD,
) -> CausalCircuit:
    """
    Ground a relational circuit for a query and wrap it as a ``CausalCircuit``.

    :param relational_probabilistic_circuit: The fitted relational circuit to ground.
    :param query: The grounding query.
    :param causal_variables: Cause variables to register, as Variables or dotted
        access-path strings resolved against the grounded circuit (see
        :func:`resolve_variable`).
    :param effect_variables: Effect variables to register, same format.
    :param adjustment_variables: Backdoor-adjustment variables to register, same
        format. Defaults to none.
    :param grounding_mode: How to treat aggregation latents the query leaves
        undetermined. Defaults to :attr:`GroundingMode.CAUSAL_SAMPLED`, which always
        succeeds; :attr:`GroundingMode.CAUSAL_EXACT` gives reproducible,
        domain-covering regions but may fall back internally if its precondition isn't
        met. See :class:`~probabilistic_model.probabilistic_circuit.relational.rspn.GroundingMode`.
    :param adjustment_region_count_warning_threshold: Under
        :attr:`GroundingMode.CAUSAL_EXACT`, warn rather than silently proceed when the
        Cartesian product of ``adjustment_variables``' leaf-region counts exceeds
        this, since ``CausalCircuit.backdoor_adjustment``'s region-extraction cost
        scales with it. See :func:`_warn_if_adjustment_regions_are_expensive` for a
        caveat on what this warning does and does not detect.
    :return: A verified, support-deterministic ``CausalCircuit`` over the grounded
        circuit.
    :raises SupportDeterminismVerificationResult: If the grounded circuit is not
        support-deterministic for ``causal_variables``.
    """
    adjustment_variables = adjustment_variables or []
    grounded_circuit = relational_probabilistic_circuit.ground(query, grounding_mode)

    causal_variables = _resolve_variables(grounded_circuit, causal_variables)
    effect_variables = _resolve_variables(grounded_circuit, effect_variables)
    adjustment_variables = _resolve_variables(grounded_circuit, adjustment_variables)

    _warn_if_adjustment_regions_are_expensive(
        grounded_circuit,
        adjustment_variables,
        grounding_mode,
        adjustment_region_count_warning_threshold,
    )

    tree = MarginalDeterminismTreeNode.from_causal_graph(
        causal_variables, effect_variables
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        grounded_circuit, tree, causal_variables, effect_variables
    )
    causal_circuit.verify_support_determinism()
    return causal_circuit


def _warn_if_adjustment_regions_are_expensive(
    grounded_circuit: ProbabilisticCircuit,
    adjustment_variables: List[Variable],
    grounding_mode: GroundingMode,
    threshold: int,
) -> None:
    """
    Warn when registering ``adjustment_variables`` together would make
    ``CausalCircuit.backdoor_adjustment``'s Cartesian product over their leaf regions
    expensive, rather than waiting to discover this at query time.

    Only relevant when the caller *requested* :attr:`GroundingMode.CAUSAL_EXACT`: that
    mode grounds every value the model learned about, so a relational adjustment
    variable's region count can grow with the training data, unlike
    :attr:`GroundingMode.CAUSAL_SAMPLED`, whose region count is capped by
    ``monte_carlo_sample_count``. This is a best-effort diagnostic based on the
    requested mode, not a guarantee: individual exchangeable parts can still fall back
    to ``CAUSAL_SAMPLED`` internally (logged separately) when their own partition isn't
    disjoint, in which case this warning's premise may not hold for them.

    :param grounded_circuit: The grounded circuit to extract leaf-region counts from.
    :param adjustment_variables: The resolved adjustment variables.
    :param grounding_mode: The grounding mode requested for ``grounded_circuit``.
    :param threshold: Warn when the Cartesian product exceeds this.
    """
    if (
        grounding_mode is not GroundingMode.CAUSAL_EXACT
        or len(adjustment_variables) < 2
    ):
        return
    region_counts = [
        len(grounded_circuit.marginal([variable]).leaves)
        for variable in adjustment_variables
    ]
    region_product = math.prod(region_counts)
    if region_product > threshold:
        logger.warning(
            "Adjustment set [%s] has a Cartesian product of %d leaf regions (%s), "
            "exceeding the configured threshold of %d; "
            "CausalCircuit.backdoor_adjustment's region-extraction cost scales with "
            "this.",
            ", ".join(variable.name for variable in adjustment_variables),
            region_product,
            " x ".join(str(count) for count in region_counts),
            threshold,
        )
