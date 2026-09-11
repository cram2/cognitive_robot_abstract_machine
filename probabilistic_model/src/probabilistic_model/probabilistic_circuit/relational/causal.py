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
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, List, Optional

import pandas as pd
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import AnnotatedVariable
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
    SumUnit,
)
from random_events.variable import Variable

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match

logger = logging.getLogger(__name__)


@dataclass
class RelationalCausalCircuit:
    """
    Factory bridging ``RelationalProbabilisticCircuit`` grounding into ``CausalCircuit``
    construction, mirroring how the rest of the ``relational`` package bridges
    ``probabilistic_model`` and ``krrood``.
    """

    adjustment_region_count_warning_threshold: int = 1000
    """
    Warn rather than silently proceed when the Cartesian product of an adjustment set's
    leaf-region counts exceeds this.

    See :meth:`from_grounded_circuit`.
    """

    @staticmethod
    def resolve_variable(
        circuit: ProbabilisticCircuit, path: str | MappedVariable
    ) -> Variable:
        """
        Resolve a dotted access-path suffix to the Variable it names in a grounded
        circuit.

        Accepts either a variable's full runtime name (e.g.
        ``"SceneRoom.objects[0].type"``), just enough of its trailing access path to be
        unambiguous (e.g. ``"objects[0].type"``, or ``"chair_count()"`` for an
        aggregation latent), or an EQL attribute-access expression (e.g.
        ``variable(SceneRoom).objects[0].type``) built the same way a query builds its
        own field access -- so callers don't need to reconstruct the class-name
        prefixing convention grounding applies, or spell it out as a string at all.

        :param circuit: The grounded circuit to resolve the path against.
        :param path: The variable's full name, an unambiguous suffix of it, or an EQL
            attribute-access expression naming it.
        :return: The matching Variable.
        :raises VariableNotFoundError: If no variable's name matches.
        :raises AmbiguousVariablePathError: If more than one variable's name matches.
        """
        if isinstance(path, MappedVariable):
            path = path._name_
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

    def fit(
        self,
        relational_probabilistic_circuit: RelationalProbabilisticCircuit,
        instances: List[DataAccessObject],
        stratify_by: str,
        dataframe_from_parent: Optional[pd.DataFrame] = None,
    ) -> RelationalProbabilisticCircuit:
        """
        Fit ``relational_probabilistic_circuit`` support-deterministically over
        ``stratify_by``, the precondition ``CausalCircuit.verify_support_determinism``
        checks.

        Partitions the training dataframe by ``stratify_by``'s exact value and fits one
        sub-circuit per partition (see :meth:`_fit_stratified_class_circuit`), rather
        than the plain, unconstrained fit ``RelationalProbabilisticCircuit.fit``
        otherwise runs: every row sharing a value then ends up under one circuit branch
        by construction, instead of possibly split across sibling leaves.

        :param relational_probabilistic_circuit: The circuit to fit, in place.
        :param instances: Training instances; all must share the same DAO class.
        :param stratify_by: Name of the class-level dataframe column to partition the
            training data by -- an EQL attribute-access expression's own ``._name_``, or
            the equivalent dotted access-path string.
        :param dataframe_from_parent: Forwarded to
            ``RelationalProbabilisticCircuit.fit``.
        :return:``relational_probabilistic_circuit``, fitted, to allow chaining.
        """
        relational_probabilistic_circuit.class_circuit_builder = (
            lambda class_dataframe, variables: self._fit_stratified_class_circuit(
                class_dataframe, variables, stratify_by
            )
        )
        return relational_probabilistic_circuit.fit(
            instances, dataframe_from_parent=dataframe_from_parent
        )

    @staticmethod
    def _fit_stratified_class_circuit(
        class_dataframe: pd.DataFrame,
        variables: List[AnnotatedVariable],
        stratify_by: str,
    ) -> ProbabilisticCircuit:
        """
        Fit one class circuit per distinct value of ``stratify_by``, combined under a
        root ``SumUnit`` weighted by each value's relative frequency.

        Every row of one partition shares the same ``stratify_by`` value, so that
        variable's fitted distribution within each partition's sub-circuit is a single
        point by construction, regardless of how the sub-circuit's own induction
        subsequently splits on the remaining variables -- unlike fitting one
        unconstrained tree over the whole dataframe, where two rows sharing a value can
        still end up under different sibling leaves.

        :param class_dataframe: The full class-level training dataframe.
        :param variables: The variables inferred over the full dataframe, reused as the
            annotation (mean, standard deviation, split thresholds) for every
            partition's own fit.
        :param stratify_by: Name of the column to partition the dataframe by.
        :return: The combined circuit.
        """
        result = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=result)
        total_row_count = len(class_dataframe)
        for _, partition in class_dataframe.groupby(stratify_by):
            partition_circuit = JointProbabilityTree(annotated_variables=variables).fit(
                partition.reset_index(drop=True)
            )
            node_index_map = result.mount(partition_circuit.root)
            root.add_subcircuit(
                node_index_map[partition_circuit.root.index],
                math.log(len(partition) / total_row_count),
            )
        return result

    def ground(
        self,
        relational_probabilistic_circuit: RelationalProbabilisticCircuit,
        query: Match,
        causal_variables: List[Variable],
        effect_variables: List[Variable],
        adjustment_variables: Optional[List[Variable]] = None,
        grounding_mode: GroundingMode = GroundingMode.SAMPLED,
        trim_to_registered_variables: bool = False,
    ) -> CausalCircuit:
        """
        Ground a relational circuit for a query and wrap it as a ``CausalCircuit``.

        Convenience wrapper combining ``RelationalProbabilisticCircuit.ground`` with
        :meth:`from_grounded_circuit`; call them separately to build a ``CausalCircuit``
        from a circuit that is already grounded.

        :param relational_probabilistic_circuit: The fitted relational circuit to
            ground.
        :param query: The grounding query.
        :param causal_variables: Already-resolved cause variables to register. If you
            only have a name, ground ``relational_probabilistic_circuit`` for the same
            query yourself first, resolve the name against that circuit (see
            :meth:`resolve_variable`), and pass the result here -- a Variable's name
            identifies it regardless of which grounding produced it.
        :param effect_variables: Effect variables to register, same format.
        :param adjustment_variables: Backdoor-adjustment variables to register, same
            format. Defaults to none.
        :param grounding_mode: How to represent aggregation latents the query leaves
            undetermined. Defaults to :attr:`GroundingMode.SAMPLED`, which always
            succeeds; :attr:`GroundingMode.EXACT` gives reproducible, domain-covering
            regions but may fall back internally if its precondition isn't met. See
            :class:`~probabilistic_model.probabilistic_circuit.relational.rspn.GroundingMode`.
        :param trim_to_registered_variables: See :meth:`from_grounded_circuit`.
        :return: A verified, support-deterministic ``CausalCircuit`` over the grounded
            circuit.
        :raises SupportDeterminismVerificationResult: If the grounded circuit is not
            support-deterministic for ``causal_variables``.
        """
        grounded_circuit = relational_probabilistic_circuit.ground(
            query, grounding_mode
        )
        return self.from_grounded_circuit(
            grounded_circuit,
            causal_variables,
            effect_variables,
            adjustment_variables,
            trim_to_registered_variables,
        )

    def from_grounded_circuit(
        self,
        grounded_circuit: ProbabilisticCircuit,
        causal_variables: List[Variable],
        effect_variables: List[Variable],
        adjustment_variables: Optional[List[Variable]] = None,
        trim_to_registered_variables: bool = False,
    ) -> CausalCircuit:
        """
        Wrap an already-grounded circuit as a verified ``CausalCircuit``.

        Registering causes and effects is a postprocessing step over grounding, not a
        distinct way of grounding: any circuit whose undetermined aggregation latents
        were retained (the only way ``RelationalProbabilisticCircuit.ground`` grounds)
        can be wrapped this way, whether or not it was built with causal use in mind.

        :param grounded_circuit: The grounded circuit to wrap.
        :param causal_variables: Already-resolved cause variables to register. Resolve
            a name against ``grounded_circuit`` first (see :meth:`resolve_variable`) if
            you only have one.
        :param effect_variables: Effect variables to register, same format.
        :param adjustment_variables: Backdoor-adjustment variables to register, same
            format. Defaults to none.
        :param trim_to_registered_variables: Marginalize ``grounded_circuit`` down to
            exactly the union of ``causal_variables``, ``effect_variables`` and
            ``adjustment_variables`` before registering it, discarding every other
            variable grounding retained. Every check and query this class runs
            afterward reads only those variables, so the discarded ones cannot change
            the result -- marginalizing to a set that includes all of them is exact,
            not an approximation. It matters for cost, not correctness: on a class
            circuit fitted over many unrelated scalar and exchangeable variables, the
            joint support ``verify_support_determinism`` and `backdoor_adjustment`
            compute grows with all of them, not just the ones actually queried, so
            trimming first keeps that cost down to the registered variables alone.
            Defaults to ``False``, preserving every variable grounding retained.
        :return: A verified, support-deterministic ``CausalCircuit`` over
            ``grounded_circuit`` (or its trim, if requested).
        :raises SupportDeterminismVerificationResult: If ``grounded_circuit`` is not
            support-deterministic for ``causal_variables``.
        """
        adjustment_variables = adjustment_variables or []

        if trim_to_registered_variables:
            registered_variables = list(
                dict.fromkeys(
                    causal_variables + effect_variables + adjustment_variables
                )
            )
            grounded_circuit = grounded_circuit.restrict_to_variables(
                registered_variables
            )

        self._warn_if_adjustment_regions_are_expensive(
            grounded_circuit, adjustment_variables
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
        self,
        grounded_circuit: ProbabilisticCircuit,
        adjustment_variables: List[Variable],
    ) -> None:
        """
        Warn when registering ``adjustment_variables`` together would make
        ``CausalCircuit.backdoor_adjustment``'s Cartesian product over their leaf
        regions expensive, rather than waiting to discover this at query time.

        A relational adjustment variable's region count can grow with the training data
        under :attr:`~probabilistic_model.probabilistic_circuit.relational.rspn.GroundingMode.EXACT`,
        unlike :attr:`~probabilistic_model.probabilistic_circuit.relational.rspn.GroundingMode.SAMPLED`,
        whose region count is capped by ``monte_carlo_sample_count``. This is a
        best-effort diagnostic based on
        ``grounded_circuit``'s actual leaf-region counts, not a guarantee: it flags
        expensive adjustment sets regardless of how they were grounded.

        :param grounded_circuit: The grounded circuit to extract leaf-region counts
            from.
        :param adjustment_variables: The resolved adjustment variables.
        """
        if len(adjustment_variables) < 2:
            return
        region_counts = [
            len(grounded_circuit.marginal([variable]).leaves)
            for variable in adjustment_variables
        ]
        region_product = math.prod(region_counts)
        if region_product > self.adjustment_region_count_warning_threshold:
            logger.warning(
                "Adjustment set [%s] has a Cartesian product of %d leaf regions (%s), "
                "exceeding the configured threshold of %d; "
                "CausalCircuit.backdoor_adjustment's region-extraction cost scales "
                "with this.",
                ", ".join(variable.name for variable in adjustment_variables),
                region_product,
                " x ".join(str(count) for count in region_counts),
                self.adjustment_region_count_warning_threshold,
            )
