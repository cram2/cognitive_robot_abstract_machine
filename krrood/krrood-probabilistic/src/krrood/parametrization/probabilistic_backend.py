"""
The backend that answers entity queries from a tractable probabilistic model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing_extensions import Iterable, Iterator, TypeVar

import random_events.variable
from random_events.product_algebra import Event


from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    OperationResult,
    ProbabilisticOperationResult,
)
from krrood.parametrization.causal_search import (
    AdjustmentCandidates,
    CandidateAdjustment,
    CauseEffectVariables,
    InterventionalRange,
    RegionRange,
    ScoredIntervention,
)
from krrood.parametrization.probabilistic_queries import (
    ProbabilisticQuery,
)
from krrood.entity_query_language.operators.aggregators import Average
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    EffectNotIdentified,
    NoLearnedCausalStructure,
    VariableNotInCausalGraph,
    NoCauseVariablesForRanking,
    NoCausesEffectConditionForCause,
    NoSolutionFound,
)
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.query import Entity

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from krrood.parametrization.exceptions import (
    BoundValuesNotSupported,
    DoRequiresCausalCircuitModel,
    MultipleEffectVariablesNotSupported,
)
from krrood.parametrization.model_registries import (
    ModelRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import (
    UnderspecifiedParameters,
    SelectedAttributesParameters,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphNode,
    NodeLevel,
    NodeReference,
)
from probabilistic_model.causal_structure.identification import (
    AdjustmentSelector,
    AdjustmentSet,
    DeclaredAdjustmentPolicy,
    InvalidDeclaredAdjustmentRefusal,
)
from probabilistic_model.causal_structure.learning import LearnedCausalStructure

from krrood.entity_query_language.backends import (
    GeneratedInstance,
    GenerativeBackend,
    InstanceConstruction,
)

T = TypeVar("T")

# %% probabilistic backend


@dataclass(kw_only=True)
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that samples instances from a tractable probabilistic model.

    A sample contradicting the match where the model does not cover it, such as in a
    property, is rejected, so fewer than :attr:`number_of_samples` may be generated.
    """

    model_registry: ModelRegistry = field(default_factory=FullyFactorizedRegistry)
    """
    A model registry that can be used to resolve match statements to probabilistic
    models.
    """

    number_of_samples: int = field(kw_only=True, default=50)
    """
    The number of samples to generate.

    This is only used if the query does not specify a limit.
    """

    causal_structures: dict[type, LearnedCausalStructure] = field(
        kw_only=True, default_factory=dict
    )
    """
    The causal structure learned for each class.

    It chooses the variables each cause of a query about the class adjusts for, or
    checks those marked ``confounder``, which a class without one uses unchecked.
    """

    declared_adjustment_policy: DeclaredAdjustmentPolicy = field(
        kw_only=True, default_factory=InvalidDeclaredAdjustmentRefusal
    )
    """
    What a causal query about a class in :attr:`causal_structures` does with the
    variables marked ``confounder`` when the learned structure finds them invalid.
    """

    def evaluate(self, *, expression: Evaluable) -> Iterable[T]:
        """
        Unlike
        :meth:`~krrood.entity_query_language.backends.GenerativeBackend.evaluate`, also
        answers a probabilistic query and a bare ``average(...)`` selection, each with
        its single resolved value.
        """
        if isinstance(expression, ProbabilisticQuery):
            yield expression._resolve_(model_registry=self.model_registry)
            return
        bare_average = self._bare_average_selection(expression=expression)
        if bare_average is not None:
            yield self._resolve_average(average=bare_average)
            return
        yield from super().evaluate(expression=expression)

    @staticmethod
    def _bare_average_selection(*, expression: Evaluable) -> Average | None:
        """
        :param expression: The expression being evaluated.
        :return: The :class:`~krrood.entity_query_language.operators.aggregators.Average`
            ``expression`` selects, if it is an otherwise-untouched
            :class:`~krrood.entity_query_language.query.query.Entity` selecting one
            (i.e. what ``average(x.A).evaluate(...)`` builds -- see
            :meth:`~krrood.entity_query_language.operators.aggregators.Aggregator.evaluate`)
            over an attribute chain, so it can be answered in closed form. ``None``
            otherwise, e.g. when the average is grouped, filtered, or over something
            other than an attribute.
        """
        if not isinstance(expression, Entity):
            return None
        aggregator = expression.selected_aggregator
        if not isinstance(aggregator, Average):
            return None
        if aggregator._leaf_attribute_ is None:
            return None
        if aggregator._distinct_:
            # The closed-form expectation cannot deduplicate values.
            return None
        if any(
            builder is not None
            for builder in (
                expression._where_builder_,
                expression._grouped_by_builder_,
                expression._having_builder_,
                expression._ordered_by_builder_,
            )
        ):
            return None
        return aggregator

    def _resolve_average(self, *, average: Average) -> float:
        """
        Resolve a bare ``average(...)`` selection to the exact expectation of its
        attribute, via ``ProbabilisticModel.expectation``, instead of sampling and
        averaging rows.

        :param average: The average aggregator to resolve.
        :return: The expectation of the averaged attribute.
        """
        attribute = average._leaf_attribute_
        parameters = SelectedAttributesParameters(attributes=(attribute,))
        model = self.model_registry.get_model(parameters=parameters)
        [random_event_variable] = parameters.variables.values()
        return model.expectation((random_event_variable,))[random_event_variable]

    def _generate(
        self, *, expression: Match[T], bound_values: Bindings
    ) -> Iterator[GeneratedInstance]:
        """
        Differs from
        :meth:`~krrood.entity_query_language.backends.GenerativeBackend._generate` in
        raising instead of keeping a bound value of a variable the match is given.

        :raises BoundValuesNotSupported: If `bound_values` binds a variable the match is
            given.
        """
        construction = InstanceConstruction(
            match=expression,
            rows=(row.bindings for row in self.sample_bindings(expression=expression)),
        )
        bound_variable_ids = construction.given_expressions.keys() & bound_values.keys()
        if bound_variable_ids:
            raise BoundValuesNotSupported(
                match=expression, bound_variable_ids=bound_variable_ids
            )
        # Reject instances contradicting a value the model does not cover.
        for instance in expression._select_satisfying_(
            instances=construction
        )._evaluate_natively_():
            yield construction.generated(instance=instance)

    def sample_bindings(
        self, *, expression: Match[T]
    ) -> Iterator[ProbabilisticOperationResult]:
        """
        Sample values for the variables of a match's pattern from the model the match
        resolves to.

        :param expression: The match whose pattern variables to sample.
        :return: The bindings of the pattern variables, one row per sample, each with
            its log-likelihood under the model, most likely first.
        :raises NoSolutionFound: If the match's conditions leave the model no solution.
        """
        parameters = UnderspecifiedParameters(statement=expression)
        model = self._resolve_model_to_sample(
            parameters=parameters, expression=expression
        )
        number_of_samples = (
            expression._get_expression_()._limit_ or self.number_of_samples
        )

        samples = model.sample(number_of_samples)
        log_likelihoods = model.log_likelihood(samples)
        most_likely_first = log_likelihoods.argsort()[::-1]
        for bindings, log_likelihood in zip(
            parameters.bindings_from_model_samples(
                variables=model.variables, samples=samples[most_likely_first]
            ),
            log_likelihoods[most_likely_first],
        ):
            yield ProbabilisticOperationResult(
                role_taker=OperationResult(bindings=bindings),
                log_likelihood=float(log_likelihood),
            )

    def compute_log_likelihood(
        self, *, expression: Match[T], bindings: Bindings
    ) -> ProbabilisticOperationResult:
        """
        Compute the log-likelihood of values for the variables of a match's pattern
        under the model :meth:`sample_bindings` samples them from.

        :param expression: The match whose pattern variables the bindings bind.
        :param bindings: The values of the pattern's variables, keyed by the identifier
            of the variable assigned to each attribute.
        :return: The bindings with their log-likelihood.
        :raises NoSolutionFound: If the match's conditions leave the model no solution.
        :raises ValueOutsideVariableDomain: If a value the bindings give is outside the
            domain of a variable of the model.
        """
        parameters = UnderspecifiedParameters(statement=expression)
        model = self._resolve_model_to_sample(
            parameters=parameters, expression=expression
        )
        [log_likelihood] = model.log_likelihood(
            parameters.model_sample_from_bindings(
                variables=model.variables, bindings=bindings
            )
        )
        return ProbabilisticOperationResult(
            role_taker=OperationResult(bindings=bindings),
            log_likelihood=float(log_likelihood),
        )

    def _resolve_model_to_sample(
        self, *, parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> ProbabilisticModel:
        """
        :param parameters: The parameters extracted from *expression*.
        :param expression: The match being evaluated.
        :return: The model of the match's class, conditioned and truncated on the
            match, and narrowed to the region of its primary cause if it searches one.
        :raises NoSolutionFound: If the match's conditions leave the model no solution.
        """
        self.choose_adjustment_sets(parameters=parameters, expression=expression)
        model = self.model_registry.get_model(parameters=parameters)

        if parameters.search_cause_variables:
            cause_effect = self._resolve_cause_and_effect_variables(
                parameters=parameters, expression=expression
            )
            if not isinstance(model, CausalCircuit):
                raise DoRequiresCausalCircuitModel(resolved_model=model)
            # Keep the primary cause's effect-truncated, region-narrowed circuit.
            primary = self._resolve_primary_intervention(
                model=model,
                cause_variables=cause_effect.cause_variables,
                effect_variable=cause_effect.effect_variable,
                effect_truncation_event=parameters.truncation_assignments_from_where_conditions,
                expression=expression,
                adjustment_variables_by_cause=cause_effect.adjustment_variables_by_cause,
            )
            truncated = parameters.apply_krrood_variable_truncation(
                model=primary.narrowed_circuit
            )
        else:
            # Same conditioning and truncation as distribution_of, without sampling.
            truncated = parameters.resolve_conditioned_and_truncated_model(model=model)

        if truncated is None:
            raise NoSolutionFound(expression=expression._get_expression_())
        return truncated

    @staticmethod
    def _resolve_cause_and_effect_variables(
        *, parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> CauseEffectVariables:
        """
        Resolve the cause candidates and the single effect variable a ``cause`` search
        optimizes for.

        :param parameters: The parameters extracted from *expression*.
        :param expression: The match being evaluated.
        :raises NoCausesEffectConditionForCause: If no ``causes_effect(...)`` condition
            declared an effect.
        :raises MultipleEffectVariablesNotSupported: If more than one effect variable
            was found.
        :return: The resolved cause candidates and effect variable.
        """
        if not parameters.effect_variables_from_causes_effect:
            raise NoCausesEffectConditionForCause(
                expression=expression._get_expression_()
            )
        if len(parameters.effect_variables_from_causes_effect) > 1:
            raise MultipleEffectVariablesNotSupported(
                variables=parameters.effect_variables_from_causes_effect
            )
        [effect_variable] = parameters.effect_variables_from_causes_effect
        return CauseEffectVariables(
            cause_variables=parameters.search_cause_variables,
            effect_variable=effect_variable,
            adjustment_variables_by_cause={
                cause_variable: parameters.adjustment_variables_of(
                    cause_variable=cause_variable
                )
                for cause_variable in parameters.search_cause_variables
            },
        )

    def choose_adjustment_sets(
        self, *, parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> None:
        """
        Let the causal structure learned for the queried class choose the variables each
        cause candidate adjusts for, or settle those marked ``confounder``.

        Does nothing for a class without a structure or a query without a cause.

        :param parameters: The parameters extracted from *expression*, which record the
            chosen variables.
        :param expression: The match being evaluated.
        :raises VariableNotInCausalGraph: If the cause or the effect has no node in the
            learned graph.
        :raises InvalidAdjustmentSet: If the variables marked ``confounder`` fail the
            adjustment criterion in the learned graph and
            :attr:`declared_adjustment_policy` refuses them.
        :raises EffectNotIdentified: If the learned graph identifies the effect by no
            set of the variables the query names; :meth:`interventional_range` answers
            an effect it identifies up to a few candidate sets.
        """
        structure = self.causal_structures.get(parameters.queried_class)
        if structure is None or not parameters.search_cause_variables:
            return
        cause_effect = self._resolve_cause_and_effect_variables(
            parameters=parameters, expression=expression
        )
        selector = structure.selector
        for cause_variable in cause_effect.cause_variables:
            candidates = self.adjustment_candidates(
                parameters=parameters,
                structure=structure,
                selector=selector,
                cause_variable=cause_variable,
                effect_variable=cause_effect.effect_variable,
            )
            if not candidates.identification.is_identified:
                raise EffectNotIdentified(
                    cause=cause_variable,
                    effect=cause_effect.effect_variable,
                    identification=candidates.identification,
                )
            parameters.derived_adjustment_variables[cause_variable] = (
                candidates.adjustments[0].variables
            )

    def possible_causes(
        self, *, parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> list[random_events.variable.Variable]:
        """
        :param parameters: The parameters of a query about a class with a learned
            structure, with a ``causes_effect(...)`` condition and no ``cause``.
        :param expression: The query.
        :return: The variables the query names whose node is a possible ancestor of the
            effect's in the structure learned for the queried class, as the cause
            candidates to rank.
        """
        effect_variable = self._resolve_cause_and_effect_variables(
            parameters=parameters, expression=expression
        ).effect_variable
        graph = self.causal_structures[parameters.queried_class].selector.graph
        effect = self._reference_in(graph=graph, variable=effect_variable)
        ancestors = graph.possible_ancestors(effect.node)
        return [
            variable
            for name, variable in parameters.variables.items()
            if (reference := graph.reference_of(name)) is not None
            and reference.node in ancestors
            and reference.part_index in (None, effect.part_index)
        ]

    def adjustment_candidates(
        self,
        *,
        parameters: UnderspecifiedParameters,
        structure: LearnedCausalStructure,
        selector: AdjustmentSelector,
        cause_variable: random_events.variable.Variable,
        effect_variable: random_events.variable.Variable,
    ) -> AdjustmentCandidates:
        """
        Ask the learned structure what intervening on one cause adjusts for, among the
        variables the query names and the aggregation statistics a relational grounding
        keeps. Variables marked ``confounder`` are settled by
        :attr:`declared_adjustment_policy`; a chosen statistic the query lacks is added.

        :param parameters: The query's parameters.
        :param structure: The structure learned for the queried class.
        :param selector: The structure's selector, built from its causal relations as
            they are now.
        :param cause_variable: The cause.
        :param effect_variable: The effect.
        :return: What the structure says, with the variables of every candidate set.
        :raises VariableNotInCausalGraph: If the cause, the effect or a variable marked
            ``confounder`` has no node in the learned graph.
        """
        cause = self._reference_in(graph=selector.graph, variable=cause_variable)
        effect = self._reference_in(graph=selector.graph, variable=effect_variable)
        eligible = self._eligible_references(
            graph=selector.graph,
            parameters=parameters,
            part_index=cause.part_index
            if cause.part_index is not None
            else effect.part_index,
        )
        declared = AdjustmentSet(
            nodes=frozenset(
                self._reference_in(graph=selector.graph, variable=variable).node
                for variable in parameters.search_confounder_variables
            )
        )
        identification = (
            self.declared_adjustment_policy.identify(
                selector,
                cause.node,
                effect.node,
                declared,
                set(eligible),
                structure.training_rows,
            )
            if declared.nodes
            else selector.identify(
                cause.node, effect.node, set(eligible), structure.training_rows
            )
        )
        return AdjustmentCandidates(
            cause_variable=cause_variable,
            identification=identification,
            adjustments=[
                CandidateAdjustment(
                    variables=[
                        parameters.variables.setdefault(
                            eligible[node].variable.name, eligible[node].variable
                        )
                        for node in sorted(adjustment.nodes)
                    ]
                )
                for adjustment in identification.candidate_adjustments
            ],
        )

    def interventional_range(
        self, *, expression: Match[T]
    ) -> list[InterventionalRange]:
        """
        Answer a causal query whose effect the learned causal structure identifies only
        up to a few candidate adjustment sets: for every cause, the lowest and highest
        probability of the effect per region of the cause across the candidates
        (Maathuis, Kalisch and Bühlmann, 2009), a single value for an identified effect.

        :param expression: A match with ``cause`` fields and a ``causes_effect(...)``
            condition.
        :return: The range of every cause.
        :raises NoLearnedCausalStructure: If :attr:`causal_structures` has no structure
            for the queried class.
        :raises EffectNotIdentified: If the structure leaves the effect of a cause
            unidentified, with no candidate sets.
        """
        parameters = UnderspecifiedParameters(statement=expression)
        structure = self.causal_structures.get(parameters.queried_class)
        if structure is None:
            raise NoLearnedCausalStructure(
                expression=expression._get_expression_(),
                queried_class=parameters.queried_class,
            )
        cause_effect = self._resolve_cause_and_effect_variables(
            parameters=parameters, expression=expression
        )
        selector = structure.selector
        candidates_by_cause = []
        for cause_variable in cause_effect.cause_variables:
            candidates = self.adjustment_candidates(
                parameters=parameters,
                structure=structure,
                selector=selector,
                cause_variable=cause_variable,
                effect_variable=cause_effect.effect_variable,
            )
            if not candidates.adjustments:
                raise EffectNotIdentified(
                    cause=cause_variable,
                    effect=cause_effect.effect_variable,
                    identification=candidates.identification,
                )
            parameters.derived_adjustment_variables[cause_variable] = (
                candidates.variables
            )
            candidates_by_cause.append(candidates)
        model = self.model_registry.get_model(parameters=parameters)
        if not isinstance(model, CausalCircuit):
            raise DoRequiresCausalCircuitModel(resolved_model=model)
        effect_event = parameters.truncation_assignments_from_where_conditions
        return [
            self._range_of(
                model=model,
                candidates=candidates,
                effect_variable=cause_effect.effect_variable,
                effect_event=effect_event,
            )
            for candidates in candidates_by_cause
        ]

    @classmethod
    def _range_of(
        cls,
        *,
        model: CausalCircuit,
        candidates: AdjustmentCandidates,
        effect_variable: random_events.variable.Variable,
        effect_event: Event,
    ) -> InterventionalRange:
        """
        :param model: The causal circuit.
        :param candidates: One cause and its candidate adjustment sets.
        :param effect_variable: The effect.
        :param effect_event: The effect's condition.
        :return: The cause's range per region.
        """
        includes_no_effect = candidates.identification.includes_no_effect
        interventional_circuits = [
            model.backdoor_adjustment(
                candidates.cause_variable, effect_variable, adjustment.variables
            )
            for adjustment in candidates.adjustments
        ]
        observational = cls._probability_within(
            circuit=model.probabilistic_circuit, region=None, effect_event=effect_event
        )
        regions = []
        for region in model._extract_disjoint_regions_for_variable(
            candidates.cause_variable
        ):
            probabilities = [
                cls._probability_within(
                    circuit=interventional,
                    region=region.event,
                    effect_event=effect_event,
                )
                for interventional in interventional_circuits
            ] + ([observational] if includes_no_effect else [])
            regions.append(
                RegionRange(
                    cause_region=region.event,
                    lowest_effect_probability=min(probabilities),
                    highest_effect_probability=max(probabilities),
                )
            )
        return InterventionalRange(
            cause_variable=candidates.cause_variable,
            candidate_adjustments=candidates.adjustments,
            includes_no_effect=includes_no_effect,
            regions=regions,
        )

    @staticmethod
    def _probability_within(
        *, circuit: ProbabilisticModel, region: Event | None, effect_event: Event
    ) -> float:
        """
        :param circuit: A joint circuit over the cause and the effect.
        :param region: The cause region to restrict to, or ``None`` for none.
        :param effect_event: The effect's condition.
        :return: The effect condition's probability within the region.
        """
        if region is not None:
            circuit = circuit.truncated(
                region.fill_missing_variables_pure(circuit.variables)
            ).distribution
            if circuit is None:
                return 0.0
        return float(
            circuit.probability(
                effect_event.fill_missing_variables_pure(circuit.variables)
            )
        )

    @staticmethod
    def _eligible_references(
        *,
        graph: CausalGraph,
        parameters: UnderspecifiedParameters,
        part_index: int | None,
    ) -> dict[GraphNode, NodeReference]:
        """
        :param graph: A learned causal graph.
        :param parameters: The parameters of a causal query.
        :param part_index: The position of the part the cause, or else the effect, sits
            on; ``None`` if neither is a part attribute.
        :return: Per node a cause may adjust for, what it stands for: the variables the
            query names, of no part or of that part, and every aggregation statistic.
        """
        named = (graph.reference_of(name) for name in parameters.variables)
        eligible = {
            reference.node: reference
            for reference in named
            if reference is not None and reference.part_index in (None, part_index)
        }
        for node in graph.nodes():
            if node.level is NodeLevel.AGGREGATION_STATISTIC:
                eligible.setdefault(node, NodeReference(node=node))
        return eligible

    @staticmethod
    def _reference_in(
        *, graph: CausalGraph, variable: random_events.variable.Variable
    ) -> NodeReference:
        """
        :param graph: A learned causal graph.
        :param variable: A query variable.
        :return: The node it stands for.
        :raises VariableNotInCausalGraph: If the graph has no node for it.
        """
        reference = graph.reference_of(variable.name)
        if reference is None:
            raise VariableNotInCausalGraph(variable=variable)
        return reference

    @classmethod
    def _resolve_primary_intervention(
        cls,
        *,
        model: CausalCircuit,
        cause_variables: list[random_events.variable.Variable],
        effect_variable: random_events.variable.Variable,
        effect_truncation_event: Event | None,
        expression: Match[T],
        adjustment_variables_by_cause: (
            dict[random_events.variable.Variable, list[random_events.variable.Variable]]
            | None
        ) = None,
    ) -> ScoredIntervention:
        """
        Search every candidate cause variable independently for the region whose
        intervention best explains the effect, and return the highest-scoring candidate
        as the primary cause. :meth:`rank_causes` gives every candidate's score.

        :param model: The causal circuit to search.
        :param cause_variables: The candidate cause variables, one or more.
        :param effect_variable: The declared effect variable.
        :param effect_truncation_event: The event the declared effect condition
            translates to, used to narrow each candidate's interventional joint to the
            effect before ranking its regions.
        :param expression: The match being evaluated, for error reporting.
        :param adjustment_variables_by_cause: Per candidate, the variables passed to
            ``backdoor_adjustment`` as its adjustment set; none if not given.
        :raises NoSolutionFound: If no candidate has a region with positive probability.
        :return: The highest-scoring candidate.
        """
        scored_interventions = cls._score_all_interventions(
            model=model,
            cause_variables=cause_variables,
            effect_variable=effect_variable,
            effect_truncation_event=effect_truncation_event,
            adjustment_variables_by_cause=adjustment_variables_by_cause,
        )
        if not scored_interventions:
            raise NoSolutionFound(expression=expression._get_expression_())
        return scored_interventions[0]

    @classmethod
    def _score_all_interventions(
        cls,
        *,
        model: CausalCircuit,
        cause_variables: list[random_events.variable.Variable],
        effect_variable: random_events.variable.Variable,
        effect_truncation_event: Event | None,
        adjustment_variables_by_cause: (
            dict[random_events.variable.Variable, list[random_events.variable.Variable]]
            | None
        ) = None,
    ) -> list[ScoredIntervention]:
        """
        Score every candidate cause variable independently, one at a time, for the
        region whose intervention best explains the effect.

        :param model: The causal circuit to search.
        :param cause_variables: The candidate cause variables, one or more.
        :param effect_variable: The declared effect variable.
        :param effect_truncation_event: The event the declared effect condition
            translates to, used to narrow each candidate's interventional joint to the
            effect before ranking its regions.
        :param adjustment_variables_by_cause: Per candidate, the variables passed to
            ``backdoor_adjustment`` as its adjustment set; none if not given.
        :return: Every candidate with a region of positive probability and a positive
            effect probability within it, highest-scoring first.
        """
        scored_interventions = [
            scored_intervention
            for cause_variable in cause_variables
            if (
                scored_intervention := cls._score_intervention(
                    model=model,
                    cause_variable=cause_variable,
                    effect_variable=effect_variable,
                    effect_truncation_event=effect_truncation_event,
                    adjustment_variables=(adjustment_variables_by_cause or {}).get(
                        cause_variable, []
                    ),
                )
            )
            is not None
        ]
        scored_interventions.sort(
            key=lambda candidate: candidate.effect_probability_given_region,
            reverse=True,
        )
        return scored_interventions

    def rank_causes(self, *, expression: Match[T]) -> list[ScoredIntervention]:
        """
        Rank every ``cause`` candidate in *expression* by how well its intervention
        explains the declared effect, with the same per-candidate search
        :meth:`_evaluate` picks its primary cause with. Each search adjusts for the
        candidate's own set (see :meth:`choose_adjustment_sets`).

        :param expression: A match with one or more ``cause`` fields and a
            ``causes_effect(...)`` condition.
        :raises NoCauseVariablesForRanking: If *expression* has no ``cause`` fields.
        :raises NoCausesEffectConditionForCause: If no ``causes_effect(...)`` condition
            declared an effect.
        :raises MultipleEffectVariablesNotSupported: If more than one effect variable
            was found.
        :raises DoRequiresCausalCircuitModel: If the resolved model is not a
            :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
        :return: Every scoreable candidate, ranked highest-scoring first.
        """
        parameters = UnderspecifiedParameters(statement=expression)
        if (
            not parameters.search_cause_variables
            and parameters.queried_class in self.causal_structures
        ):
            parameters.search_cause_variables.extend(
                self.possible_causes(parameters=parameters, expression=expression)
            )
        if not parameters.search_cause_variables:
            raise NoCauseVariablesForRanking(expression=expression._get_expression_())
        self.choose_adjustment_sets(parameters=parameters, expression=expression)
        model = self.model_registry.get_model(parameters=parameters)
        if not isinstance(model, CausalCircuit):
            raise DoRequiresCausalCircuitModel(resolved_model=model)
        cause_effect = self._resolve_cause_and_effect_variables(
            parameters=parameters, expression=expression
        )
        return self._score_all_interventions(
            model=model,
            cause_variables=cause_effect.cause_variables,
            effect_variable=cause_effect.effect_variable,
            effect_truncation_event=parameters.truncation_assignments_from_where_conditions,
            adjustment_variables_by_cause=cause_effect.adjustment_variables_by_cause,
        )

    @staticmethod
    def _score_intervention(
        *,
        model: CausalCircuit,
        cause_variable: random_events.variable.Variable,
        effect_variable: random_events.variable.Variable,
        effect_truncation_event: Event | None,
        adjustment_variables: Iterable[random_events.variable.Variable] = (),
    ) -> ScoredIntervention | None:
        """
        Find ``cause_variable``'s best region, searched on the effect-truncated circuit,
        and score it by ``P(effect | do(cause_variable in best_region))`` on the
        interventional joint restricted to that region alone, so the score is comparable
        across candidates whatever share of their domain the region covers.

        :param model: The causal circuit to search.
        :param cause_variable: The candidate cause variable.
        :param effect_variable: The declared effect variable.
        :param effect_truncation_event: The event the declared effect condition
            translates to.
        :param adjustment_variables: The variables passed to ``backdoor_adjustment`` as
            its adjustment set.
        :return: The scored candidate, or ``None`` if it has no region with positive
            probability, or the effect has zero probability within that region.
        """
        interventional = model.backdoor_adjustment(
            cause_variable, effect_variable, list(adjustment_variables)
        )

        # `.truncated()` mutates the event it is given, so each call needs a fresh copy.
        if effect_truncation_event:
            effect_truncated = interventional.truncated(
                copy.deepcopy(effect_truncation_event)
            ).distribution
        else:
            effect_truncated = interventional
        if effect_truncated is None:
            return None

        best_region = model._best_disjoint_region(cause_variable, effect_truncated)
        if best_region is None:
            return None

        region = interventional.truncated(
            best_region.fill_missing_variables_pure(interventional.variables)
        )
        if region.distribution is None or region.probability <= 0.0:
            return None

        if not effect_truncation_event:
            return ScoredIntervention(
                cause_variable=cause_variable,
                effect_probability_given_region=1.0,
                narrowed_circuit=region.distribution,
            )

        narrowed = region.distribution.truncated(copy.deepcopy(effect_truncation_event))
        if narrowed.distribution is None or narrowed.probability <= 0.0:
            return None
        return ScoredIntervention(
            cause_variable=cause_variable,
            effect_probability_given_region=float(narrowed.probability),
            narrowed_circuit=narrowed.distribution,
        )
