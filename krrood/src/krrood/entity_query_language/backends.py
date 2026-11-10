import copy
import enum
import uuid
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from types import NoneType
from typing_extensions import ClassVar, Iterable, Iterator, TypeVar

import random_events.variable
from random_events.product_algebra import Event
from sqlalchemy.orm import sessionmaker

from krrood import logger
from krrood.entity_query_language.verbalization.vocabulary.english import Directive

from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    OperationResult,
    ProbabilisticOperationResult,
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.operators.causal import (
    AdjustmentCandidates,
    CandidateAdjustment,
    Cause,
    CauseEffectVariables,
    InterventionalRange,
    RegionRange,
    ScoredIntervention,
)
from krrood.entity_query_language.operators.probabilistic_queries import (
    ProbabilisticQuery,
)
from krrood.entity_query_language.operators.aggregators import Average
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    EffectNotIdentified,
    NoLearnedCausalStructure,
    VariableNotInCausalGraph,
    BackendCannotEvaluateCause,
    NoCauseVariablesForRanking,
    NoCausesEffectConditionForCause,
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
    SelectiveBackendCannotResolveEllipsisMatch,
    UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration,
)
from krrood.entity_query_language.factories import set_of, variable
from krrood.entity_query_language.query.match import Match, AttributeMatch
from krrood.entity_query_language.query.query import Entity, Query
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.ormatic.utils import classproperty

try:
    from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
        CausalCircuit,
    )
    from krrood.parametrization.exceptions import (
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
except ImportError as e:
    logger.debug(f"Couldn't import probabilistic model needed classes: {e}")
    CausalCircuit = NoneType
    DoRequiresCausalCircuitModel = NoneType
    MultipleEffectVariablesNotSupported = NoneType
    ModelRegistry = NoneType
    FullyFactorizedRegistry = NoneType
    SelectedAttributesParameters = NoneType
    UnderspecifiedParameters = NoneType
    ProbabilisticModel = NoneType
    CausalGraph = NoneType
    GraphNode = NoneType
    AdjustmentSet = NoneType
    NodeReference = NoneType
    NodeLevel = NoneType
    AdjustmentSelector = NoneType
    DeclaredAdjustmentPolicy = NoneType
    InvalidDeclaredAdjustmentRefusal = NoneType
    LearnedCausalStructure = NoneType

T = TypeVar("T")


@dataclass
class QueryBackend(ABC):
    """
    Base class for all query backends.

    Query backends are objects that answer queries by different means.
    """

    opening_directive: ClassVar[Directive | None] = None
    """
    The opening verb a verbalization uses when this backend evaluates the expression
    (``None`` keeps the query-type default).

    A backend declares its own performative so the verbalization layer never inspects
    concrete backend types.
    """

    raise_on_unresolvable_cause: bool = field(default=False, kw_only=True)
    """
    Whether to raise instead of warning when an expression contains a `Cause` (`cause`)
    intervention this backend cannot resolve causally.

    Defaults to ``False``: the `Cause` is then treated as an ordinary unspecified field
    (a warning is logged explaining why) rather than failing the query. Set ``True`` to
    fail loudly instead -- for example in tests that want to catch accidental `cause`
    misuse against a non-causal backend. Read only by :class:`SelectiveBackend` and
    :class:`EntityQueryLanguageGenerativeBackend`; :class:`ProbabilisticBackend` always
    raises when it cannot resolve a causal model, regardless of this flag.
    """

    @classproperty
    def reads_state(cls) -> bool:
        """
        Whether this backend's evaluations read the values of ground terms from the
        state they are given.
        """
        return False

    @abstractmethod
    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        """
        Generate answers that match the expression.

        :param expression: The expression to generate answers for.
        :return: An iterable of answers.
        """

    def _warn_or_raise_on_unresolved_cause_(self, expression: Evaluable) -> None:
        """
        Warn (or, if :attr:`raise_on_unresolvable_cause` is set, raise) when
        *expression* is a :class:`~krrood.entity_query_language.query.match.Match`
        containing a `Cause` this backend has no causal graph to resolve.

        :param expression: The expression about to be evaluated.
        """
        if not (isinstance(expression, Match) and expression._has_cause_attributes_):
            return
        if self.raise_on_unresolvable_cause:
            raise BackendCannotEvaluateCause(expression, backend_type=type(self))
        logger.warning(BackendCannotEvaluateCause(expression, backend_type=type(self)))


@dataclass
class SelectiveBackend(QueryBackend, ABC):
    """
    Selective backends are backends that select elements from existing data.

    These can take any query as input.
    """

    opening_directive: ClassVar[Directive | None] = Directive.FIND
    """
    Selecting from existing data reads as *"Find …"*.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if isinstance(expression, Match) and expression._has_ellipsis_attributes_:
            raise SelectiveBackendCannotResolveEllipsisMatch(expression)
        self._warn_or_raise_on_unresolved_cause_(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Evaluable) -> Iterable[T]: ...


@dataclass
class GenerativeBackend(QueryBackend, ABC):
    """
    Generative backends are backends that generate new elements.

    Generative backends have to take match expressions as input, since they need to construct new objects, and currently
    {py:class}`~krrood.entity_query_language.query.match.Match` is the only way to do so.
    """

    opening_directive: ClassVar[Directive | None] = Directive.GENERATE
    """
    Generating new elements reads as *"Generate …"*.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Match[T]) -> Iterable[T]: ...


@dataclass
class SQLAlchemyBackend(SelectiveBackend):
    """
    A backend that selects elements from a database that is available via SQLAlchemy.
    """

    session_maker: sessionmaker
    """
    The session maker used for the database interactions.
    """

    def _evaluate(self, expression: Query) -> Iterable:
        session = self.session_maker()
        translator = eql_to_sql(expression, session)
        yield from translator.evaluate()


@dataclass
class EntityQueryLanguageBackend(SelectiveBackend):
    """
    A backend that selects elements in this python process.

    This is just ordinary EQL: each expression evaluates itself natively (queries and matches both select over their domains).
    Constructing new instances is the job of a :class:`GenerativeBackend`.
    """

    @classproperty
    def reads_state(cls) -> bool:
        """
        Unlike :meth:`QueryBackend.reads_state`, holds: evaluation runs in this process.
        """
        return True

    def _evaluate(self, expression: Evaluable) -> Iterable:
        yield from expression._evaluate_natively_()


@dataclass
class EntityQueryLanguageGenerativeBackend(GenerativeBackend):
    """
    A generative backend that constructs new instances deterministically: it treats a
    match's unspecified leaves as variables, enumerates every combination over their
    (discrete) domains, constructs an instance per combination via the type's
    constructor, and keeps those that satisfy the match: the values its pattern states
    and its ``where`` conditions.
    """

    @classproperty
    def reads_state(cls) -> bool:
        """
        Unlike :meth:`QueryBackend.reads_state`, holds: generation selects in this
        process.
        """
        return True

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        self._warn_or_raise_on_unresolved_cause_(expression)
        enumerated_variables: dict[uuid.UUID, Selectable] = {}
        for attribute_match in expression._matches_with_variables_:
            self._check_attribute_match_is_suitable_for_generation(attribute_match)
            enumerated_variables[attribute_match.assigned_variable._id_] = (
                self._convert_attribute_match_to_variable(attribute_match)
            )

        instances = (
            expression._construct_instance_from_bindings_(bindings)
            for bindings in self._enumerate_bindings(enumerated_variables)
        )
        satisfying = expression._select_satisfying_(instances)
        yield from satisfying._quantify_(
            expression._quantifier_type_
        )._evaluate_natively_()

    @staticmethod
    def _check_attribute_match_is_suitable_for_generation(
        attribute_match: AttributeMatch,
    ) -> None:
        """
        Raise if an assignment in the match cannot be used to generate solutions.

        :param attribute_match: The attribute match to check.
        :raises UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration: If a
            non-enum leaf is left fully unspecified (``...`` or ``cause``), which
            deterministic generation cannot enumerate (use the
            :class:`ProbabilisticBackend` instead).
        """
        if isinstance(
            attribute_match.assigned_value, (type(Ellipsis), Cause)
        ) and not issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            raise UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration(
                attribute_match
            )

    @staticmethod
    def _convert_attribute_match_to_variable(
        attribute_match: AttributeMatch,
    ) -> Selectable:
        """
        Convert an attribute match into a variable to enumerate, handling ellipsis (and,
        identically, ``cause``) assignments for enum fields and concrete values.

        :param attribute_match: The attribute match to convert.
        :return: A variable (or symbolic expression) representing the attribute match.
        """
        if isinstance(
            attribute_match.assigned_value, (type(Ellipsis), Cause)
        ) and issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            return variable(
                attribute_match.assigned_variable._type_,
                list(attribute_match.assigned_variable._type_),
            )
        if isinstance(attribute_match.assigned_value, SymbolicExpression):
            return attribute_match.assigned_value
        return variable(
            type(attribute_match.assigned_value),
            [attribute_match.assigned_value],
        )

    @staticmethod
    def _enumerate_bindings(
        enumerated_variables: dict[uuid.UUID, Selectable],
    ) -> Iterator[Bindings]:
        """
        :param enumerated_variables: The variable whose values each pattern variable
            takes, keyed by the identifier of that pattern variable.
        :return: The bindings of the pattern variables, one per combination of the
            enumerated values.
        """
        for combination in set_of(*enumerated_variables.values())._evaluate_natively_():
            yield {
                pattern_variable_id: combination[enumerated_variable]
                for pattern_variable_id, enumerated_variable in enumerated_variables.items()
            }


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model using a model
    registry.

    A sampled instance contradicting the match, in a value the model does not cover such
    as a property's, is rejected, so fewer instances than :attr:`number_of_samples` may
    be generated.
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
    The causal structure learned for each class, which chooses the variables each cause
    of a causal query about the class adjusts for, or checks those marked
    ``confounder``.

    A query about a class without one adjusts for the variables marked ``confounder``
    alone, unchecked.
    """

    declared_adjustment_policy: DeclaredAdjustmentPolicy = field(
        kw_only=True, default_factory=InvalidDeclaredAdjustmentRefusal
    )
    """
    What a causal query about a class in :attr:`causal_structures` does with the
    variables marked ``confounder`` when the learned structure finds them invalid.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if isinstance(expression, ProbabilisticQuery):
            yield expression._resolve_(self.model_registry)
            return
        bare_average = self._bare_average_selection(expression)
        if bare_average is not None:
            yield self._resolve_average(bare_average)
            return
        yield from super().evaluate(expression)

    @staticmethod
    def _bare_average_selection(expression: Evaluable) -> Average | None:
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
            # native evaluation deduplicates values before averaging; the closed-form
            # expectation has no notion of that, so it must not silently stand in
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

    def _resolve_average(self, average: Average) -> float:
        """
        Resolve a bare ``average(...)`` selection to the exact expectation of its
        attribute, via ``ProbabilisticModel.expectation``, instead of sampling and
        averaging rows.

        :param average: The average aggregator to resolve.
        :return: The expectation of the averaged attribute.
        """
        attribute = average._leaf_attribute_
        parameters = SelectedAttributesParameters((attribute,))
        model = self.model_registry.get_model(parameters)
        [random_event_variable] = parameters.variables.values()
        return model.expectation((random_event_variable,))[random_event_variable]

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        # create new objects with the values from the samples, and reject those
        # contradicting a value the model does not cover, such as a property's
        instances = (
            expression._construct_instance_from_bindings_(row.bindings)
            for row in self.sample_bindings(expression)
        )
        yield from expression._select_satisfying_(instances)._evaluate_natively_()

    def sample_bindings(
        self, expression: Match[T]
    ) -> Iterator[ProbabilisticOperationResult]:
        """
        Sample values for the variables of a match's pattern from the model the match
        resolves to.

        :param expression: The match whose pattern variables to sample.
        :return: The bindings of the pattern variables, one row per sample, each with
            its log-likelihood under the model, most likely first.
        :raises NoSolutionFound: If the match's conditions leave the model no solution.
        """
        parameters = UnderspecifiedParameters(expression)
        model = self._resolve_model_to_sample(parameters, expression)
        number_of_samples = (
            expression._get_expression_()._limit_ or self.number_of_samples
        )

        samples = model.sample(number_of_samples)
        log_likelihoods = model.log_likelihood(samples)
        most_likely_first = log_likelihoods.argsort()[::-1]
        for bindings, log_likelihood in zip(
            parameters.bindings_from_model_samples(
                model.variables, samples[most_likely_first]
            ),
            log_likelihoods[most_likely_first],
        ):
            yield ProbabilisticOperationResult(
                role_taker=OperationResult(bindings),
                log_likelihood=float(log_likelihood),
            )

    def compute_log_likelihood(
        self, expression: Match[T], bindings: Bindings
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
        parameters = UnderspecifiedParameters(expression)
        model = self._resolve_model_to_sample(parameters, expression)
        [log_likelihood] = model.log_likelihood(
            parameters.model_sample_from_bindings(model.variables, bindings)
        )
        return ProbabilisticOperationResult(
            role_taker=OperationResult(bindings),
            log_likelihood=float(log_likelihood),
        )

    def _resolve_model_to_sample(
        self, parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> ProbabilisticModel:
        """
        :param parameters: The parameters extracted from *expression*.
        :param expression: The match being evaluated.
        :return: The model of the match's class, conditioned and truncated on the
            match, and narrowed to the region of its primary cause if it searches one.
        :raises NoSolutionFound: If the match's conditions leave the model no solution.
        """
        self.choose_adjustment_sets(parameters, expression)
        model = self.model_registry.get_model(parameters)

        if parameters.search_cause_variables:
            cause_effect = self._resolve_cause_and_effect_variables(
                parameters, expression
            )
            if not isinstance(model, CausalCircuit):
                raise DoRequiresCausalCircuitModel(model)
            # search every candidate cause independently and keep the primary one's
            # already effect-truncated, already region-narrowed circuit -- see
            # _resolve_primary_intervention for why this needs no joint intervention
            primary = self._resolve_primary_intervention(
                model,
                cause_effect.cause_variables,
                cause_effect.effect_variable,
                parameters.truncation_assignments_from_where_conditions,
                expression,
                cause_effect.adjustment_variables_by_cause,
            )
            truncated = parameters.apply_krrood_variable_truncation(
                primary.narrowed_circuit
            )
        else:
            # apply literal-assignment conditions, then where-conditions, then
            # krrood-variable-assignment truncations -- see
            # UnderspecifiedParameters.resolve_conditioned_and_truncated_model, which
            # Distribution._resolve_ also reuses to answer a distribution(...) query
            # with exactly this same sequence, without the sampling step.
            truncated = parameters.resolve_conditioned_and_truncated_model(model)

        if truncated is None:
            raise NoSolutionFound(expression._get_expression_())
        return truncated

    @staticmethod
    def _resolve_cause_and_effect_variables(
        parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> CauseEffectVariables:
        """
        Resolve the cause candidates and the single effect variable a ``cause`` search
        optimizes for.

        Any number of cause candidates is fine -- each is searched independently (see
        :meth:`_resolve_primary_intervention`). Exactly one effect variable is required:
        :meth:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit.backdoor_adjustment`
        has no multi-effect form to route several through.

        :param parameters: The parameters extracted from *expression*.
        :param expression: The match being evaluated.
        :raises NoCausesEffectConditionForCause: If no ``causes_effect(...)`` condition
            declared an effect.
        :raises MultipleEffectVariablesNotSupported: If more than one effect variable
            was found.
        :return: The resolved cause candidates and effect variable.
        """
        if not parameters.effect_variables_from_causes_effect:
            raise NoCausesEffectConditionForCause(expression._get_expression_())
        if len(parameters.effect_variables_from_causes_effect) > 1:
            raise MultipleEffectVariablesNotSupported(
                parameters.effect_variables_from_causes_effect
            )
        [effect_variable] = parameters.effect_variables_from_causes_effect
        return CauseEffectVariables(
            parameters.search_cause_variables,
            effect_variable,
            {
                cause_variable: parameters.adjustment_variables_of(cause_variable)
                for cause_variable in parameters.search_cause_variables
            },
        )

    def choose_adjustment_sets(
        self, parameters: UnderspecifiedParameters, expression: Match[T]
    ) -> None:
        """
        Let the causal structure learned for the queried class choose, for every cause
        candidate, the variables its intervention adjusts for, or settle the variables
        marked ``confounder`` (see :meth:`adjustment_candidates`). Does nothing for a
        class without a structure in :attr:`causal_structures`, or a query without a
        ``cause``.

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
        cause_effect = self._resolve_cause_and_effect_variables(parameters, expression)
        selector = structure.selector
        for cause_variable in cause_effect.cause_variables:
            candidates = self.adjustment_candidates(
                parameters,
                structure,
                selector,
                cause_variable,
                cause_effect.effect_variable,
            )
            if not candidates.identification.is_identified:
                raise EffectNotIdentified(
                    cause_variable,
                    cause_effect.effect_variable,
                    candidates.identification,
                )
            parameters.derived_adjustment_variables[cause_variable] = (
                candidates.adjustments[0].variables
            )

    def possible_causes(
        self, parameters: UnderspecifiedParameters, expression: Match[T]
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
            parameters, expression
        ).effect_variable
        graph = self.causal_structures[parameters.queried_class].selector.graph
        effect = self._reference_in(graph, effect_variable)
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
        parameters: UnderspecifiedParameters,
        structure: LearnedCausalStructure,
        selector: AdjustmentSelector,
        cause_variable: random_events.variable.Variable,
        effect_variable: random_events.variable.Variable,
    ) -> AdjustmentCandidates:
        """
        Ask the learned structure what intervening on one cause adjusts for, among the
        variables the query names and the aggregation statistics a relational grounding
        keeps; the variables marked ``confounder`` are settled by
        :attr:`declared_adjustment_policy`. A chosen statistic the query does not name is
        added to its parameters' variables. A part attribute's node stands for the copy
        on the part the cause, or else the effect, sits on.

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
        cause = self._reference_in(selector.graph, cause_variable)
        effect = self._reference_in(selector.graph, effect_variable)
        eligible = self._eligible_references(
            selector.graph,
            parameters,
            cause.part_index if cause.part_index is not None else effect.part_index,
        )
        declared = AdjustmentSet(
            frozenset(
                self._reference_in(selector.graph, variable).node
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
                    [
                        parameters.variables.setdefault(
                            eligible[node].variable.name, eligible[node].variable
                        )
                        for node in sorted(adjustment.nodes)
                    ]
                )
                for adjustment in identification.candidate_adjustments
            ],
        )

    def interventional_range(self, expression: Match[T]) -> list[InterventionalRange]:
        """
        Answer a causal query whose effect the learned causal structure identifies only
        up to a few candidate adjustment sets: for every cause, the lowest and highest
        probability of the effect per region of the cause, across the candidates
        (Maathuis, Kalisch and Bühlmann, 2009). An identified effect has one candidate,
        so its range is a single value per region.

        :param expression: A match with ``cause`` fields and a ``causes_effect(...)``
            condition.
        :return: The range of every cause.
        :raises NoLearnedCausalStructure: If :attr:`causal_structures` has no structure
            for the queried class.
        :raises EffectNotIdentified: If the structure leaves the effect of a cause
            unidentified, with no candidate sets.
        """
        parameters = UnderspecifiedParameters(expression)
        structure = self.causal_structures.get(parameters.queried_class)
        if structure is None:
            raise NoLearnedCausalStructure(
                expression._get_expression_(), parameters.queried_class
            )
        cause_effect = self._resolve_cause_and_effect_variables(parameters, expression)
        selector = structure.selector
        candidates_by_cause = []
        for cause_variable in cause_effect.cause_variables:
            candidates = self.adjustment_candidates(
                parameters,
                structure,
                selector,
                cause_variable,
                cause_effect.effect_variable,
            )
            if not candidates.adjustments:
                raise EffectNotIdentified(
                    cause_variable,
                    cause_effect.effect_variable,
                    candidates.identification,
                )
            parameters.derived_adjustment_variables[cause_variable] = (
                candidates.variables
            )
            candidates_by_cause.append(candidates)
        model = self.model_registry.get_model(parameters)
        if not isinstance(model, CausalCircuit):
            raise DoRequiresCausalCircuitModel(model)
        effect_event = parameters.truncation_assignments_from_where_conditions
        return [
            self._range_of(
                model, candidates, cause_effect.effect_variable, effect_event
            )
            for candidates in candidates_by_cause
        ]

    @classmethod
    def _range_of(
        cls,
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
            model.probabilistic_circuit, None, effect_event
        )
        regions = []
        for region in model._extract_disjoint_regions_for_variable(
            candidates.cause_variable
        ):
            probabilities = [
                cls._probability_within(interventional, region.event, effect_event)
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
        circuit: ProbabilisticModel, region: Event | None, effect_event: Event
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
        graph: CausalGraph, parameters: UnderspecifiedParameters, part_index: int | None
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
                eligible.setdefault(node, NodeReference(node))
        return eligible

    @staticmethod
    def _reference_in(
        graph: CausalGraph, variable: random_events.variable.Variable
    ) -> NodeReference:
        """
        :param graph: A learned causal graph.
        :param variable: A query variable.
        :return: The node it stands for.
        :raises VariableNotInCausalGraph: If the graph has no node for it.
        """
        reference = graph.reference_of(variable.name)
        if reference is None:
            raise VariableNotInCausalGraph(variable)
        return reference

    @classmethod
    def _resolve_primary_intervention(
        cls,
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
        as the primary cause.

        This is the same per-candidate approach
        :meth:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit.diagnose_failure`
        already uses to identify a ``primary_cause_variable`` -- trying one candidate at
        a time needs no joint, multi-variable intervention, which
        ``backdoor_adjustment`` does not support. See :meth:`rank_causes` for every
        candidate's score, not just this one.

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
            model,
            cause_variables,
            effect_variable,
            effect_truncation_event,
            adjustment_variables_by_cause,
        )
        if not scored_interventions:
            raise NoSolutionFound(expression._get_expression_())
        return scored_interventions[0]

    @classmethod
    def _score_all_interventions(
        cls,
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
        Score every candidate cause variable independently for the region whose
        intervention best explains the effect.

        Shared by :meth:`_resolve_primary_intervention` (which keeps only the top
        result) and :meth:`rank_causes` (which keeps them all) -- trying one candidate
        at a time needs no joint, multi-variable intervention, which
        ``backdoor_adjustment`` does not support.

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
                    model,
                    cause_variable,
                    effect_variable,
                    effect_truncation_event,
                    (adjustment_variables_by_cause or {}).get(cause_variable, []),
                )
            )
            is not None
        ]
        scored_interventions.sort(
            key=lambda candidate: candidate.effect_probability_given_region,
            reverse=True,
        )
        return scored_interventions

    def rank_causes(self, expression: Match[T]) -> list[ScoredIntervention]:
        """
        Rank every ``cause`` candidate in *expression* by how well its intervention
        explains the declared effect.

        Runs the same per-candidate search :meth:`_evaluate` uses internally for a
        multi-``cause`` query, but returns every scoreable candidate instead of only
        the primary one :meth:`_evaluate` picks -- useful when several plausible causes
        exist and how they compare matters, not just which one wins (for example, both
        ``arm`` and ``force`` scoring high for a pick failure). Leaves
        :meth:`_evaluate` and the primary-cause search it uses entirely unchanged; this
        is an additional, independent read of the same candidates.

        Each candidate's search adjusts for its own set: the fields marked
        ``confounder``, or, with a structure in :attr:`causal_structures`, the set the learned
        structure chooses (see :meth:`choose_adjustment_sets`), so a variable that drives
        both a candidate and the effect does not inflate that candidate's score with mere
        correlation.

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
        parameters = UnderspecifiedParameters(expression)
        if (
            not parameters.search_cause_variables
            and parameters.queried_class in self.causal_structures
        ):
            parameters.search_cause_variables.extend(
                self.possible_causes(parameters, expression)
            )
        if not parameters.search_cause_variables:
            raise NoCauseVariablesForRanking(expression._get_expression_())
        self.choose_adjustment_sets(parameters, expression)
        model = self.model_registry.get_model(parameters)
        if not isinstance(model, CausalCircuit):
            raise DoRequiresCausalCircuitModel(model)
        cause_effect = self._resolve_cause_and_effect_variables(parameters, expression)
        return self._score_all_interventions(
            model,
            cause_effect.cause_variables,
            cause_effect.effect_variable,
            parameters.truncation_assignments_from_where_conditions,
            cause_effect.adjustment_variables_by_cause,
        )

    @staticmethod
    def _score_intervention(
        model: CausalCircuit,
        cause_variable: random_events.variable.Variable,
        effect_variable: random_events.variable.Variable,
        effect_truncation_event: Event | None,
        adjustment_variables: Iterable[random_events.variable.Variable] = (),
    ) -> ScoredIntervention | None:
        """
        Compute ``cause_variable``'s best-region search result and score it by how
        *reliably* that region produces the effect: ``P(effect | do(cause_variable in
        best_region))``. Comparable across candidates because it is a conditional
        probability under each candidate's own interventional distribution, not scaled
        by how much of that candidate's domain the region happens to cover.

        This must be scored on the interventional joint restricted to *only* the region,
        before conditioning on the effect: scoring on the already effect-truncated
        circuit instead would read close to 100% for *any* region, including a
        candidate's entire, unrestricted domain (an uninformative candidate's only
        "region"), since conditioning on the effect first makes whatever region is
        examined trivially compatible with it by construction. The region itself is
        still chosen by searching the effect-truncated circuit -- that search still
        needs to know which values are compatible with the effect; only the *score* is
        measured against the region-only, effect-free distribution.

        # `_best_disjoint_region` is private: `causal_circuit.py` is a stable #
        dependency this glue code does not modify beyond what #
        doc/eql/user/causality.md documents. It is the disjoint counterpart of #
        `_best_region` (which `diagnose_failure` uses for `recommended_region`): #
        `_best_region`'s regions always collapse to the variable's whole domain, which #
        cannot discriminate between candidates -- `_best_disjoint_region` keeps #
        separate SumUnit branches separate, which this ranking needs.

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

        # `.truncated()` fills in missing variables *in place* on the event it is
        # given, so reusing the same event object across several `.truncated()` calls
        # (each on a differently-shaped circuit) would leak one call's variables into
        # the next -- pass a fresh copy each time.
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
            return ScoredIntervention(cause_variable, 1.0, region.distribution)

        narrowed = region.distribution.truncated(copy.deepcopy(effect_truncation_event))
        if narrowed.distribution is None or narrowed.probability <= 0.0:
            return None
        return ScoredIntervention(
            cause_variable, float(narrowed.probability), narrowed.distribution
        )
