import enum
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from types import NoneType
from typing import Iterable, TypeVar

import random_events.variable
from random_events.product_algebra import Event
from sqlalchemy.orm import sessionmaker
from typing_extensions import ClassVar, Dict, List, Optional

from krrood import logger
from krrood.entity_query_language.verbalization.vocabulary.english import Directive

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.operators.causal import (
    Cause,
    CauseEffectVariables,
    ScoredIntervention,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    BackendCannotEvaluateCause,
    NoCauseVariablesForRanking,
    NoCausesEffectConditionForCause,
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
    SelectiveBackendCannotResolveEllipsisMatch,
    UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration,
)
from krrood.entity_query_language.factories import entity, set_of, variable
from krrood.entity_query_language.query.match import Match, AttributeMatch
from krrood.entity_query_language.query.query import Query
from krrood.ormatic.eql_interface import eql_to_sql

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
    )
except ImportError as e:
    logger.debug(f"Couldn't import probabilistic model needed classes: {e}")
    CausalCircuit = NoneType
    DoRequiresCausalCircuitModel = NoneType
    MultipleEffectVariablesNotSupported = NoneType
    ModelRegistry = NoneType
    FullyFactorizedRegistry = NoneType
    UnderspecifiedParameters = NoneType

T = TypeVar("T")


@dataclass
class QueryBackend(ABC):
    """
    Base class for all query backends.

    Query backends are objects that answer queries by different means.
    """

    opening_directive: ClassVar[Optional[Directive]] = None
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
        if not (isinstance(expression, Match) and expression.has_cause_attributes):
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

    opening_directive: ClassVar[Optional[Directive]] = Directive.FIND
    """
    Selecting from existing data reads as *"Find …"*.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if isinstance(expression, Match) and expression.has_ellipsis_attributes:
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

    opening_directive: ClassVar[Optional[Directive]] = Directive.GENERATE
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

    def _evaluate(self, expression: Evaluable) -> Iterable:
        yield from expression._evaluate_natively_()


@dataclass
class EntityQueryLanguageGenerativeBackend(GenerativeBackend):
    """
    A generative backend that constructs new instances deterministically: it treats a
    match's unspecified leaves as variables, enumerates every combination over their
    (discrete) domains, constructs an instance per combination via the type's
    constructor, and keeps those that satisfy the match's ``where`` conditions.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        self._warn_or_raise_on_unresolved_cause_(expression)
        variables: Dict[str, Variable] = {}
        for attribute_match in expression.matches_with_variables:
            self._check_attribute_match_is_suitable_for_generation(attribute_match)
            variables[attribute_match.name_from_variable_access_path] = (
                self._convert_attribute_match_to_variable(attribute_match)
            )

        expression.variable._update_domain_(
            self._generate_raw_results(expression, variables)
        )

        filtered_results = entity(expression.variable)._quantify_(
            expression._quantifier_type_
        )
        if expression._where_conditions_:
            filtered_results = filtered_results.where(*expression._where_conditions_)
        yield from filtered_results._evaluate_natively_()

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

    def _generate_raw_results(
        self, expression: Match[T], variables: Dict[str, Variable]
    ) -> Iterable[T]:
        """
        Construct instances from the given match and enumerable variables.

        :param expression: The match expression to construct instances from.
        :param variables: The variables to enumerate, keyed by access- path name.
        :return: A generator yielding an instance per variable combination.
        """
        all_combinations = set_of(*variables.values())
        for combination in all_combinations._evaluate_natively_():
            for variable_name, value in zip(variables, combination.values()):
                mapped_variable = expression._get_mapped_variable_by_name(variable_name)
                mapped_variable._value_ = value
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model using a model
    registry.
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

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:

        # generate parameters from example instance values
        parameters = UnderspecifiedParameters(expression)

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
                cause_effect.confounder_variables,
            )
            truncated = primary.narrowed_circuit
        else:
            # apply conditions from literal assignments to underspecified variables
            conditioned, _ = model.conditional(
                parameters.conditioning_assignments_from_literal_values
            )
            if conditioned is None:
                raise NoSolutionFound(expression.expression)

            # apply conditions from the where statements
            if parameters.truncation_assignments_from_where_conditions:
                truncated, _ = conditioned.truncated(
                    parameters.truncation_assignments_from_where_conditions
                )
            else:
                truncated = conditioned

        # apply conditions from variable assignments to underspecified variables
        if parameters.truncation_assignments_from_krrood_variables:
            complete_event = parameters.truncation_assignments_from_krrood_variables[0]
            complete_event.fill_missing_variables(parameters.variables.values())
            for event in parameters.truncation_assignments_from_krrood_variables[1:]:
                complete_event = complete_event.intersection_with(event)
            truncated, _ = truncated.truncated(complete_event, singleton_allowed=True)

            if truncated is None:
                raise NoSolutionFound(expression.expression)

        number_of_samples = expression.expression._limit_ or self.number_of_samples

        # sample and sort by log likelihood
        samples = truncated.sample(number_of_samples)
        log_likelihoods = truncated.log_likelihood(samples)
        samples = samples[log_likelihoods.argsort()[::-1]]

        # create new objects with the values from the samples
        for sample in samples:
            instance = parameters.construct_instance_from_model_sample(
                truncated.variables, sample
            )
            yield instance

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
            raise NoCausesEffectConditionForCause(expression.expression)
        if len(parameters.effect_variables_from_causes_effect) > 1:
            raise MultipleEffectVariablesNotSupported(
                parameters.effect_variables_from_causes_effect
            )
        [effect_variable] = parameters.effect_variables_from_causes_effect
        return CauseEffectVariables(
            parameters.search_cause_variables,
            effect_variable,
            parameters.search_confounder_variables,
        )

    @classmethod
    def _resolve_primary_intervention(
        cls,
        model: CausalCircuit,
        cause_variables: List[random_events.variable.Variable],
        effect_variable: random_events.variable.Variable,
        effect_truncation_event: Optional[Event],
        expression: Match[T],
        confounder_variables: Iterable[random_events.variable.Variable] = (),
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
        :param confounder_variables: Variables marked ``confounder`` in the query,
            passed through to ``backdoor_adjustment`` as its adjustment set.
        :raises NoSolutionFound: If no candidate has a region with positive probability.
        :return: The highest-scoring candidate.
        """
        scored_interventions = cls._score_all_interventions(
            model,
            cause_variables,
            effect_variable,
            effect_truncation_event,
            confounder_variables,
        )
        if not scored_interventions:
            raise NoSolutionFound(expression.expression)
        return scored_interventions[0]

    @classmethod
    def _score_all_interventions(
        cls,
        model: CausalCircuit,
        cause_variables: List[random_events.variable.Variable],
        effect_variable: random_events.variable.Variable,
        effect_truncation_event: Optional[Event],
        confounder_variables: Iterable[random_events.variable.Variable] = (),
    ) -> List[ScoredIntervention]:
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
        :param confounder_variables: Variables marked ``confounder`` in the query,
            passed through to ``backdoor_adjustment`` as its adjustment set for every
            candidate.
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
                    confounder_variables,
                )
            )
            is not None
        ]
        scored_interventions.sort(
            key=lambda candidate: candidate.effect_probability_given_region,
            reverse=True,
        )
        return scored_interventions

    def rank_causes(self, expression: Match[T]) -> List[ScoredIntervention]:
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

        Any field marked ``confounder`` is passed to every candidate's search as
        ``backdoor_adjustment``'s adjustment set, so a variable that drives both a
        candidate and the effect does not inflate that candidate's score with mere
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
        if not parameters.search_cause_variables:
            raise NoCauseVariablesForRanking(expression.expression)
        model = self.model_registry.get_model(parameters)
        if not isinstance(model, CausalCircuit):
            raise DoRequiresCausalCircuitModel(model)
        cause_effect = self._resolve_cause_and_effect_variables(parameters, expression)
        return self._score_all_interventions(
            model,
            cause_effect.cause_variables,
            cause_effect.effect_variable,
            parameters.truncation_assignments_from_where_conditions,
            cause_effect.confounder_variables,
        )

    @staticmethod
    def _score_intervention(
        model: CausalCircuit,
        cause_variable: random_events.variable.Variable,
        effect_variable: random_events.variable.Variable,
        effect_truncation_event: Optional[Event],
        confounder_variables: Iterable[random_events.variable.Variable] = (),
    ) -> Optional[ScoredIntervention]:
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
        :param confounder_variables: Variables marked ``confounder`` in the query,
            passed through to ``backdoor_adjustment`` as its adjustment set.
        :return: The scored candidate, or ``None`` if it has no region with positive
            probability, or the effect has zero probability within that region.
        """
        interventional = model.backdoor_adjustment(
            cause_variable, effect_variable, list(confounder_variables)
        )

        # `.truncated()` fills in missing variables *in place* on the event it is
        # given, so reusing the same event object across several `.truncated()` calls
        # (each on a differently-shaped circuit) would leak one call's variables into
        # the next -- pass a fresh copy each time. `Event` defines its own
        # no-argument `__deepcopy__`, incompatible with the `copy` module's
        # memo-passing convention, so it is called directly rather than through
        # `copy.deepcopy`.
        if effect_truncation_event:
            effect_truncated, _ = interventional.truncated(
                effect_truncation_event.__deepcopy__()
            )
        else:
            effect_truncated = interventional
        if effect_truncated is None:
            return None

        best_region = model._best_disjoint_region(cause_variable, effect_truncated)
        if best_region is None:
            return None

        region_only, region_prior_probability = interventional.truncated(
            best_region.fill_missing_variables_pure(interventional.variables)
        )
        if region_only is None or region_prior_probability <= 0.0:
            return None

        if effect_truncation_event:
            narrowed, effect_given_region_probability = region_only.truncated(
                effect_truncation_event.__deepcopy__()
            )
        else:
            narrowed, effect_given_region_probability = region_only, 1.0
        if narrowed is None or effect_given_region_probability <= 0.0:
            return None
        return ScoredIntervention(
            cause_variable, float(effect_given_region_probability), narrowed
        )
