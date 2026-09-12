"""
Asking every question of the catalogue to every pipeline and writing down what each
answered, how fast, and how much model it took.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.exceptions import DoRequiresCausalCircuitModel
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    EmptyInterventionalCircuitError,
    SupportDeterminismVerificationResult,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    ClassCircuitGroundingFailedError,
    PartCircuitGroundingFailedError,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.interval import Interval
from random_events.product_algebra import Event
from random_events.variable import Variable
from typing_extensions import Dict, List, Optional, Sequence, Tuple, Type

from experiments.causal_reasoning.tracy_clutter_picking.dataset import (
    ClutterPickDataset,
    SuccessRate,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterPickSceneAggregations,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    FlatTableSchemaMismatchError,
)
from experiments.causal_reasoning.tracy_clutter_picking.pipelines import (
    CausalQueryPipeline,
    FitReport,
    LikelihoodReport,
    pipelines_for,
)
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    CausalQueryCase,
    query_catalogue,
)

# %% what one question yields on one pipeline


class Refusal(StrEnum):
    """
    Why a pipeline could not answer a question.
    """

    SCHEMA_MISMATCH = "the fitted table has no column for the queried variables"
    """
    The question is about a clutter of another size than the flat table was built for.
    """

    NOT_SUPPORT_DETERMINISTIC = "the model is not support-deterministic over the cause"
    """
    The fitted circuit mixes branches that overlap on the cause, so backdoor adjustment
    has no disjoint regions to intervene on.
    """

    GROUNDING_FAILED = "grounding the query left no circuit"
    """
    Conditioning the relational circuit on the query emptied it.
    """

    NO_INTERVENTION_REGION = "no cause region carries probability"
    """
    Every region of the cause truncates the circuit to nothing.
    """

    NOT_A_CAUSAL_MODEL = "the registry returned no causal circuit"
    """
    The pipeline served a plain circuit for a question that marks a cause.
    """

    @classmethod
    def by_exception(cls) -> Dict[Type[Exception], Refusal]:
        """
        :return: The exceptions a pipeline refuses a question with, and what each one
            means.
        """
        return {
            FlatTableSchemaMismatchError: cls.SCHEMA_MISMATCH,
            SupportDeterminismVerificationResult: cls.NOT_SUPPORT_DETERMINISTIC,
            ClassCircuitGroundingFailedError: cls.GROUNDING_FAILED,
            PartCircuitGroundingFailedError: cls.GROUNDING_FAILED,
            EmptyInterventionalCircuitError: cls.NO_INTERVENTION_REGION,
            DoRequiresCausalCircuitModel: cls.NOT_A_CAUSAL_MODEL,
        }

    @classmethod
    def exception_types(cls) -> Tuple[Type[Exception], ...]:
        """
        :return: Every exception a pipeline refuses a question with.
        """
        return tuple(cls.by_exception())


@dataclass(frozen=True)
class InterventionalEffect:
    """
    The effect's probability under an intervention setting the cause to one region.
    """

    cause_region: str
    """
    The region of the cause, written out.
    """

    region_probability: float
    """
    The region's own probability under the fitted model.
    """

    naive_probability: float
    """
    ``P(effect | cause in region)``, read off the model with no adjustment.
    """

    adjusted_probability: float
    """
    ``P(effect | do(cause in region))``, backdoor-adjusted for the question's
    confounders; equal to the naive one when the question names none.
    """


@dataclass
class QueryOutcome:
    """
    What one pipeline made of one question.
    """

    case: CausalQueryCase
    """
    The question.
    """

    pipeline_name: str
    """
    The pipeline that was asked.
    """

    duration: float
    """
    Wall-clock seconds from asking to the answer or the refusal, the first time the
    question was asked -- including the fit of the cause-specific model if that cause
    had not been asked about before.
    """

    repeat_duration: float = float("nan")
    """
    Wall-clock seconds the same question took asked again, with every model fitted:
    grounding, verification and adjustment alone.
    """

    refusal: Optional[Refusal] = None
    """
    Why the pipeline could not answer, or ``None`` if it did.
    """

    best_region: Optional[str] = None
    """
    The cause region EQL's own search settles on, written out: the region most probable
    once the effect is required to hold, which is the one the query's samples are drawn
    from.
    """

    effect_probability_given_best_region: Optional[float] = None
    """
    ``P(effect | do(cause in best region))``.
    """

    effects: List[InterventionalEffect] = field(default_factory=list)
    """
    The effect's probability under every cause region the model distinguishes.
    """

    @property
    def answered(self) -> bool:
        """
        Whether the pipeline answered the question.
        """
        return self.refusal is None

    @property
    def most_effective(self) -> Optional[InterventionalEffect]:
        """
        The cause region whose intervention gives the effect the highest adjusted
        probability, or ``None`` if the question was refused.
        """
        if not self.effects:
            return None
        return max(self.effects, key=lambda effect: effect.adjusted_probability)


# %% asking


def describe_region(event: Event, variable: Variable) -> str:
    """
    :param event: An event restricting ``variable``.
    :param variable: The variable to describe the restriction of.
    :return: The restriction written out: a point as its value, a range as its bounds,
        a set of symbols as their names.
    """
    parts = []
    for simple_event in event.simple_sets:
        value = simple_event[variable]
        if isinstance(value, Interval):
            for interval in value.simple_sets:
                if interval.lower == interval.upper:
                    parts.append(f"{interval.lower:g}")
                else:
                    parts.append(f"[{interval.lower:g}, {interval.upper:g}]")
        else:
            parts.extend(str(element) for element in value.simple_sets)
    return " or ".join(parts)


@dataclass
class QuestionAsker:
    """
    Asks the questions and records the outcomes.
    """

    random_seed: int = 0
    """
    Seed applied to the global NumPy random state before each question, since
    relational grounding draws Monte-Carlo samples for open aggregation statistics.
    """

    def ask(self, pipeline: CausalQueryPipeline, case: CausalQueryCase) -> QueryOutcome:
        """
        Ask one pipeline one question.

        :param pipeline: The pipeline to ask.
        :param case: The question.
        :return: What came of it.
        """
        backend = ProbabilisticBackend(model_registry=pipeline.registry)
        np.random.seed(self.random_seed)
        started = time.perf_counter()
        try:
            [primary] = backend.rank_causes(case.build())
        except Refusal.exception_types() as refusal:
            return QueryOutcome(
                case=case,
                pipeline_name=pipeline.name,
                duration=time.perf_counter() - started,
                refusal=Refusal.by_exception()[type(refusal)],
            )
        duration = time.perf_counter() - started
        cause = primary.cause_variable
        outcome = QueryOutcome(
            case=case,
            pipeline_name=pipeline.name,
            duration=duration,
            best_region=describe_region(
                primary.narrowed_circuit.marginal([cause]).support, cause
            ),
            effect_probability_given_best_region=(
                primary.effect_probability_given_region
            ),
        )
        np.random.seed(self.random_seed)
        outcome.effects = self._effects(pipeline, case)
        return outcome

    def time_repeat(
        self, pipeline: CausalQueryPipeline, case: CausalQueryCase
    ) -> float:
        """
        Ask a question again and time it alone.

        :param pipeline: The pipeline to ask, with every model it needs fitted.
        :param case: The question.
        :return: Wall-clock seconds to the answer or the refusal.
        """
        backend = ProbabilisticBackend(model_registry=pipeline.registry)
        np.random.seed(self.random_seed)
        started = time.perf_counter()
        try:
            backend.rank_causes(case.build())
        except Refusal.exception_types():
            pass
        return time.perf_counter() - started

    @staticmethod
    def _effects(
        pipeline: CausalQueryPipeline, case: CausalQueryCase
    ) -> List[InterventionalEffect]:
        """
        Read the effect's naive and adjusted probability off every cause region.

        :param pipeline: The pipeline whose model to read.
        :param case: The question.
        :return: One row per region, in the model's own order.
        """
        parameters = UnderspecifiedParameters(case.build())
        causal_circuit: CausalCircuit = pipeline.registry.get_model(parameters)
        [cause] = causal_circuit.causal_variables
        [effect] = causal_circuit.effect_variables
        effect_event = parameters.truncation_assignments_from_where_conditions
        naive = causal_circuit.backdoor_adjustment(cause, effect)
        adjusted = causal_circuit.backdoor_adjustment(
            cause, effect, adjustment_variables=parameters.search_confounder_variables
        )
        return [
            InterventionalEffect(
                cause_region=describe_region(region.event, cause),
                region_probability=region.probability,
                naive_probability=_effect_probability_in(
                    naive, region.event, effect_event
                ),
                adjusted_probability=_effect_probability_in(
                    adjusted, region.event, effect_event
                ),
            )
            for region in causal_circuit._extract_disjoint_regions_for_variable(cause)
        ]


def _effect_probability_in(
    interventional: ProbabilisticCircuit, region: Event, effect_event: Event
) -> float:
    """
    :param interventional: A joint circuit over the cause and the effect.
    :param region: The cause region to restrict to.
    :param effect_event: The effect's condition.
    :return: The effect condition's probability within the region.
    """
    truncated, _ = interventional.truncated(
        region.fill_missing_variables_pure(interventional.variables)
    )
    if truncated is None:
        return 0.0
    return float(
        truncated.probability(
            effect_event.fill_missing_variables_pure(truncated.variables)
        )
    )


# %% the whole comparison


@dataclass
class PipelineReport:
    """
    Everything one pipeline reported over the comparison.
    """

    name: str
    """
    The pipeline's name.
    """

    fit: FitReport
    """
    What its fits cost.
    """

    likelihood: LikelihoodReport
    """
    How well its plain model explains the held-out attempts.
    """

    outcomes: List[QueryOutcome] = field(default_factory=list)
    """
    What it made of every question, in catalogue order.
    """


@dataclass
class EvaluationReport:
    """
    The comparison of both pipelines on one dataset.
    """

    training_scene_count: int
    """
    How many attempts the pipelines were fitted on.
    """

    test_scene_count: int
    """
    How many held-out attempts they were scored on.
    """

    recorded_neighbour_count: int
    """
    How many neighbours the recorded attempts have.
    """

    success_rate: float
    """
    Share of all recorded attempts whose target was lifted.
    """

    success_by_environment: Dict[str, SuccessRate] = field(default_factory=dict)
    """
    The lifted share per environment, keyed by the environment's name.
    """

    success_by_friction: Dict[float, SuccessRate] = field(default_factory=dict)
    """
    The lifted share per friction level.
    """

    success_by_crowding: Dict[int, SuccessRate] = field(default_factory=dict)
    """
    The lifted share per number of adjacent neighbours.
    """

    min_samples_per_leaf: int = 0
    """
    The fewest training rows a leaf of a cause-specific tree was allowed to hold.
    """

    plain_min_samples_per_leaf: int = 0
    """
    The fewest training rows a leaf of the plain tree was allowed to hold.
    """

    pipelines: List[PipelineReport] = field(default_factory=list)
    """
    One report per pipeline.
    """

    shared_coverage_log_likelihoods: Dict[str, float] = field(default_factory=dict)
    """
    Each pipeline's mean log-likelihood over the held-out attempts every pipeline
    covers, so the numbers are comparable, keyed by pipeline name.
    """


def evaluate(
    dataset: ClutterPickDataset,
    train_fraction: float = 0.8,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[int] = None,
    plain_min_samples_per_leaf: Optional[int] = None,
    cases: Optional[Sequence[CausalQueryCase]] = None,
) -> EvaluationReport:
    """
    Fit both pipelines on part of the dataset, score them on the rest, and ask them
    every question.

    :param dataset: The recorded attempts.
    :param train_fraction: Share of attempts to fit on.
    :param random_seed: Seed of the split and of the questions' Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        tree may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain
        tree may hold; the pipelines' own default if not given.
    :param cases: The questions to ask; defaults to :func:`query_catalogue`.
    :return: The comparison.
    """
    training, test = dataset.split(train_fraction, np.random.default_rng(random_seed))
    [recorded_neighbour_count] = {len(scene.neighbours) for scene in dataset.scenes}
    cases = list(cases or query_catalogue(recorded_neighbour_count))
    report = EvaluationReport(
        training_scene_count=len(training.scenes),
        test_scene_count=len(test.scenes),
        recorded_neighbour_count=recorded_neighbour_count,
        success_rate=dataset.success_rate,
        success_by_environment=dataset.success_rate_by(
            lambda scene: str(scene.environment)
        ),
        success_by_friction=dataset.success_rate_by(
            lambda scene: scene.friction_coefficient
        ),
        success_by_crowding=dataset.success_rate_by(
            lambda scene: ClutterPickSceneAggregations(instance=scene).crowding_count()
        ),
    )
    asker = QuestionAsker(random_seed=random_seed)
    pipelines = pipelines_for(recorded_neighbour_count)
    log_likelihoods: Dict[str, np.ndarray] = {}
    for pipeline in pipelines:
        if min_samples_per_leaf is not None:
            pipeline.min_samples_per_leaf = min_samples_per_leaf
        if plain_min_samples_per_leaf is not None:
            pipeline.plain_min_samples_per_leaf = plain_min_samples_per_leaf
        report.min_samples_per_leaf = pipeline.min_samples_per_leaf
        report.plain_min_samples_per_leaf = pipeline.plain_min_samples_per_leaf
        fit = pipeline.fit(training.scenes)
        likelihood = pipeline.log_likelihood(test.scenes)
        log_likelihoods[pipeline.name] = likelihood.log_likelihoods
        pipeline_report = PipelineReport(
            name=pipeline.name, fit=fit, likelihood=likelihood
        )
        for case in cases:
            pipeline_report.outcomes.append(asker.ask(pipeline, case))
        for outcome in pipeline_report.outcomes:
            outcome.repeat_duration = asker.time_repeat(pipeline, outcome.case)
        report.pipelines.append(pipeline_report)
    covered_by_all = np.all(
        [np.isfinite(values) for values in log_likelihoods.values()], axis=0
    )
    report.shared_coverage_log_likelihoods = {
        name: (
            float(values[covered_by_all].mean())
            if covered_by_all.any()
            else float("nan")
        )
        for name, values in log_likelihoods.items()
    }
    return report
