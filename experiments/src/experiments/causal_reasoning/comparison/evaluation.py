"""
Asking every question of the catalogue to every pipeline and writing down what each
answered, how fast, and how much model it took.
"""

from __future__ import annotations

import math
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
from scipy.stats import spearmanr
from abc import ABC, abstractmethod
from typing_extensions import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from experiments.causal_reasoning.comparison.dataset import EffectRate, ExampleDataset
from experiments.causal_reasoning.comparison.domain import ExampleView, RelationalDomain
from experiments.causal_reasoning.comparison.exceptions import (
    FlatTableSchemaMismatchError,
)
from experiments.causal_reasoning.comparison.pipelines import (
    CausalQueryPipeline,
    FitReport,
    LikelihoodReport,
    RelationalPipeline,
    pipelines,
)
from experiments.causal_reasoning.comparison.queries import CausalQueryCase

# %% what one question yields on one pipeline


class Refusal(StrEnum):
    """
    Why a pipeline could not answer a question.
    """

    SCHEMA_MISMATCH = "the fitted table has no column for the queried variables"
    """
    The question constrains a part's attribute, which the flat table
    has no column for.
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

    EFFECT_NEVER_OCCURS = "the effect has zero probability under every cause region"
    """
    No region of the cause gives the effect any probability, so the search over
    interventions has nothing to rank.
    """

    OVERLAPPING_REGIONS = "the cause regions read off the model overlap"
    """
    The regions of the cause the model distinguishes carry more probability together than
    one, so they overlap instead of partitioning the cause, and the effect read off each
    of them is not an interventional probability.
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
class ProportionInterval:
    """
    A confidence interval for a probability read off a region of the training data.
    """

    lower: float
    """
    The lower bound.
    """

    upper: float
    """
    The upper bound.
    """

    @classmethod
    def wilson(
        cls, probability: float, count: int, z: float = 1.96
    ) -> ProportionInterval:
        """
        The Wilson score interval of a proportion over ``count`` observations.

        :param probability: The proportion.
        :param count: How many observations it is read off; zero gives the whole unit
            interval.
        :param z: The standard-normal quantile of the confidence level; 1.96 for 95%.
        :return: The interval.
        """
        if count == 0:
            return cls(0.0, 1.0)
        centre = (probability + z**2 / (2 * count)) / (1 + z**2 / count)
        half_width = (
            z
            * math.sqrt(probability * (1 - probability) / count + z**2 / (4 * count**2))
            / (1 + z**2 / count)
        )
        return cls(max(0.0, centre - half_width), min(1.0, centre + half_width))

    def difference_from(
        self, other: ProportionInterval, probability: float, other_probability: float
    ) -> ProportionInterval:
        """
        Newcombe's interval for the difference between two proportions, this one minus
        the other.

        :param other: The other proportion's interval.
        :param probability: This proportion.
        :param other_probability: The other proportion.
        :return: The interval of the difference.
        """
        difference = probability - other_probability
        return ProportionInterval(
            difference
            - math.sqrt(
                (probability - self.lower) ** 2 + (other.upper - other_probability) ** 2
            ),
            difference
            + math.sqrt(
                (self.upper - probability) ** 2 + (other_probability - other.lower) ** 2
            ),
        )


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

    support_count: int
    """
    How many training rows of the cause fall in the region: examples for an example-level
    cause, parts for a part's own attribute.
    """

    ordinal: Optional[float] = None
    """
    Where the region sits on the cause's scale, for a numeric cause; ``None`` for a
    symbolic one, whose regions have no order.
    """

    @property
    def adjusted_interval(self) -> ProportionInterval:
        """
        The Wilson interval of the adjusted probability over the region's support.
        """
        return ProportionInterval.wilson(self.adjusted_probability, self.support_count)

    @property
    def naive_interval(self) -> ProportionInterval:
        """
        The Wilson interval of the naive probability over the region's support.
        """
        return ProportionInterval.wilson(self.naive_probability, self.support_count)

    def is_supported(self, min_region_support: int) -> bool:
        """
        :param min_region_support: The fewest training rows a region may hold to be
            read.
        :return: Whether the region holds at least that many.
        """
        return self.support_count >= min_region_support


@dataclass(frozen=True)
class Contrast:
    """
    The adjusted probability at the high end of the cause against the low end.
    """

    high_region: str
    """
    The region at the high end, written out.
    """

    low_region: str
    """
    The region at the low end, written out.
    """

    difference: float
    """
    The adjusted probability at the high end minus at the low end.
    """

    interval: ProportionInterval
    """
    Newcombe's interval of the difference.
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
    question was asked, including the fit of the cause-specific model if that cause had
    not been asked about before.
    """

    min_region_support: int = 10
    """
    The fewest training rows a cause region may hold for its effect to be read as an
    answer; a region below it is reported but takes no part in the summary.
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
    def supported_effects(self) -> List[InterventionalEffect]:
        """
        The effects read off regions holding at least :attr:`min_region_support`
        training rows.
        """
        return [
            effect
            for effect in self.effects
            if effect.is_supported(self.min_region_support)
        ]

    @property
    def most_effective(self) -> Optional[InterventionalEffect]:
        """
        The supported cause region whose intervention gives the effect the highest
        adjusted probability, or ``None`` if the question was refused or no region is
        supported.
        """
        supported = self.supported_effects
        if not supported:
            return None
        return max(supported, key=lambda effect: effect.adjusted_probability)

    @property
    def least_effective(self) -> Optional[InterventionalEffect]:
        """
        The supported cause region whose intervention gives the effect the lowest
        adjusted probability, or ``None`` if the question was refused or no region is
        supported.
        """
        supported = self.supported_effects
        if not supported:
            return None
        return min(supported, key=lambda effect: effect.adjusted_probability)

    @property
    def trend(self) -> Optional[float]:
        """
        Spearman's rank correlation between the cause's value and the adjusted
        probability over the supported regions; ``None`` for a symbolic cause, whose
        regions have no order, or with fewer than three supported regions.
        """
        ordered = [
            effect for effect in self.supported_effects if effect.ordinal is not None
        ]
        probabilities = [effect.adjusted_probability for effect in ordered]
        if len(ordered) < 3 or len(set(probabilities)) == 1:
            return None
        correlation = spearmanr(
            [effect.ordinal for effect in ordered], probabilities
        ).statistic
        return None if math.isnan(correlation) else float(correlation)

    @property
    def contrast(self) -> Optional[Contrast]:
        """
        The adjusted probability at the highest supported region of a numeric cause
        against the lowest, or at the most effective supported region of a symbolic
        cause against the least; ``None`` with fewer than two supported regions.
        """
        supported = self.supported_effects
        if len(supported) < 2:
            return None
        if all(effect.ordinal is not None for effect in supported):
            low = min(supported, key=lambda effect: effect.ordinal)
            high = max(supported, key=lambda effect: effect.ordinal)
        else:
            low, high = self.least_effective, self.most_effective
        return Contrast(
            high_region=high.cause_region,
            low_region=low.cause_region,
            difference=high.adjusted_probability - low.adjusted_probability,
            interval=high.adjusted_interval.difference_from(
                low.adjusted_interval,
                high.adjusted_probability,
                low.adjusted_probability,
            ),
        )


# %% asking


def describe_region(event: Event, variable: Variable) -> str:
    """
    :param event: An event restricting ``variable``.
    :param variable: The variable to describe the restriction of.
    :return: The restriction written out: a point as its value, a range as its bounds, a
        set of symbols as their names.
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


def regions_partition_the_cause(
    effects: Sequence[InterventionalEffect], tolerance: float
) -> bool:
    """
    :param effects: The effect under every region of the cause the model distinguishes.
    :param tolerance: How far the regions' probabilities may sum away from one.
    :return: Whether the regions partition the cause, so that their probabilities sum to
        one.
    """
    return abs(sum(effect.region_probability for effect in effects) - 1) <= tolerance


@dataclass
class QuestionAsker:
    """
    Asks the questions and records the outcomes.
    """

    random_seed: int = 0
    """
    Seed applied to the global NumPy random state before each question, since relational
    grounding draws Monte-Carlo samples for open aggregation counts.
    """

    region_probability_tolerance: float = 1e-6
    """
    How far the probabilities of the cause regions may sum away from one before the
    regions count as overlapping.
    """

    min_region_support: int = 10
    """
    The fewest training rows a cause region may hold for its effect to be read as an
    answer.
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
            ranked = backend.rank_causes(case.build())
        except Refusal.exception_types() as refusal:
            return QueryOutcome(
                case=case,
                pipeline_name=pipeline.name,
                min_region_support=self.min_region_support,
                duration=time.perf_counter() - started,
                refusal=Refusal.by_exception()[type(refusal)],
            )
        duration = time.perf_counter() - started
        if not ranked:
            return QueryOutcome(
                case=case,
                pipeline_name=pipeline.name,
                min_region_support=self.min_region_support,
                duration=duration,
                refusal=Refusal.EFFECT_NEVER_OCCURS,
            )
        [primary] = ranked
        cause = primary.cause_variable
        outcome = QueryOutcome(
            case=case,
            pipeline_name=pipeline.name,
            min_region_support=self.min_region_support,
            duration=duration,
            best_region=describe_region(
                primary.narrowed_circuit.marginal([cause]).support, cause
            ),
            effect_probability_given_best_region=(
                primary.effect_probability_given_region
            ),
        )
        np.random.seed(self.random_seed)
        effects = self._effects(pipeline, case)
        if not regions_partition_the_cause(effects, self.region_probability_tolerance):
            return QueryOutcome(
                case=case,
                pipeline_name=pipeline.name,
                min_region_support=self.min_region_support,
                duration=duration,
                refusal=Refusal.OVERLAPPING_REGIONS,
            )
        outcome.effects = effects
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
        Read the effect's naive and adjusted probability off every cause region, with
        how many training rows the region holds.

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
        training_values = pipeline.training_values_of(cause.name)
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
                support_count=_support_count(region.event, cause, training_values),
                ordinal=_ordinal(region.event, cause),
            )
            for region in causal_circuit._extract_disjoint_regions_for_variable(cause)
        ]


def _support_count(region: Event, variable: Variable, values: Sequence[Any]) -> int:
    """
    :param region: An event restricting ``variable``.
    :param variable: The variable the region restricts.
    :param values: The training values of that variable.
    :return: How many of the values fall in the region.
    """
    [simple_region] = region.simple_sets
    restriction = simple_region[variable]
    return sum(
        not variable.make_value(value).intersection_with(restriction).is_empty()
        for value in values
    )


def _ordinal(region: Event, variable: Variable) -> Optional[float]:
    """
    :param region: An event restricting ``variable``.
    :param variable: The variable the region restricts.
    :return: The region's lower bound for a numeric variable, ``None`` for a symbolic
        one.
    """
    [simple_region] = region.simple_sets
    restriction = simple_region[variable]
    if not isinstance(restriction, Interval):
        return None
    return float(min(interval.lower for interval in restriction.simple_sets))


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

    likelihoods: Dict[ExampleView, Optional[LikelihoodReport]] = field(
        default_factory=dict
    )
    """
    How well its plain model explains the held-out examples, per view; ``None`` for a view
    the pipeline models less than.
    """

    outcomes: List[QueryOutcome] = field(default_factory=list)
    """
    What it made of every question, in catalogue order.
    """


@dataclass
class EvaluationReport:
    """
    The comparison of every pipeline on one split of one dataset.
    """

    random_seed: int
    """
    The seed of the split and of the questions' Monte-Carlo grounding.
    """

    training_example_count: int
    """
    How many examples the pipelines were fitted on.
    """

    test_example_count: int
    """
    How many held-out examples they were scored on.
    """

    effect_rate: float
    """
    Share of all examples that show the effect the questions ask about.
    """

    effect_rates_by: Dict[str, Dict[Any, EffectRate]] = field(default_factory=dict)
    """
    The share per value of what the experiment summarises the examples by, per
    summary's title.
    """

    min_region_support: int = 10
    """
    The fewest training rows a cause region was allowed to hold for its effect to be
    read as an answer.
    """

    min_samples_per_leaf: float = 0.0
    """
    The share of its training rows a leaf of a cause-specific model was allowed to hold.
    """

    plain_min_samples_per_leaf: float = 0.0
    """
    The share of its training rows a leaf of the plain model was allowed to hold.
    """

    pipelines: List[PipelineReport] = field(default_factory=list)
    """
    One report per pipeline.
    """

    shared_coverage_log_likelihoods: Dict[ExampleView, Dict[str, float]] = field(
        default_factory=dict
    )
    """
    Per view, each pipeline's mean log-likelihood over the held-out examples every pipeline
    modelling that view covers, so the numbers are comparable, keyed by pipeline name.
    """

    def pipeline(self, name: str) -> PipelineReport:
        """
        :param name: A pipeline's name.
        :return: Its report.
        """
        [report] = [report for report in self.pipelines if report.name == name]
        return report


def shared_coverage_log_likelihoods(
    log_likelihoods: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    :param log_likelihoods: Each pipeline's log-likelihood per held-out example, by
        pipeline name.
    :return: Each pipeline's mean over the examples every pipeline covers.
    """
    covered_by_all = np.all(
        [np.isfinite(values) for values in log_likelihoods.values()], axis=0
    )
    return {
        name: (
            float(values[covered_by_all].mean())
            if covered_by_all.any()
            else float("nan")
        )
        for name, values in log_likelihoods.items()
    }


@dataclass(frozen=True)
class Comparison:
    """
    What one experiment compares: its example, which of its parts are positional, and
    how it summarises the effect over its examples.
    """

    domain: RelationalDomain
    """
    The example and its parts.
    """

    positional_fields: Tuple[str, ...] = ()
    """
    The part fields a position means the same thing in for every example; a hybrid
    circuit holding them by position is compared when there are any.
    """

    summaries: Callable[[ExampleDataset], Dict[str, Dict[Any, EffectRate]]] = (
        lambda dataset: {}
    )
    """
    The effect's rate per value of whatever the experiment groups its examples by,
    per summary's title, for the report.
    """

    def pipelines(
        self,
        examples: Sequence[Any],
        min_samples_per_leaf: Optional[float],
        plain_min_samples_per_leaf: Optional[float],
    ) -> List[CausalQueryPipeline]:
        """
        :param examples: The examples the pipelines will be fitted on.
        :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
            model may hold; the pipelines' own default if not given.
        :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain
            model may hold; the pipelines' own default if not given.
        :return: Every pipeline, unfitted, with those settings.
        """
        configured = pipelines(self.domain, examples, self.positional_fields)
        for pipeline in configured:
            if min_samples_per_leaf is not None:
                pipeline.min_samples_per_leaf = min_samples_per_leaf
            if plain_min_samples_per_leaf is not None:
                pipeline.plain_min_samples_per_leaf = plain_min_samples_per_leaf
        return configured


def score_every_view(
    pipeline: CausalQueryPipeline, examples: Sequence[Any]
) -> Dict[ExampleView, Optional[LikelihoodReport]]:
    """
    :param pipeline: A fitted pipeline.
    :param examples: The held-out examples.
    :return: The pipeline's likelihood report per view.
    """
    return {view: pipeline.log_likelihood(examples, view) for view in ExampleView}


def evaluate(
    comparison: Comparison,
    dataset: ExampleDataset,
    cases: Sequence[CausalQueryCase],
    train_fraction: float = 0.8,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    time_repeats: bool = True,
    min_region_support: int = 10,
) -> EvaluationReport:
    """
    Fit every pipeline on part of the dataset, score them on the rest, and ask them every
    question.

    :param comparison: What is compared.
    :param dataset: The examples.
    :param cases: The questions to ask.
    :param train_fraction: Share of examples to fit on.
    :param random_seed: Seed of the split and of the questions' Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :param time_repeats: Whether to ask every question a second time to time it alone.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The comparison.
    """
    training, test = dataset.split(train_fraction, np.random.default_rng(random_seed))
    report = EvaluationReport(
        random_seed=random_seed,
        training_example_count=len(training.examples),
        test_example_count=len(test.examples),
        min_region_support=min_region_support,
        effect_rate=dataset.effect_rate,
        effect_rates_by=comparison.summaries(dataset),
    )
    asker = QuestionAsker(
        random_seed=random_seed, min_region_support=min_region_support
    )
    log_likelihoods: Dict[ExampleView, Dict[str, np.ndarray]] = {
        view: {} for view in ExampleView
    }
    for pipeline in comparison.pipelines(
        training.examples, min_samples_per_leaf, plain_min_samples_per_leaf
    ):
        report.min_samples_per_leaf = pipeline.min_samples_per_leaf
        report.plain_min_samples_per_leaf = pipeline.plain_min_samples_per_leaf
        fit = pipeline.fit(training.examples)
        pipeline_report = PipelineReport(
            name=pipeline.name,
            fit=fit,
            likelihoods=score_every_view(pipeline, test.examples),
        )
        for view, likelihood in pipeline_report.likelihoods.items():
            if likelihood is not None:
                log_likelihoods[view][pipeline.name] = likelihood.log_likelihoods
        for case in cases:
            pipeline_report.outcomes.append(asker.ask(pipeline, case))
        if time_repeats:
            for outcome in pipeline_report.outcomes:
                outcome.repeat_duration = asker.time_repeat(pipeline, outcome.case)
        report.pipelines.append(pipeline_report)
    for baseline_report in baseline_reports(
        comparison.domain, training.examples, cases, min_region_support
    ):
        report.pipelines.append(baseline_report)
    report.shared_coverage_log_likelihoods = {
        view: shared_coverage_log_likelihoods(values)
        for view, values in log_likelihoods.items()
    }
    return report


def baselines_of(domain: RelationalDomain, min_region_support: int) -> List[Any]:
    """
    The estimators that are not circuits, which every comparison asks the same
    questions of.

    :param domain: The example and its parts.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The estimators, unfitted.
    """
    from experiments.causal_reasoning.comparison.baselines import (
        RegressionAdjustmentBaseline,
    )
    from experiments.causal_reasoning.comparison.neural_baseline import (
        NeuralAdjustmentBaseline,
    )

    return [
        RegressionAdjustmentBaseline(
            domain=domain, min_region_support=min_region_support
        ),
        NeuralAdjustmentBaseline(domain=domain, min_region_support=min_region_support),
    ]


def baseline_reports(
    domain: RelationalDomain,
    examples: Sequence[Any],
    cases: Sequence[CausalQueryCase],
    min_region_support: int,
) -> List[PipelineReport]:
    """
    Ask every estimator that is not a circuit every question, as reports shaped like a
    pipeline's, with no likelihoods since they model no distribution.

    :param domain: The example and its parts.
    :param examples: The examples to fit on.
    :param cases: The questions.
    :param min_region_support: The fewest training examples a cause region may hold
        for its effect to be read as an answer.
    :return: One report per estimator.
    """
    return [
        _baseline_report(baseline, examples, cases)
        for baseline in baselines_of(domain, min_region_support)
    ]


def _baseline_report(
    baseline: Any, examples: Sequence[Any], cases: Sequence[CausalQueryCase]
) -> PipelineReport:
    """
    :param baseline: An estimator that is not a circuit.
    :param examples: The examples to fit on.
    :param cases: The questions.
    :return: What it made of them.
    """
    fit = baseline.fit(examples)
    return PipelineReport(
        name=baseline.name,
        fit=fit,
        likelihoods={view: None for view in ExampleView},
        outcomes=[baseline.ask(case) for case in cases],
    )


# %% the order of the parts


@dataclass(frozen=True)
class Spread:
    """
    How a number varies over the reorderings.
    """

    mean: float
    """
    Its mean.
    """

    standard_deviation: float
    """
    Its standard deviation.
    """

    lowest: float
    """
    Its lowest value.
    """

    highest: float
    """
    Its highest value.
    """

    @classmethod
    def of(cls, values: Sequence[float]) -> Spread:
        """
        :param values: The values, at least one.
        :return: Their spread.
        """
        return cls(
            mean=float(np.mean(values)),
            standard_deviation=float(np.std(values)),
            lowest=float(np.min(values)),
            highest=float(np.max(values)),
        )

    @property
    def range(self) -> float:
        """
        The highest value minus the lowest.
        """
        return self.highest - self.lowest


@dataclass
class PermutedOutcomes:
    """
    What one pipeline made of one question with the parts in the dataset's own order
    and under several random reorderings.
    """

    pipeline_name: str
    """
    The pipeline that was asked.
    """

    case: CausalQueryCase
    """
    The question.
    """

    in_dataset_order: QueryOutcome
    """
    The outcome with the parts in the order the dataset lists them.
    """

    reordered: List[QueryOutcome] = field(default_factory=list)
    """
    One outcome per random reordering.
    """

    @property
    def outcomes(self) -> List[QueryOutcome]:
        """
        Every outcome, the dataset's order first.
        """
        return [self.in_dataset_order] + self.reordered

    @property
    def answered(self) -> List[QueryOutcome]:
        """
        The reordered outcomes that were answered.
        """
        return [outcome for outcome in self.reordered if outcome.answered]

    @property
    def best_regions(self) -> List[str]:
        """
        The distinct most effective cause regions found over every ordering.
        """
        return sorted(
            {
                outcome.most_effective.cause_region
                for outcome in self.outcomes
                if outcome.most_effective is not None
            }
        )

    @property
    def adjusted_spread_by_region(self) -> Dict[str, Spread]:
        """
        Per cause region every answered ordering distinguishes, how the effect's
        adjusted probability varies over the orderings, the dataset's order included.
        """
        answered = [outcome for outcome in self.outcomes if outcome.answered]
        if not answered:
            return {}
        adjusted_by_region = [
            {
                effect.cause_region: effect.adjusted_probability
                for effect in outcome.effects
            }
            for outcome in answered
        ]
        shared_regions = set.intersection(
            *(set(by_region) for by_region in adjusted_by_region)
        )
        return {
            region: Spread.of([by_region[region] for by_region in adjusted_by_region])
            for region in sorted(shared_regions)
        }

    @property
    def largest_adjusted_difference(self) -> float:
        """
        Over the cause regions every answered ordering distinguishes, the widest range
        of the effect's adjusted probability; ``nan`` if no region is shared.
        """
        spreads = self.adjusted_spread_by_region
        if not spreads:
            return float("nan")
        return max(spread.range for spread in spreads.values())

    @property
    def mean_adjusted_standard_deviation(self) -> float:
        """
        Over the cause regions every answered ordering distinguishes, the mean standard
        deviation of the effect's adjusted probability; ``nan`` if no region is shared.
        """
        spreads = self.adjusted_spread_by_region
        if not spreads:
            return float("nan")
        return float(
            np.mean([spread.standard_deviation for spread in spreads.values()])
        )

    @property
    def argmax_flip_share(self) -> float:
        """
        The share of answered reorderings whose most effective region is not the one
        found in the dataset's order; ``nan`` if the dataset's order gave none.
        """
        baseline = self.in_dataset_order.most_effective
        answered = [
            outcome for outcome in self.answered if outcome.most_effective is not None
        ]
        if baseline is None or not answered:
            return float("nan")
        return float(
            np.mean(
                [
                    outcome.most_effective.cause_region != baseline.cause_region
                    for outcome in answered
                ]
            )
        )

    @property
    def trend_sign_flip_share(self) -> float:
        """
        The share of answered reorderings whose trend has the other sign than in the
        dataset's order; ``nan`` where the question has no trend.
        """
        baseline = self.in_dataset_order.trend
        trends = [
            outcome.trend for outcome in self.answered if outcome.trend is not None
        ]
        if baseline is None or not trends:
            return float("nan")
        return float(np.mean([np.sign(trend) != np.sign(baseline) for trend in trends]))


@dataclass
class PermutationReport:
    """
    How the pipelines' answers and likelihoods move when the parts of every example are
    put in another order.
    """

    ordering_count: int
    """
    How many random reorderings were tried.
    """

    questions: List[PermutedOutcomes] = field(default_factory=list)
    """
    One entry per pipeline and question.
    """

    whole_example_likelihoods: Dict[str, List[LikelihoodReport]] = field(
        default_factory=dict
    )
    """
    Per pipeline that models whole examples, its held-out likelihood report with the
    parts in the dataset's order first, then under each reordering.
    """

    def in_dataset_order(self, pipeline_name: str) -> LikelihoodReport:
        """
        :param pipeline_name: A pipeline modelling whole examples.
        :return: Its held-out whole-example likelihood with the parts in the dataset's
            order.
        """
        return self.whole_example_likelihoods[pipeline_name][0]

    def reordered(self, pipeline_name: str) -> List[LikelihoodReport]:
        """
        :param pipeline_name: A pipeline modelling whole examples.
        :return: Its held-out whole-example likelihood under each reordering.
        """
        return self.whole_example_likelihoods[pipeline_name][1:]

    def largest_likelihood_drop(self, pipeline_name: str) -> float:
        """
        :param pipeline_name: A pipeline modelling whole examples.
        :return: How far below the dataset-order likelihood the worst reordering took
            its mean whole-example log-likelihood.
        """
        means = [report.mean_log_likelihood for report in self.reordered(pipeline_name)]
        return float(
            self.in_dataset_order(pipeline_name).mean_log_likelihood - np.nanmin(means)
        )


def permutation_study(
    comparison: Comparison,
    dataset: ExampleDataset,
    cases: Sequence[CausalQueryCase],
    ordering_count: int = 20,
    train_fraction: float = 0.8,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    min_region_support: int = 10,
) -> PermutationReport:
    """
    Fit the pipelines that model the parts with the parts in the dataset's own order,
    then put every example's parts in a random order, several times over,
    and each time refit them and ask them the questions about one part again. The split
    is the same every time; only the order within an example changes.

    :param comparison: What is compared.
    :param dataset: The examples.
    :param cases: The questions to ask, the ones about one part.
    :param ordering_count: How many random reorderings to try.
    :param train_fraction: Share of examples to fit on.
    :param random_seed: Seed of the split, the orderings and the Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The study.
    """
    training, test = dataset.split(train_fraction, np.random.default_rng(random_seed))
    asker = QuestionAsker(
        random_seed=random_seed, min_region_support=min_region_support
    )
    report = PermutationReport(ordering_count=ordering_count)
    questions: Dict[Tuple[str, str], PermutedOutcomes] = {}
    orderings = [(training, test)]
    for ordering in range(ordering_count):
        random_state = np.random.default_rng([random_seed, ordering])
        orderings.append(
            (
                training.with_shuffled_parts(random_state),
                test.with_shuffled_parts(random_state),
            )
        )
    for ordering_index, (ordered_training, ordered_test) in enumerate(orderings):
        for pipeline in comparison.pipelines(
            ordered_training.examples, min_samples_per_leaf, plain_min_samples_per_leaf
        ):
            if not pipeline.models_parts:
                continue
            pipeline.fit(ordered_training.examples)
            likelihood = pipeline.log_likelihood(
                ordered_test.examples, ExampleView.WHOLE
            )
            report.whole_example_likelihoods.setdefault(pipeline.name, []).append(
                likelihood
            )
            for case in cases:
                outcome = asker.ask(pipeline, case)
                key = (pipeline.name, case.name)
                if ordering_index == 0:
                    questions[key] = PermutedOutcomes(
                        pipeline_name=pipeline.name,
                        case=case,
                        in_dataset_order=outcome,
                    )
                else:
                    questions[key].reordered.append(outcome)
    report.questions = list(questions.values())
    return report


# %% several splits


@dataclass
class SplitReport:
    """
    The comparison repeated over several random splits of the dataset.
    """

    reports: List[EvaluationReport] = field(default_factory=list)
    """
    One comparison per split.
    """

    @property
    def pipeline_names(self) -> List[str]:
        """
        The pipelines' names, in the order they were run.
        """
        return [pipeline.name for pipeline in self.reports[0].pipelines]

    def coverage(self, pipeline_name: str, view: ExampleView) -> List[float]:
        """
        :param pipeline_name: A pipeline's name.
        :param view: How much of an example to look at.
        :return: The pipeline's held-out coverage per split.
        """
        return [
            report.pipeline(pipeline_name).likelihoods[view].coverage
            for report in self.reports
        ]

    def shared_log_likelihood(
        self, pipeline_name: str, view: ExampleView
    ) -> List[float]:
        """
        :param pipeline_name: A pipeline's name.
        :param view: How much of an example to look at.
        :return: The pipeline's mean log-likelihood over the examples every pipeline
            modelling that view covers, per split.
        """
        return [
            report.shared_coverage_log_likelihoods[view][pipeline_name]
            for report in self.reports
        ]

    def outcomes(self, pipeline_name: str, case_index: int) -> List[QueryOutcome]:
        """
        :param pipeline_name: A pipeline's name.
        :param case_index: A question's position in the catalogue.
        :return: What the pipeline made of that question, per split.
        """
        return [
            report.pipeline(pipeline_name).outcomes[case_index]
            for report in self.reports
        ]


def split_study(
    comparison: Comparison,
    dataset: ExampleDataset,
    cases: Sequence[CausalQueryCase],
    random_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    train_fraction: float = 0.8,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    min_region_support: int = 10,
) -> SplitReport:
    """
    Repeat the comparison over several random splits.

    :param comparison: What is compared.
    :param dataset: The examples.
    :param cases: The questions to ask.
    :param random_seeds: One seed per split.
    :param train_fraction: Share of examples to fit on.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The study.
    """
    return SplitReport(
        reports=[
            evaluate(
                comparison,
                dataset,
                cases,
                train_fraction=train_fraction,
                random_seed=random_seed,
                min_samples_per_leaf=min_samples_per_leaf,
                plain_min_samples_per_leaf=plain_min_samples_per_leaf,
                time_repeats=False,
                min_region_support=min_region_support,
            )
            for random_seed in random_seeds
        ]
    )


# %% how much training data it takes


@dataclass(frozen=True)
class LearningCurvePoint:
    """
    One pipeline's held-out likelihood at one training-set size on one split.
    """

    pipeline_name: str
    """
    The pipeline.
    """

    train_fraction: float
    """
    Share of the examples it was fitted on.
    """

    random_seed: int
    """
    The seed of the split.
    """

    likelihoods: Dict[ExampleView, Optional[LikelihoodReport]]
    """
    Its likelihood report per view.
    """


@dataclass
class LearningCurveReport:
    """
    How the pipelines' held-out likelihoods grow with the training set.
    """

    points: List[LearningCurvePoint] = field(default_factory=list)
    """
    Every measurement.
    """

    @property
    def train_fractions(self) -> List[float]:
        """
        The training-set sizes measured, ascending.
        """
        return sorted({point.train_fraction for point in self.points})

    @property
    def pipeline_names(self) -> List[str]:
        """
        The pipelines measured, in the order they were run.
        """
        return list(dict.fromkeys(point.pipeline_name for point in self.points))

    def points_of(
        self, pipeline_name: str, train_fraction: float
    ) -> List[LearningCurvePoint]:
        """
        :param pipeline_name: A pipeline's name.
        :param train_fraction: A training-set size.
        :return: The pipeline's measurements at that size, one per split.
        """
        return [
            point
            for point in self.points
            if point.pipeline_name == pipeline_name
            and point.train_fraction == train_fraction
        ]


def learning_curve(
    comparison: Comparison,
    dataset: ExampleDataset,
    train_fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    random_seeds: Sequence[int] = (0, 1, 2),
    plain_min_samples_per_leaf: Optional[float] = None,
) -> LearningCurveReport:
    """
    Fit every pipeline's plain model on growing shares of the dataset and score the same
    held-out examples each time.

    The held-out examples are the last fifth of every split's shuffle, whatever the training
    share, so every size is scored on the same examples.

    :param comparison: What is compared.
    :param dataset: The examples.
    :param train_fractions: The training-set sizes to measure.
    :param random_seeds: One seed per split.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :return: The curve.
    """
    report = LearningCurveReport()
    for random_seed in random_seeds:
        available, test = dataset.split(
            max(train_fractions), np.random.default_rng(random_seed)
        )
        for train_fraction in train_fractions:
            training = available.examples[
                : round(train_fraction / max(train_fractions) * len(available.examples))
            ]
            for pipeline in comparison.pipelines(
                training, None, plain_min_samples_per_leaf
            ):
                pipeline.fit(training)
                report.points.append(
                    LearningCurvePoint(
                        pipeline_name=pipeline.name,
                        train_fraction=train_fraction,
                        random_seed=random_seed,
                        likelihoods=score_every_view(pipeline, test.examples),
                    )
                )
    return report


# %% error against known truth


@dataclass(frozen=True)
class RegionError:
    """
    A pipeline's adjusted probability for one cause region against the model's true
    interventional probability.
    """

    cause_region: str
    """
    The region, written out.
    """

    adjusted_probability: float
    """
    What the pipeline answered.
    """

    true_probability: float
    """
    What the model gives, by forced sampling.
    """

    support_count: int
    """
    How many training rows the region holds.
    """

    @property
    def absolute_error(self) -> float:
        """
        How far the answer is from the truth.
        """
        return abs(self.adjusted_probability - self.true_probability)


@dataclass
class GroundTruthOutcome:
    """
    What one pipeline made of one question on one synthetic dataset, scored against the
    truth.
    """

    pipeline_name: str
    """
    The pipeline that was asked.
    """

    case: CausalQueryCase
    """
    The question.
    """

    ordering: int
    """
    Which ordering of the parts the pipeline was fitted on: zero for the model's own,
    and counting up for random reorderings.
    """

    outcome: QueryOutcome
    """
    What the pipeline answered, or why it refused.
    """

    errors: List[RegionError] = field(default_factory=list)
    """
    One error per supported cause region the pipeline answered for.
    """

    @property
    def answered(self) -> bool:
        """
        Whether the pipeline answered with at least one supported region.
        """
        return bool(self.errors)

    @property
    def mean_absolute_error(self) -> float:
        """
        The mean absolute error over the supported regions; ``nan`` if none.
        """
        if not self.errors:
            return float("nan")
        return float(np.mean([error.absolute_error for error in self.errors]))

    @property
    def max_absolute_error(self) -> float:
        """
        The largest absolute error over the supported regions; ``nan`` if none.
        """
        if not self.errors:
            return float("nan")
        return max(error.absolute_error for error in self.errors)

    @property
    def rank_correlation(self) -> float:
        """
        Spearman's rank correlation between the answered and the true probabilities
        over the supported regions; ``nan`` with fewer than three.
        """
        answered = [error.adjusted_probability for error in self.errors]
        true = [error.true_probability for error in self.errors]
        if len(self.errors) < 3 or len(set(answered)) == 1 or len(set(true)) == 1:
            return float("nan")
        return float(spearmanr(answered, true).statistic)


@dataclass
class GroundTruthReport:
    """
    Every pipeline's error against the synthetic model's truth, over the model's
    settings and over reorderings of the parts.
    """

    example_count: int
    """
    How many examples each pipeline was fitted on per setting.
    """

    ordering_count: int
    """
    How many random reorderings were tried per setting.
    """

    outcomes: List[Tuple[Hashable, GroundTruthOutcome]] = field(default_factory=list)
    """
    Every scored answer, with the setting it was scored under.
    """

    descriptions: Dict[Hashable, Dict[str, str]] = field(default_factory=dict)
    """
    Per setting, its parameters by name, for the tables.
    """

    @property
    def configurations(self) -> List[Hashable]:
        """
        The settings, in the order they were run.
        """
        return list(dict.fromkeys(configuration for configuration, _ in self.outcomes))

    @property
    def pipeline_names(self) -> List[str]:
        """
        The pipelines, in the order they were run.
        """
        return list(
            dict.fromkeys(outcome.pipeline_name for _, outcome in self.outcomes)
        )

    def of(
        self,
        pipeline_name: str,
        configuration: Optional[Hashable] = None,
        ordering: Optional[int] = None,
    ) -> List[GroundTruthOutcome]:
        """
        :param pipeline_name: A pipeline's name.
        :param configuration: A setting to restrict to, or every setting.
        :param ordering: An ordering to restrict to, or every ordering.
        :return: The pipeline's scored answers.
        """
        return [
            outcome
            for scored_configuration, outcome in self.outcomes
            if outcome.pipeline_name == pipeline_name
            and (configuration is None or scored_configuration == configuration)
            and (ordering is None or outcome.ordering == ordering)
        ]

    @staticmethod
    def mean_absolute_error(outcomes: Sequence[GroundTruthOutcome]) -> float:
        """
        :param outcomes: Scored answers.
        :return: The mean absolute error over every supported region of every answered
            one; ``nan`` if none was answered.
        """
        errors = [
            error.absolute_error for outcome in outcomes for error in outcome.errors
        ]
        return float(np.mean(errors)) if errors else float("nan")

    @staticmethod
    def weighted_absolute_error(outcomes: Sequence[GroundTruthOutcome]) -> float:
        """
        :param outcomes: Scored answers.
        :return: The mean absolute error over every supported region of every answered
            one, each region weighted by how many training rows it holds, so that the
            sparse extremes of a count weigh as little as they do in the data; ``nan``
            if none was answered.
        """
        errors = [error for outcome in outcomes for error in outcome.errors]
        if not errors:
            return float("nan")
        weights = np.array([error.support_count for error in errors], dtype=float)
        values = np.array([error.absolute_error for error in errors])
        return float((weights * values).sum() / weights.sum())

    @staticmethod
    def max_absolute_error(outcomes: Sequence[GroundTruthOutcome]) -> float:
        """
        :param outcomes: Scored answers.
        :return: The largest absolute error over every supported region of every
            answered one; ``nan`` if none was answered.
        """
        errors = [
            error.absolute_error for outcome in outcomes for error in outcome.errors
        ]
        return max(errors) if errors else float("nan")

    @staticmethod
    def mean_rank_correlation(outcomes: Sequence[GroundTruthOutcome]) -> float:
        """
        :param outcomes: Scored answers.
        :return: The mean rank correlation with the truth over the answers that have
            one; ``nan`` if none has.
        """
        correlations = [
            outcome.rank_correlation
            for outcome in outcomes
            if not math.isnan(outcome.rank_correlation)
        ]
        return float(np.mean(correlations)) if correlations else float("nan")

    def worst_ordering_mean_absolute_error(self, pipeline_name: str) -> float:
        """
        :param pipeline_name: A pipeline's name.
        :return: Its mean absolute error under the reordering it did worst on, over
            every setting; the model's own order if it was never refitted.
        """
        orderings = sorted({outcome.ordering for outcome in self.of(pipeline_name)})
        return max(
            self.mean_absolute_error(self.of(pipeline_name, ordering=ordering))
            for ordering in orderings
        )

    def answered_share(self, pipeline_name: str) -> float:
        """
        :param pipeline_name: A pipeline's name.
        :return: The share of its questions it answered with at least one supported
            region.
        """
        outcomes = self.of(pipeline_name)
        return float(np.mean([outcome.answered for outcome in outcomes]))


class KnownTruth(ABC):
    """
    A model of the domain whose interventional probabilities are known, under several
    settings, so that a pipeline's answers can be scored against them.
    """

    @property
    @abstractmethod
    def configurations(self) -> Sequence[Hashable]:
        """
        The settings to run, each a hashable description of one model.
        """

    @abstractmethod
    def describe(self, configuration: Hashable) -> Dict[str, str]:
        """
        :param configuration: One setting.
        :return: Its parameters, by name, for the report's tables.
        """

    @abstractmethod
    def examples(
        self, configuration: Hashable, count: int, random_state: np.random.Generator
    ) -> List[Any]:
        """
        :param configuration: One setting.
        :param count: How many examples to sample.
        :param random_state: Source of randomness.
        :return: Examples sampled from the model under that setting.
        """

    @abstractmethod
    def probability(
        self,
        configuration: Hashable,
        case: CausalQueryCase,
        effect: InterventionalEffect,
    ) -> float:
        """
        :param configuration: One setting.
        :param case: A question.
        :param effect: One region of its cause, as a pipeline answered for it.
        :return: The true probability of the question's effect under the intervention
            setting the cause to that region.
        """

    def score(
        self,
        configuration: Hashable,
        outcome: QueryOutcome,
        pipeline_name: str,
        ordering: int,
    ) -> GroundTruthOutcome:
        """
        :param configuration: The setting the pipeline was fitted under.
        :param outcome: What the pipeline answered.
        :param pipeline_name: The pipeline.
        :param ordering: Which ordering of the parts it was fitted on.
        :return: The answer scored against the truth, over its supported regions.
        """
        return GroundTruthOutcome(
            pipeline_name=pipeline_name,
            case=outcome.case,
            ordering=ordering,
            outcome=outcome,
            errors=[
                RegionError(
                    cause_region=effect.cause_region,
                    adjusted_probability=effect.adjusted_probability,
                    true_probability=self.probability(
                        configuration, outcome.case, effect
                    ),
                    support_count=effect.support_count,
                )
                for effect in outcome.supported_effects
            ],
        )


def ground_truth_study(
    comparison: Comparison,
    truth: KnownTruth,
    cases: Sequence[CausalQueryCase],
    example_count: int = 400,
    ordering_count: int = 5,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    min_region_support: int = 10,
) -> GroundTruthReport:
    """
    Fit every pipeline on examples sampled from a model whose interventional
    probabilities are known, under each of its settings, ask the questions, and score
    every answer against the truth; then reorder the parts and score the pipelines whose
    fit can see the order again.

    :param comparison: What is compared.
    :param truth: The model and its settings.
    :param cases: The questions to ask.
    :param example_count: How many examples to fit on per setting.
    :param ordering_count: How many random reorderings to try per setting.
    :param random_seed: Seed of the sampled examples, the reorderings and the
        Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be scored.
    :return: The study.
    """
    asker = QuestionAsker(
        random_seed=random_seed, min_region_support=min_region_support
    )
    report = GroundTruthReport(
        example_count=example_count, ordering_count=ordering_count
    )
    for configuration in truth.configurations:
        report.descriptions[configuration] = truth.describe(configuration)
        dataset = ExampleDataset(
            comparison.domain,
            truth.examples(
                configuration, example_count, np.random.default_rng(random_seed)
            ),
        )
        orderings = [dataset] + [
            dataset.with_shuffled_parts(np.random.default_rng([random_seed, ordering]))
            for ordering in range(ordering_count)
        ]
        for ordering, ordered in enumerate(orderings):
            estimators = comparison.pipelines(
                ordered.examples, min_samples_per_leaf, plain_min_samples_per_leaf
            ) + baselines_of(comparison.domain, min_region_support)
            for estimator in estimators:
                if ordering > 0 and getattr(estimator, "order_invariant", False):
                    continue
                estimator.fit(ordered.examples)
                for case in cases:
                    answered = (
                        asker.ask(estimator, case)
                        if isinstance(estimator, CausalQueryPipeline)
                        else estimator.ask(case)
                    )
                    report.outcomes.append(
                        (
                            configuration,
                            truth.score(
                                configuration, answered, estimator.name, ordering
                            ),
                        )
                    )
    return report


# %% how many samples grounding needs


@dataclass(frozen=True)
class SampleCountOutcome:
    """
    What the relational circuit answered one question with at one number of grounding
    samples.
    """

    sample_count: int
    """
    How many samples grounding drew for each count the query left open.
    """

    outcome: QueryOutcome
    """
    The answer, with its duration.
    """


@dataclass
class MonteCarloReport:
    """
    How the relational circuit's answers settle as grounding draws more samples.
    """

    reference_sample_count: int
    """
    The largest number of samples tried, whose answer every other is measured against.
    """

    stability_tolerance: float = 0.01
    """
    How close to the reference an answer must be, on every region, to count as
    settled.
    """

    outcomes: Dict[str, List[SampleCountOutcome]] = field(default_factory=dict)
    """
    Per question name, the answer at each number of samples, ascending.
    """

    def deviation(self, case_name: str, sample_count: int) -> float:
        """
        :param case_name: A question.
        :param sample_count: A number of samples tried for it.
        :return: The largest difference, over the cause regions both answers
            distinguish, between the adjusted probability at that number of samples
            and at the reference; ``nan`` if either was refused or they share no
            region.
        """
        by_count = {
            scored.sample_count: scored.outcome for scored in self.outcomes[case_name]
        }
        answered = by_count[sample_count]
        reference = by_count[self.reference_sample_count]
        if not answered.answered or not reference.answered:
            return float("nan")
        adjusted = {
            effect.cause_region: effect.adjusted_probability
            for effect in answered.effects
        }
        expected = {
            effect.cause_region: effect.adjusted_probability
            for effect in reference.effects
        }
        shared = set(adjusted) & set(expected)
        if not shared:
            return float("nan")
        return max(abs(adjusted[region] - expected[region]) for region in shared)

    def settled_from(self, case_name: str) -> Optional[int]:
        """
        :param case_name: A question.
        :return: The smallest number of samples from which every larger number tried
            stays within the tolerance of the reference, or ``None`` if only the
            reference does.
        """
        counts = [scored.sample_count for scored in self.outcomes[case_name]]
        settled = None
        for count in reversed(counts[:-1]):
            deviation = self.deviation(case_name, count)
            if math.isnan(deviation) or deviation > self.stability_tolerance:
                break
            settled = count
        return settled


def monte_carlo_study(
    comparison: Comparison,
    dataset: ExampleDataset,
    cases: Sequence[CausalQueryCase],
    sample_counts: Sequence[int] = (50, 200, 1000, 2000, 8000, 32000),
    train_fraction: float = 0.8,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    min_region_support: int = 10,
) -> MonteCarloReport:
    """
    Fit the relational circuit once and ask it the same questions with grounding
    drawing more and more samples for the counts a query leaves open, to find how many
    it takes for the answers to settle.

    :param comparison: What is compared.
    :param dataset: The examples.
    :param cases: The questions to ask, ones that leave every count open.
    :param sample_counts: The numbers of samples to try; the largest is the reference.
    :param train_fraction: Share of examples to fit on.
    :param random_seed: Seed of the split and of the Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipeline's own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipeline's own default if not given.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The study.
    """
    training, _ = dataset.split(train_fraction, np.random.default_rng(random_seed))
    asker = QuestionAsker(
        random_seed=random_seed, min_region_support=min_region_support
    )
    [pipeline] = [
        candidate
        for candidate in comparison.pipelines(
            training.examples, min_samples_per_leaf, plain_min_samples_per_leaf
        )
        if isinstance(candidate, RelationalPipeline)
    ]
    pipeline.fit(training.examples)
    report = MonteCarloReport(reference_sample_count=max(sample_counts))
    for sample_count in sorted(sample_counts):
        pipeline.set_monte_carlo_sample_count(sample_count)
        for case in cases:
            report.outcomes.setdefault(case.name, []).append(
                SampleCountOutcome(sample_count, asker.ask(pipeline, case))
            )
    return report


# %% cost against relational size


@dataclass(frozen=True)
class ScalingPoint:
    """
    What one pipeline cost on examples of one size.
    """

    pipeline_name: str
    """
    The pipeline.
    """

    size: int
    """
    The typical number of parts in an example.
    """

    fit: FitReport
    """
    What fitting the plain model cost.
    """

    query_duration: float
    """
    Wall-clock seconds one part-level question took, its cause-specific model
    fitted first.
    """

    repeat_duration: float
    """
    Wall-clock seconds the same question took asked again, every model fitted.
    """


@dataclass
class ScalingReport:
    """
    How the pipelines' fit time, query time and circuit size grow with the number of
    parts in an example.
    """

    example_count: int
    """
    How many examples each pipeline was fitted on per size.
    """

    points: List[ScalingPoint] = field(default_factory=list)
    """
    Every measurement.
    """

    @property
    def sizes(self) -> List[int]:
        """
        The sizes measured, ascending.
        """
        return sorted({point.size for point in self.points})

    @property
    def pipeline_names(self) -> List[str]:
        """
        The pipelines measured, in the order they were run.
        """
        return list(dict.fromkeys(point.pipeline_name for point in self.points))

    def point(self, pipeline_name: str, size: int) -> ScalingPoint:
        """
        :param pipeline_name: A pipeline's name.
        :param size: A size.
        :return: The pipeline's measurement at that size.
        """
        [point] = [
            point
            for point in self.points
            if point.pipeline_name == pipeline_name and point.size == size
        ]
        return point


def scaling_study(
    comparison: Comparison,
    examples_of_size: Callable[[int, int, np.random.Generator], List[Any]],
    case: CausalQueryCase,
    sizes: Sequence[int] = (5, 10, 20, 50),
    example_count: int = 400,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
) -> ScalingReport:
    """
    Fit the pipelines that model the parts on synthetic examples of growing size and
    time one part-level question on each.

    :param comparison: What is compared.
    :param examples_of_size: How to sample a number of examples with a typical number
        of parts.
    :param case: The question to time.
    :param sizes: The typical numbers of parts per example to measure.
    :param example_count: How many examples to fit on per size.
    :param random_seed: Seed of the sampled examples and the Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :return: The study.
    """
    asker = QuestionAsker(random_seed=random_seed, min_region_support=1)
    report = ScalingReport(example_count=example_count)
    for size in sizes:
        examples = examples_of_size(
            size, example_count, np.random.default_rng(random_seed)
        )
        for pipeline in comparison.pipelines(
            examples, min_samples_per_leaf, plain_min_samples_per_leaf
        ):
            if not pipeline.models_parts:
                continue
            fit = pipeline.fit(examples)
            outcome = asker.ask(pipeline, case)
            report.points.append(
                ScalingPoint(
                    pipeline_name=pipeline.name,
                    size=size,
                    fit=FitReport(
                        training_example_count=fit.training_example_count,
                        model_count=fit.model_count,
                        training_duration=fit.training_duration,
                        size=fit.size,
                    ),
                    query_duration=outcome.duration,
                    repeat_duration=asker.time_repeat(pipeline, case),
                )
            )
    return report
