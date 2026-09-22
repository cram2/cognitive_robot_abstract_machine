"""
Rendering a comparison whose outcomes are given by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from krrood.entity_query_language.factories import count_range, entity, variable
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)
from typing_extensions import List

from experiments.causal_reasoning.comparison.domain import ExampleView, RelationalDomain
from experiments.causal_reasoning.comparison.evaluation import (
    EvaluationReport,
    InterventionalEffect,
    PipelineReport,
    QueryOutcome,
)
from experiments.causal_reasoning.comparison.pipelines import FitReport
from experiments.causal_reasoning.comparison.queries import (
    AdjustedCountCase,
    Confounder,
)
from experiments.causal_reasoning.comparison.report import MarkdownReport, ReportText

SIZE = Confounder(name="size", noun="the size")


@dataclass
class Part:
    """
    One exchangeable part of a :class:`Whole`.
    """

    flag: bool
    """
    The part's one attribute.
    """


@dataclass
class Whole:
    """
    An example with one kind of part.
    """

    size: int
    """
    The example's own attribute.
    """

    effect: bool
    """
    Whether the example shows the effect.
    """

    parts: List[Part]
    """
    The parts.
    """


@dataclass
class WholeAggregations(AggregationStatistic[Whole]):
    """
    The one count over the parts.
    """

    @aggregation_statistic("parts")
    def count(self) -> int:
        """
        Count of flagged parts.
        """
        flag = variable(Part, self.instance.parts).flag
        [result] = entity(count_range(flag)).where(flag == True).tolist()
        return result


def whole_domain() -> RelationalDomain:
    return RelationalDomain(
        example_class=Whole,
        aggregation_class=WholeAggregations,
        effect_field="effect",
        noun="whole",
        plural="wholes",
    )


@dataclass(frozen=True, kw_only=True)
class CountQuestion(AdjustedCountCase):
    """
    A count question whose query is never built.
    """

    @property
    def domain(self) -> RelationalDomain:
        return whole_domain()

    @property
    def name(self) -> str:
        adjusting = "_".join(confounder.name for confounder in self.confounders)
        return f"{self.statistic_name}_{self.open_part_count}_{adjusting}"

    @property
    def question(self) -> str:
        return self.name

    def build(self) -> Match:
        raise NotImplementedError

    def describe_cause(self, region: str) -> str:
        return region

    @property
    def effect(self) -> str:
        return "the effect"


def _answered(
    case: AdjustedCountCase,
    pipeline_name: str = "relational circuit",
    best_region: str = "0",
    repeat_duration: float = 0.0,
) -> QueryOutcome:
    return QueryOutcome(
        case=case,
        pipeline_name=pipeline_name,
        duration=0.0,
        repeat_duration=repeat_duration,
        min_region_support=1,
        best_region=best_region,
        effect_probability_given_best_region=0.5,
        effects=[
            InterventionalEffect(
                cause_region=region,
                region_probability=0.5,
                naive_probability=0.5,
                adjusted_probability=0.9 if region == best_region else 0.1,
                support_count=10,
                ordinal=float(region),
            )
            for region in ("0", "1")
        ],
    )


def _pipeline(name: str, outcomes) -> PipelineReport:
    return PipelineReport(
        name=name,
        fit=FitReport(training_example_count=10),
        likelihoods={view: None for view in ExampleView},
        outcomes=outcomes,
    )


def _render(pipelines) -> str:
    report = EvaluationReport(
        random_seed=0,
        training_example_count=10,
        test_example_count=2,
        effect_rate=0.5,
        pipelines=pipelines,
        shared_coverage_log_likelihoods={view: {} for view in ExampleView},
    )
    text = ReportText(title="test", introduction=(), effect_summary="nothing")
    return MarkdownReport(whole_domain(), text, report).render()


def _report(cases) -> str:
    return _render(
        [_pipeline("relational circuit", [_answered(case) for case in cases])]
    )


def test_pipelines_that_disagree_are_grouped_by_their_answer():
    case = CountQuestion(statistic_name="count", confounders=(SIZE,))
    rendered = _render(
        [
            _pipeline("relational circuit", [_answered(case, "relational circuit")]),
            _pipeline("propositional tree", [_answered(case, "propositional tree")]),
            _pipeline(
                "regression adjustment",
                [_answered(case, "regression adjustment", "1", float("nan"))],
            ),
        ]
    )
    assert (
        "the relational circuit and the propositional tree say 0 (0.90, 0.90); "
        "the regression adjustment says 1 (0.90)." in rendered
    )


def test_a_pipeline_that_is_not_timed_again_has_no_latency_finding():
    case = CountQuestion(statistic_name="count", confounders=(SIZE,))
    rendered = _render(
        [
            _pipeline("relational circuit", [_answered(case, repeat_duration=1.5)]),
            _pipeline(
                "regression adjustment",
                [_answered(case, "regression adjustment", "0", float("nan"))],
            ),
        ]
    )
    assert "- The relational circuit takes 1.50 seconds" in rendered
    assert "- The regression adjustment takes" not in rendered


def test_the_adjustments_table_holds_one_column_per_adjustment_of_one_question():
    rendered = _report(
        [
            CountQuestion(statistic_name="count", confounders=(SIZE,)),
            CountQuestion(statistic_name="count", confounders=()),
        ]
    )
    section = rendered.split("## What adjusting for changes")[1].split("\n## ")[0]
    [header] = [
        line for line in section.splitlines() if line.startswith("| cause region")
    ]
    assert header == "| cause region | n | naive | adjusted for the size | unadjusted |"


def test_the_same_count_asked_of_another_part_count_is_another_question():
    rendered = _report(
        [
            CountQuestion(statistic_name="count", confounders=(SIZE,)),
            CountQuestion(statistic_name="count", confounders=(), open_part_count=3),
        ]
    )
    assert "## What adjusting for changes" not in rendered


@pytest.mark.parametrize("count", [1, 3])
def test_a_single_adjustment_renders_no_adjustments_section(count):
    rendered = _report(
        [
            CountQuestion(
                statistic_name="count", confounders=(SIZE,), open_part_count=count
            )
        ]
    )
    assert "## What adjusting for changes" not in rendered
