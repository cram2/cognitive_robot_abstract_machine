"""
Writing the comparison out as Markdown, with every table explained and the answers put
into words.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from typing_extensions import Dict, Iterable, List, Optional, Sequence, Tuple

from experiments.causal_reasoning.comparison.domain import ExampleView, RelationalDomain
from experiments.causal_reasoning.comparison.evaluation import (
    EvaluationReport,
    GroundTruthReport,
    InterventionalEffect,
    LearningCurveReport,
    MonteCarloReport,
    PermutationReport,
    PipelineReport,
    QueryOutcome,
    ScalingReport,
    SplitReport,
)
from experiments.causal_reasoning.comparison.pipelines import LikelihoodReport
from experiments.causal_reasoning.comparison.queries import AdjustedCountCase


class Verdict(StrEnum):
    """
    How an outcome is marked in the answerability table.
    """

    ANSWERED = "answered"
    REFUSED = "refused"


@dataclass(frozen=True)
class ReportText:
    """
    What an experiment says about itself in its report: the prose around the tables
    that describes the dataset and the pipelines rather than the numbers.
    """

    title: str
    """
    The document's title.
    """

    introduction: Tuple[str, ...]
    """
    The paragraphs before the setup, one string each: what the dataset is, what the
    pipelines see of it and how the questions are asked.
    """

    effect_summary: str
    """
    What the effect-rate tables group the examples by, in words, completing "grouped
    by ...".
    """

    answerability_note: str = ""
    """
    What the pattern of answers and refusals means, after the answerability table.
    """

    reordering_note: str = ""
    """
    What the order the dataset lists the parts in means, if anything, in the
    reordering section.
    """

    ground_truth_note: str = ""
    """
    How the model with known truth is built and what to expect of it, before its
    tables.
    """

    learning_curve_note: str = ""
    """
    Why the pipelines need the amounts of data they do, before the curve.
    """

    scaling_note: str = ""
    """
    Why the pipelines grow the way they do, before the scaling table.
    """


@dataclass
class MarkdownReport:
    """
    Renders a comparison as Markdown.
    """

    domain: RelationalDomain
    """
    The example and its parts.
    """

    text: ReportText
    """
    What the experiment says about itself.
    """

    report: EvaluationReport
    """
    The comparison on one split, with every question asked and timed.
    """

    permutations: Optional[PermutationReport] = None
    """
    The same questions under random orderings of the parts, if run.
    """

    splits: Optional[SplitReport] = None
    """
    The comparison repeated over several splits, if run.
    """

    curve: Optional[LearningCurveReport] = None
    """
    The likelihoods over growing training sets, if run.
    """

    truth: Optional[GroundTruthReport] = None
    """
    The error against the synthetic model's known interventional probabilities, if run.
    """

    monte_carlo: Optional[MonteCarloReport] = None
    """
    How the relational circuit's answers settle with the number of grounding samples, if
    run.
    """

    scaling: Optional[ScalingReport] = None
    """
    How fit time, query time and circuit size grow with the number of parts in an
    example, if run.
    """

    def render(self) -> str:
        """
        :return: The whole document.
        """
        sections = [
            self._setup(),
            self._effect_rates(),
            self._answerability(),
            self._trends(),
            self._adjustments(),
            self._quantities(),
            self._latencies(),
        ]
        if self.permutations is not None:
            sections.append(self._permutations())
        if self.splits is not None:
            sections.append(self._splits())
        if self.curve is not None:
            sections.append(self._learning_curve())
        if self.truth is not None:
            sections.append(self._ground_truth())
        if self.monte_carlo is not None:
            sections.append(self._monte_carlo())
        if self.scaling is not None:
            sections.append(self._scaling())
        sections.append(self._findings())
        for case_index in range(len(self._first_pipeline.outcomes)):
            sections.append(self._effects(case_index))
        return "\n".join(line for section in sections for line in section + [""])

    @property
    def _first_pipeline(self) -> PipelineReport:
        return self.report.pipelines[0]

    @staticmethod
    def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
        """
        :param header: The column titles.
        :param rows: The cells, row by row.
        :return: The lines of a Markdown table.
        """
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return lines

    @staticmethod
    def _number(value: float, digits: int = 3) -> str:
        """
        :param value: A number, possibly ``nan``.
        :param digits: How many decimals to keep.
        :return: The number rounded, or a dash for ``nan``.
        """
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"

    @staticmethod
    def _percent(share: float) -> str:
        """
        :param share: A share between 0 and 1.
        :return: The share as a percentage with one decimal.
        """
        return f"{100 * share:.1f}%"

    def _mean_and_spread(self, values: Sequence[float], digits: int = 2) -> str:
        """
        :param values: Numbers, possibly ``nan``.
        :param digits: How many decimals to keep.
        :return: Their mean and standard deviation over the finite ones, or a dash.
        """
        finite = [value for value in values if not math.isnan(value)]
        if not finite:
            return "-"
        return (
            f"{self._number(float(np.mean(finite)), digits)} ± "
            f"{self._number(float(np.std(finite)), digits)}"
        )

    @staticmethod
    def _note(note: str) -> str:
        """
        :param note: A sentence the experiment adds to a section, or nothing.
        :return: The note with a leading space, or an empty string.
        """
        return f" {note}" if note else ""

    @staticmethod
    def _region_order(region: str) -> Tuple[int, float, str]:
        """
        :param region: A cause region, written out.
        :return: A sort key putting numeric regions in numeric order ahead of the others
            in alphabetical order.
        """
        if re.fullmatch(r"-?\d+(\.\d+)?", region):
            return (0, float(region), region)
        return (1, 0.0, region)

    # %% sections

    def _setup(self) -> List[str]:
        report = self.report
        plural = self.domain.plural
        lines = [f"# {self.text.title}", ""]
        for paragraph in self.text.introduction:
            lines += [paragraph, ""]
        lines += [
            "## Setup",
            "",
            f"- {plural}: {report.training_example_count + report.test_example_count}"
            f" ({report.training_example_count} to fit on, "
            f"{report.test_example_count} held out)",
            f"- {plural} where {self.domain.effect_phrase}: "
            f"{self._percent(report.effect_rate)}",
            "- fewest training rows per leaf, as a share of the rows fitted on: "
            f"{report.min_samples_per_leaf} in a cause-specific model, "
            f"{report.plain_min_samples_per_leaf} in the plain model that scores "
            f"held-out {plural}",
            f"- split seed: {report.random_seed}",
            f"- fewest training {plural} a cause region may hold for its effect to be "
            f"read as an answer: {report.min_region_support}; a region below that is "
            "marked † in the tables and takes no part in any summary",
        ]
        return lines

    def _effect_rates(self) -> List[str]:
        plural = self.domain.plural
        lines = [
            f"## How often {self.domain.effect_phrase}",
            "",
            f"The {plural} themselves, before any model: the share where "
            f"{self.domain.effect_phrase}, grouped by {self.text.effect_summary}. "
            "This is the signal the models are asked to explain.",
            "",
        ]
        for title, rates in self.report.effect_rates_by.items():
            lines += self._table(
                [title, plural, "effect"],
                [
                    [str(value), str(rate.example_count), self._percent(rate.rate)]
                    for value, rate in rates.items()
                ],
            )
            lines.append("")
        return lines

    def _answerability(self) -> List[str]:
        lines = [
            "## Which questions each pipeline can answer",
            "",
            "One row per question, one column per pipeline. An answered cell says, in "
            "words, which setting of the cause makes the effect most likely after "
            "adjustment and how likely, against the least favourable setting, over "
            f"the regions that hold enough training {self.domain.plural} to be read; "
            "a refused cell "
            "says why the pipeline could not answer at all.",
            "",
        ]
        names = [pipeline.name for pipeline in self.report.pipelines]
        rows = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            row = [outcome.case.question]
            for pipeline in self.report.pipelines:
                row.append(self._verdict(pipeline.outcomes[case_index]))
            rows.append(row)
        lines += self._table(["question"] + names, rows)
        if self.text.answerability_note:
            lines += ["", self.text.answerability_note]
        return lines

    def _verdict(self, outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its answer put into words, or its refusal, in one cell.
        """
        if not outcome.answered:
            return f"{Verdict.REFUSED}: {outcome.refusal}."
        case = outcome.case
        best = outcome.most_effective
        worst = outcome.least_effective
        if best is None:
            return (
                f"{Verdict.ANSWERED}, but no region of the cause holds "
                f"{outcome.min_region_support} training {self.domain.plural}."
            )
        return (
            f"{Verdict.ANSWERED}: with {case.describe_cause(best.cause_region)}, "
            f"{case.effect} with probability "
            f"{self._number(best.adjusted_probability, 2)}, the highest of any setting; "
            f"with {case.describe_cause(worst.cause_region)} it is only "
            f"{self._number(worst.adjusted_probability, 2)}."
        )

    def _trends(self) -> List[str]:
        lines = [
            "## Trend and contrast",
            "",
            "The most effective setting is an argmax over up to twenty sparse regions "
            "and moves with the split. Two summaries that do not: *trend* is "
            "Spearman's rank correlation between the cause's value and the adjusted "
            "probability over the supported regions, for a numeric cause; *contrast* "
            "is the adjusted probability at the highest supported region minus at "
            "the lowest (for a symbolic cause, at the most effective minus at the "
            "least), with Newcombe's interval from the Wilson intervals of the two "
            "regions' support.",
            "",
        ]
        header = ["question"]
        for pipeline in self.report.pipelines:
            header += [f"{pipeline.name}, trend", f"{pipeline.name}, contrast"]
        rows = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            row = [outcome.case.name]
            for pipeline in self.report.pipelines:
                asked = pipeline.outcomes[case_index]
                row += [self._trend_cell(asked), self._contrast_cell(asked)]
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _trend_cell(self, outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its trend, or a dash.
        """
        if not outcome.answered or outcome.trend is None:
            return "-"
        return self._number(outcome.trend, 2)

    def _contrast_cell(self, outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its contrast with its interval and the regions it is between, or a
            dash.
        """
        if not outcome.answered or outcome.contrast is None:
            return "-"
        contrast = outcome.contrast
        return (
            f"{self._number(contrast.difference, 2)} "
            f"[{self._number(contrast.interval.lower, 2)}, "
            f"{self._number(contrast.interval.upper, 2)}] "
            f"({contrast.low_region} → {contrast.high_region})"
        )

    def _adjustments(self) -> List[str]:
        by_statistic: Dict[Tuple[str, int], List[QueryOutcome]] = {}
        for outcome in self._first_pipeline.outcomes:
            if isinstance(outcome.case, AdjustedCountCase):
                by_statistic.setdefault(
                    (outcome.case.statistic_name, outcome.case.open_part_count), []
                ).append(outcome)
        if not any(len(asked) > 1 for asked in by_statistic.values()):
            return []
        lines = [
            "## What adjusting for changes",
            "",
            "The same count question under each set of confounders it was asked with, "
            f"read off the {self._first_pipeline.name}. *n* is how many training "
            f"{self.domain.plural} hold that value of the cause; † marks a region "
            "below the support threshold.",
            "",
        ]
        for (statistic_name, _), asked in by_statistic.items():
            cases = [outcome.case for outcome in asked]
            if len(asked) < 2 or not all(outcome.answered for outcome in asked):
                continue
            lines += [f"### {statistic_name}", ""]
            by_region: Dict[str, Dict[str, InterventionalEffect]] = {}
            for outcome in asked:
                for effect in outcome.effects:
                    by_region.setdefault(effect.cause_region, {})[
                        outcome.case.name
                    ] = effect
            header = ["cause region", "n", "naive"] + [
                (
                    "adjusted for "
                    + " and ".join(confounder.noun for confounder in case.confounders)
                    if case.confounders
                    else "unadjusted"
                )
                for case in cases
            ]
            rows = []
            for region in sorted(by_region, key=self._region_order):
                effects = by_region[region]
                first = next(iter(effects.values()))
                row = [
                    self._region_label(first, asked[0].min_region_support),
                    str(first.support_count),
                    self._number(first.naive_probability),
                ]
                for case in cases:
                    effect = effects.get(case.name)
                    row.append(
                        "-"
                        if effect is None
                        else self._number(effect.adjusted_probability)
                    )
                rows.append(row)
            lines += self._table(header, rows)
            lines.append("")
        return lines

    @staticmethod
    def _region_label(effect: InterventionalEffect, min_region_support: int) -> str:
        """
        :param effect: One region's effect.
        :param min_region_support: The support threshold.
        :return: The region, marked if it is below the threshold.
        """
        if effect.is_supported(min_region_support):
            return effect.cause_region
        return f"{effect.cause_region} †"

    def _quantities(self) -> List[str]:
        plural, noun = self.domain.plural, self.domain.noun
        lines = [
            "## Fit and likelihood",
            "",
            "What each pipeline cost. *Models fitted* counts the plain model plus one "
            "support-deterministic model per distinct cause the questions asked about "
            "and the pipeline could fit; *training seconds* and the *nodes*/*edges* of "
            "every fitted circuit are summed over them, which for the relational "
            "circuit includes the part templates.",
            "",
        ]
        lines += self._table(
            ["pipeline", "models fitted", "training seconds", "nodes", "edges"],
            [
                [
                    pipeline.name,
                    str(pipeline.fit.model_count),
                    self._number(pipeline.fit.training_duration, 2),
                    str(pipeline.fit.size.node_count),
                    str(pipeline.fit.size.edge_count),
                ]
                for pipeline in self.report.pipelines
            ],
        )
        lines += [
            "",
            f"How well each explains {plural} it never saw, on three views of one "
            f"{noun}: its own scalars, which every pipeline models; its scalars and "
            f"counts; and the whole {noun}, parts included, which only the pipelines "
            "that model the parts can score. The relational circuit scores a whole "
            f"{noun} as its class circuit over the scalars and counts times each part "
            "template over one part given the counts; the unrolled tree scores it as "
            f"one row. *Held-out coverage* is the share of held-out {plural} that lie "
            "inside the plain model's support at all, since a tree's leaves span only "
            f"the value ranges they were fitted on, and a whole {noun} is covered only "
            "if every one of its parts is. The *mean log-likelihood* is over the "
            f"covered {plural} only; the last column restricts it to the {plural} every "
            "pipeline in the table covers, so the numbers are over the same rows.",
        ]
        for view in ExampleView:
            scored = [
                pipeline
                for pipeline in self.report.pipelines
                if pipeline.likelihoods[view] is not None
            ]
            lines += ["", f"### {self.domain.view_label(view)}", ""]
            lines += self._table(
                [
                    "pipeline",
                    "held-out coverage",
                    "mean log-likelihood (covered)",
                    "mean log-likelihood (covered by all)",
                ],
                [
                    [
                        pipeline.name,
                        self._percent(pipeline.likelihoods[view].coverage),
                        self._number(pipeline.likelihoods[view].mean_log_likelihood, 2),
                        self._number(
                            self.report.shared_coverage_log_likelihoods[view][
                                pipeline.name
                            ],
                            2,
                        ),
                    ]
                    for pipeline in scored
                ],
            )
        return lines

    def _latencies(self) -> List[str]:
        lines = [
            "## Seconds per question",
            "",
            "Wall-clock time from asking to the answer or the refusal. The *first ask* "
            "of a cause includes fitting that cause's own support-deterministic model; "
            "*asked again* repeats the question with every model fitted, so only "
            "grounding (for the relational circuit), verification and backdoor "
            "adjustment remain. A refusal is fast when it is a schema check; a "
            "relational answer draws Monte-Carlo samples for every count the query "
            "leaves open and grounds one part template per sampled value, which is "
            "where its time goes.",
            "",
        ]
        header = ["question"]
        for pipeline in self.report.pipelines:
            header += [f"{pipeline.name}, first ask", f"{pipeline.name}, asked again"]
        rows = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            row = [outcome.case.name]
            for pipeline in self.report.pipelines:
                asked = pipeline.outcomes[case_index]
                row += [
                    self._number(asked.duration, 2),
                    self._number(asked.repeat_duration, 2),
                ]
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _permutations(self) -> List[str]:
        permutations = self.permutations
        noun, plural = self.domain.noun, self.domain.plural
        first_part = self.domain.part_fields[0]
        part_noun = self.domain.part_noun(first_part)
        lines = [
            "## Does the order of the parts matter?",
            "",
            f"Every {noun}'s parts were put in a random order, "
            f"{permutations.ordering_count} times over, and each time the pipelines "
            "that model the parts were refitted on the same split and asked the "
            "questions about parts again; the parts in the order the dataset lists "
            "them is the baseline every reordering is measured against. A relational "
            f"circuit treats the {part_noun}s as exchangeable, so nothing about it can "
            f"depend on the order; an unrolled table's column `{first_part}[0]` holds a "
            f"different {part_noun} of every {noun} after each reordering. Per question "
            "and pipeline: how many reorderings were answered; over the cause regions "
            "every answered ordering distinguishes, the mean standard deviation and "
            "the widest range of the adjusted probability; the share of reorderings "
            "whose most effective region is not the dataset-order one; and the share "
            "whose trend changed sign.",
            "",
        ]
        lines += self._table(
            [
                "question",
                "pipeline",
                "reorderings answered",
                "mean sd of adjusted P(effect)",
                "widest range",
                "argmax moved",
                "trend sign flipped",
            ],
            [
                [
                    question.case.name,
                    question.pipeline_name,
                    f"{len(question.answered)} of {len(question.reordered)}",
                    self._number(question.mean_adjusted_standard_deviation, 3),
                    self._number(question.largest_adjusted_difference, 2),
                    self._share(question.argmax_flip_share),
                    self._share(question.trend_sign_flip_share),
                ]
                for question in permutations.questions
            ],
        )
        lines += [
            "",
            f"The whole-{noun} likelihood of the same held-out {plural} with the parts "
            "in the order the dataset lists them, and over the reorderings. "
            + (self.text.reordering_note + " " if self.text.reordering_note else "")
            + "*Largest drop* is how far below the dataset-order likelihood the worst "
            "reordering took each pipeline.",
            "",
        ]
        lines += self._table(
            [
                "pipeline",
                "dataset order, coverage / mean log-likelihood",
                "reorderings, coverage / mean log-likelihood (mean ± sd)",
                "largest drop",
            ],
            [
                [
                    name,
                    self._likelihood_cell(permutations.in_dataset_order(name)),
                    f"{self._mean_and_spread([report.coverage for report in reordered], 3)}"
                    f" / {self._mean_and_spread([report.mean_log_likelihood for report in reordered])}",
                    self._number(permutations.largest_likelihood_drop(name), 2),
                ]
                for name in permutations.whole_example_likelihoods
                for reordered in [permutations.reordered(name)]
            ],
        )
        return lines

    def _share(self, share: float) -> str:
        """
        :param share: A share between 0 and 1, or ``nan``.
        :return: The share as a percentage, or a dash.
        """
        if math.isnan(share):
            return "-"
        return self._percent(share)

    def _likelihood_cell(self, likelihood: LikelihoodReport) -> str:
        """
        :param likelihood: A likelihood report.
        :return: Its coverage and mean log-likelihood in one cell.
        """
        return (
            f"{self._percent(likelihood.coverage)} / "
            f"{self._number(likelihood.mean_log_likelihood, 2)}"
        )

    def _splits(self) -> List[str]:
        splits = self.splits
        lines = [
            "## Over several splits",
            "",
            f"The comparison repeated over {len(splits.reports)} random splits (seeds "
            f"{', '.join(str(report.random_seed) for report in splits.reports)}), "
            "mean ± standard deviation. The likelihoods are over the "
            f"{self.domain.plural} every pipeline modelling the view covers.",
            "",
        ]
        for view in ExampleView:
            names = [
                name
                for name in splits.pipeline_names
                if splits.reports[0].pipeline(name).likelihoods[view] is not None
            ]
            lines += [f"### {self.domain.view_label(view)}", ""]
            lines += self._table(
                [
                    "pipeline",
                    "held-out coverage",
                    "mean log-likelihood (covered by all)",
                ],
                [
                    [
                        name,
                        self._mean_and_spread(splits.coverage(name, view), 3),
                        self._mean_and_spread(splits.shared_log_likelihood(name, view)),
                    ]
                    for name in names
                ],
            )
            lines.append("")
        lines += [
            "Per question, how many splits each pipeline answered, and the mean ± "
            "standard deviation over the splits of its trend and of its contrast:",
            "",
        ]
        header = ["question"]
        for name in splits.pipeline_names:
            header += [f"{name}, answered", f"{name}, trend", f"{name}, contrast"]
        rows = []
        for case_index, outcome in enumerate(splits.reports[0].pipelines[0].outcomes):
            row = [outcome.case.name]
            for name in splits.pipeline_names:
                outcomes = splits.outcomes(name, case_index)
                answered = [one for one in outcomes if one.answered]
                row += [
                    f"{len(answered)} of {len(outcomes)}",
                    self._mean_and_spread(
                        [
                            float("nan") if one.trend is None else one.trend
                            for one in answered
                        ]
                    ),
                    self._mean_and_spread(
                        [
                            (
                                float("nan")
                                if one.contrast is None
                                else one.contrast.difference
                            )
                            for one in answered
                        ]
                    ),
                ]
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _learning_curve(self) -> List[str]:
        curve = self.curve
        seeds = sorted({point.random_seed for point in curve.points})
        lines = [
            "## How much training data it takes",
            "",
            f"Every pipeline's plain model fitted on a growing share of the "
            f"{self.domain.plural} and scored on the same held-out fifth, over "
            f"{len(seeds)} splits, mean ± standard deviation of the held-out coverage "
            f"and of the mean log-likelihood over the covered {self.domain.plural}. "
            "The relational circuit's templates pool every part of every training "
            f"{self.domain.noun}, where the unrolled tree sees one row per "
            f"{self.domain.noun}." + self._note(self.text.learning_curve_note),
        ]
        for view in (ExampleView.SCALARS_AND_COUNTS, ExampleView.WHOLE):
            names = [
                name
                for name in curve.pipeline_names
                if curve.points_of(name, curve.train_fractions[0])[0].likelihoods[view]
                is not None
            ]
            lines += ["", f"### {self.domain.view_label(view)}", ""]
            header = ["training share"]
            for name in names:
                header += [f"{name}, coverage", f"{name}, mean log-likelihood"]
            rows = []
            for train_fraction in curve.train_fractions:
                row = [self._percent(train_fraction)]
                for name in names:
                    likelihoods = [
                        point.likelihoods[view]
                        for point in curve.points_of(name, train_fraction)
                    ]
                    row += [
                        self._mean_and_spread(
                            [likelihood.coverage for likelihood in likelihoods], 3
                        ),
                        self._mean_and_spread(
                            [
                                likelihood.mean_log_likelihood
                                for likelihood in likelihoods
                            ]
                        ),
                    ]
                rows.append(row)
            lines += self._table(header, rows)
        return lines

    def _ground_truth(self) -> List[str]:
        truth = self.truth
        lines = [
            "## Error against known truth",
            "",
            f"{self.domain.plural.capitalize()} sampled from a structural causal model "
            "over the same domain, whose interventional probabilities are known by "
            "construction."
            + self._note(self.text.ground_truth_note)
            + f" Every pipeline was fitted on {truth.example_count} "
            f"{self.domain.plural} per setting and asked the questions; *mean* and "
            "*max absolute error* are over every supported cause region of every "
            "answered question, the *support-weighted* error weighs each region by "
            "the training rows it holds, *worst ordering* is the mean absolute error "
            "under the reordering of the parts the pipeline did worst on, and *rank "
            "correlation* is Spearman's between the answered and the true "
            "probabilities over a question's regions.",
            "",
        ]
        lines += self._table(
            [
                "pipeline",
                "questions answered",
                "mean abs. error",
                "support-weighted abs. error",
                "max abs. error",
                "mean abs. error, worst ordering",
                "rank correlation with truth",
            ],
            [
                [
                    name,
                    self._percent(truth.answered_share(name)),
                    self._number(truth.mean_absolute_error(truth.of(name, ordering=0))),
                    self._number(
                        truth.weighted_absolute_error(truth.of(name, ordering=0))
                    ),
                    self._number(truth.max_absolute_error(truth.of(name, ordering=0))),
                    self._number(truth.worst_ordering_mean_absolute_error(name)),
                    self._number(
                        truth.mean_rank_correlation(truth.of(name, ordering=0)), 2
                    ),
                ]
                for name in truth.pipeline_names
            ],
        )
        lines += ["", "Mean absolute error per setting of the model:", ""]
        setting_names = list(truth.descriptions[truth.configurations[0]])
        lines += self._table(
            setting_names + truth.pipeline_names,
            [
                list(truth.descriptions[configuration].values())
                + [
                    self._number(
                        truth.mean_absolute_error(
                            truth.of(name, configuration=configuration, ordering=0)
                        )
                    )
                    for name in truth.pipeline_names
                ]
                for configuration in truth.configurations
            ],
        )
        lines += ["", "Mean absolute error per question, over every setting:", ""]
        case_names = list(
            dict.fromkeys(outcome.case.name for _, outcome in truth.outcomes)
        )
        lines += self._table(
            ["question"] + truth.pipeline_names,
            [
                [case_name]
                + [
                    self._number(
                        truth.mean_absolute_error(
                            [
                                outcome
                                for outcome in truth.of(name, ordering=0)
                                if outcome.case.name == case_name
                            ]
                        )
                    )
                    for name in truth.pipeline_names
                ]
                for case_name in case_names
            ],
        )
        return lines

    def _monte_carlo(self) -> List[str]:
        monte_carlo = self.monte_carlo
        lines = [
            "## How many grounding samples it takes",
            "",
            "Inference on a grounded circuit is exact; grounding itself draws "
            "Monte-Carlo samples for every count the query leaves open and mixes one "
            "copy of the part templates per sampled value, so marginalising the open "
            "counts is a consistent estimate, not an exact sum. The relational "
            "circuit was fitted once and asked the same two questions with grounding "
            "drawing more and more samples; *deviation* is the largest difference, "
            "over the cause regions, from the answer at "
            f"{monte_carlo.reference_sample_count:,} samples, and *settled from* is "
            "the smallest number of samples from which every larger one stays within "
            f"{monte_carlo.stability_tolerance} of it.",
            "",
        ]
        for case_name, scored in monte_carlo.outcomes.items():
            settled = monte_carlo.settled_from(case_name)
            lines += [
                f"### {case_name}",
                "",
                "Settled from "
                + (
                    f"{settled:,} samples."
                    if settled is not None
                    else "the reference alone."
                ),
                "",
            ]
            lines += self._table(
                ["samples", "answered", "deviation from reference", "seconds"],
                [
                    [
                        f"{one.sample_count:,}",
                        Verdict.ANSWERED if one.outcome.answered else Verdict.REFUSED,
                        self._number(
                            monte_carlo.deviation(case_name, one.sample_count)
                        ),
                        self._number(one.outcome.duration, 1),
                    ]
                    for one in scored
                ],
            )
            lines.append("")
        return lines

    def _scaling(self) -> List[str]:
        scaling = self.scaling
        lines = [
            "## Cost against the number of objects",
            "",
            "The pipelines that model the parts, fitted on synthetic "
            f"{self.domain.plural} of growing size ({scaling.example_count} "
            f"{self.domain.plural} each) and asked one question about a part. The "
            "relational circuit's part templates pool every part of every "
            f"{self.domain.noun} into one circuit, so their size follows the number of "
            "distinct part attributes, not the number of parts; the unrolled table "
            "carries one block of columns per position, so its tree grows with the "
            f"widest {self.domain.noun}. *First ask* includes fitting the "
            "cause-specific model, *asked again* is grounding and adjustment alone."
            + self._note(self.text.scaling_note),
            "",
        ]
        header = [f"parts per {self.domain.noun}"]
        for name in scaling.pipeline_names:
            header += [
                f"{name}, fit seconds",
                f"{name}, nodes",
                f"{name}, first ask",
                f"{name}, asked again",
            ]
        rows = []
        for size in scaling.sizes:
            row = [str(size)]
            for name in scaling.pipeline_names:
                point = scaling.point(name, size)
                row += [
                    self._number(point.fit.training_duration, 1),
                    str(point.fit.size.node_count),
                    self._number(point.query_duration, 1),
                    self._number(point.repeat_duration, 1),
                ]
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _findings(self) -> List[str]:
        """
        What the numbers add up to, read off the report itself.
        """
        lines = ["## What the results show", ""]
        for pipeline in self.report.pipelines:
            answered = [outcome for outcome in pipeline.outcomes if outcome.answered]
            refused = [outcome for outcome in pipeline.outcomes if not outcome.answered]
            sentence = (
                f"- The {pipeline.name} answered {len(answered)} of "
                f"{len(pipeline.outcomes)} questions"
            )
            if refused:
                sentence += ", refusing " + "; ".join(
                    f"`{outcome.case.name}` because {outcome.refusal}"
                    for outcome in refused
                )
            lines.append(sentence + ".")
        lines += self._agreement_findings()
        lines += self._likelihood_findings()
        if self.permutations is not None:
            lines += self._permutation_findings()
        lines += self._latency_findings()
        return lines

    def _agreement_findings(self) -> List[str]:
        """
        Where several pipelines answered the same question, whether they agree.
        """
        lines = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            answers = [
                pipeline.outcomes[case_index]
                for pipeline in self.report.pipelines
                if pipeline.outcomes[case_index].most_effective is not None
            ]
            if len(answers) < 2:
                continue
            best_regions = {answer.most_effective.cause_region for answer in answers}
            probabilities = ", ".join(
                f"{answer.pipeline_name} "
                f"{self._number(answer.most_effective.adjusted_probability, 2)}"
                for answer in answers
            )
            if len(best_regions) == 1:
                lines.append(
                    f"- On `{outcome.case.name}`, every pipeline that answered finds "
                    f"{outcome.case.describe_cause(best_regions.pop())} the most "
                    f"effective setting (adjusted probabilities: {probabilities})."
                )
                continue
            by_region: Dict[str, List[QueryOutcome]] = {}
            for answer in answers:
                by_region.setdefault(answer.most_effective.cause_region, []).append(
                    answer
                )
            lines.append(
                f"- On `{outcome.case.name}`, the pipelines disagree on the most "
                "effective setting: "
                + "; ".join(
                    self._names(agreeing)
                    + (" say " if len(agreeing) > 1 else " says ")
                    + outcome.case.describe_cause(region)
                    + " ("
                    + ", ".join(
                        self._number(answer.most_effective.adjusted_probability, 2)
                        for answer in agreeing
                    )
                    + ")"
                    for region, agreeing in by_region.items()
                )
                + "."
            )
        return lines

    @staticmethod
    def _names(outcomes: Sequence[QueryOutcome]) -> str:
        """
        :param outcomes: Outcomes of different pipelines.
        :return: The pipelines' names as one phrase, such as ``the relational circuit
            and the unrolled tree``.
        """
        names = [f"the {outcome.pipeline_name}" for outcome in outcomes]
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + " and " + names[-1]

    def _likelihood_findings(self) -> List[str]:
        """
        Which pipeline explains the held-out examples best, per view.
        """
        lines = []
        for view in ExampleView:
            shared = self.report.shared_coverage_log_likelihoods[view]
            if len(shared) < 2:
                continue
            finite = {
                name: value for name, value in shared.items() if not math.isnan(value)
            }
            if not finite:
                continue
            best = max(finite, key=finite.get)
            if len(set(finite.values())) == 1:
                lines.append(
                    f"- On the {self.domain.view_label(view)}, every pipeline modelling "
                    f"it assigns the same mean log-likelihood "
                    f"({self._number(finite[best], 2)}) to the held-out "
                    f"{self.domain.plural}: on those columns they are the same tree "
                    "fitted on the same rows."
                )
                continue
            others = ", ".join(
                f"{name} {self._number(value, 2)}"
                for name, value in finite.items()
                if name != best
            )
            coverage = ", ".join(
                f"{pipeline.name} {self._percent(pipeline.likelihoods[view].coverage)}"
                for pipeline in self.report.pipelines
                if pipeline.likelihoods[view] is not None
            )
            lines.append(
                f"- On the {self.domain.view_label(view)}, the {best} assigns the "
                f"highest mean log-likelihood ({self._number(finite[best], 2)}, against "
                f"{others}) to the held-out {self.domain.plural} every pipeline covers; "
                f"coverage: {coverage}."
            )
        return lines

    def _permutation_findings(self) -> List[str]:
        """
        Which pipelines' answers moved when the parts were reordered.
        """
        lines = []
        by_pipeline = {}
        for question in self.permutations.questions:
            by_pipeline.setdefault(question.pipeline_name, []).append(question)
        for name, questions in by_pipeline.items():
            ranges = [
                question.largest_adjusted_difference
                for question in questions
                if not math.isnan(question.largest_adjusted_difference)
            ]
            flips = [
                question.argmax_flip_share
                for question in questions
                if not math.isnan(question.argmax_flip_share)
            ]
            sentence = (
                f"- Over {self.permutations.ordering_count} reorderings, the {name}'s "
                "adjusted effect probabilities ranged by up to "
                f"{self._number(max(ranges), 2) if ranges else '-'} and its most "
                "effective region moved in "
                f"{self._share(float(np.mean(flips))) if flips else '-'} of the "
                f"reorderings; its whole-{self.domain.noun} mean log-likelihood fell "
                "by up to "
                f"{self._number(self.permutations.largest_likelihood_drop(name), 2)}"
                " from the dataset's own order"
            )
            lines.append(sentence + ".")
        return lines

    def _latency_findings(self) -> List[str]:
        """
        How the pipelines compare on time per answered question.
        """
        lines = []
        for pipeline in self.report.pipelines:
            answered = [outcome for outcome in pipeline.outcomes if outcome.answered]
            if not answered:
                continue
            mean_repeat = sum(outcome.repeat_duration for outcome in answered) / len(
                answered
            )
            if math.isnan(mean_repeat):
                continue
            lines.append(
                f"- The {pipeline.name} takes {self._number(mean_repeat, 2)} seconds per "
                "answered question on average once its models are fitted."
            )
        return lines

    def _effects(self, case_index: int) -> List[str]:
        case = self._first_pipeline.outcomes[case_index].case
        lines = [
            f"## {case.question}",
            "",
            "One row per region of the cause the model distinguishes. *n* is how many "
            "training rows the region holds, and † marks a region below the support "
            "threshold; *P(region)* is how much of the fitted population it holds; "
            "*naive P(effect)* is the effect's probability simply conditioned on the "
            "region; *adjusted* is the interventional probability after summing out "
            "the question's confounders, which is what the question asks for, and the "
            "*interval* is its Wilson interval over the region's n. Where naive and "
            "adjusted agree, the confounders carried no further information within "
            "that region.",
            "",
        ]
        for pipeline in self.report.pipelines:
            outcome = pipeline.outcomes[case_index]
            lines.append(f"### {pipeline.name}")
            lines.append("")
            if not outcome.answered:
                lines.append(f"Refused: {outcome.refusal}.")
                lines.append("")
                continue
            lines += self._table(
                [
                    "cause region",
                    "n",
                    "P(region)",
                    "naive P(effect)",
                    "adjusted P(effect | do(cause))",
                    "95% interval",
                ],
                [
                    [
                        self._region_label(effect, outcome.min_region_support),
                        str(effect.support_count),
                        self._number(effect.region_probability),
                        self._number(effect.naive_probability),
                        self._number(effect.adjusted_probability),
                        f"[{self._number(effect.adjusted_interval.lower, 2)}, "
                        f"{self._number(effect.adjusted_interval.upper, 2)}]",
                    ]
                    for effect in sorted(
                        outcome.effects,
                        key=lambda effect: self._region_order(effect.cause_region),
                    )
                ],
            )
            lines += [
                "",
                f"EQL's own `cause` search settles on {outcome.best_region}: the region "
                "most probable once the effect is required to hold, from which the "
                "query's samples are drawn (P(effect | do) = "
                f"{self._number(outcome.effect_probability_given_best_region, 2)}).",
                "",
            ]
        return lines
