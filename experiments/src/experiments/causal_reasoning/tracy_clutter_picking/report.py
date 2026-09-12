"""
Writing an :class:`~experiments.causal_reasoning.tracy_clutter_picking.evaluation.EvaluationReport`
out as Markdown, with every table explained and the answers put into words.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Iterable, List, Optional, Sequence

from experiments.causal_reasoning.tracy_clutter_picking.evaluation import (
    EvaluationReport,
    InterventionalEffect,
    PipelineReport,
    QueryOutcome,
)


class Verdict(StrEnum):
    """
    How an outcome is marked in the answerability table.
    """

    ANSWERED = "answered"
    REFUSED = "refused"


@dataclass
class MarkdownReport:
    """
    Renders a comparison as Markdown.
    """

    report: EvaluationReport
    """
    The comparison to render.
    """

    def render(self) -> str:
        """
        :return: The whole document.
        """
        sections = [
            self._setup(),
            self._success(),
            self._answerability(),
            self._quantities(),
            self._latencies(),
            self._findings(),
        ]
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

    # %% sections

    def _setup(self) -> List[str]:
        report = self.report
        return [
            "# Tracy clutter picking: relational circuit against flat-table tree",
            "",
            "Tracy's left arm picks one milk carton out of a ten-carton clutter in "
            "MuJoCo, holding it by contact friction alone. Every attempt is recorded as "
            "a relational scene: the attempt's own attributes (environment, grasp "
            "friction, grasp yaw, whether the target came up) and one exchangeable "
            "part per neighbouring carton (its position relative to the target, its "
            "distance band, which side of the fingers' closing axis it stands on, and "
            "how far the pick shoved it).",
            "",
            "Two pipelines were fitted on the same recorded attempts and asked the same "
            "`cause`/`causes_effect` EQL queries:",
            "",
            "- **relational circuit**: a relational probabilistic circuit fitted on the "
            "scenes' relational structure -- one circuit over the attempt's own "
            "attributes and its aggregation statistics, one template over a "
            "neighbour's attributes -- grounded per query into a circuit over exactly "
            "the queried objects and registered as a causal circuit;",
            "- **flat-table tree**: a joint probability tree fitted on the same attempts "
            "flattened into one fixed-width table, one block of columns per neighbour "
            "index, registered as a causal circuit the same way.",
            "",
            "Both answer a query by backdoor adjustment: the model is stratified so it "
            "is support-deterministic over the cause, the effect's probability is read "
            "off every region of the cause, and any variable the query marks as a "
            "confounder is summed out of that reading.",
            "",
            "## Setup",
            "",
            f"- recorded attempts: {report.training_scene_count + report.test_scene_count}"
            f" ({report.training_scene_count} to fit on, {report.test_scene_count} held out)",
            f"- neighbours per attempt: {report.recorded_neighbour_count}",
            f"- attempts whose target was lifted: {self._percent(report.success_rate)}",
            f"- fewest training rows per leaf: {report.min_samples_per_leaf} in a "
            f"cause-specific model, {report.plain_min_samples_per_leaf} in the plain "
            "model that scores held-out attempts",
        ]

    def _success(self) -> List[str]:
        lines = [
            "## How often the pick came up",
            "",
            "The recorded attempts themselves, before any model: the share of attempts "
            "whose target was still held at the end, grouped by the environment the "
            "clutter stood in, by the grasp's friction coefficient, and by how many "
            "neighbours stood adjacent to the target (closer than the fingers' sweep). "
            "This is the picking efficiency in clutter the models are asked to explain.",
            "",
        ]
        for title, rates in (
            ("environment", self.report.success_by_environment),
            ("friction coefficient", self.report.success_by_friction),
            ("adjacent neighbours", self.report.success_by_crowding),
        ):
            lines += self._table(
                [title, "attempts", "lifted"],
                [
                    [str(value), str(rate.attempt_count), self._percent(rate.rate)]
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
            "adjustment and how likely, against the least favourable setting; a "
            "refused cell says why the pipeline could not answer at all.",
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
        lines += [
            "",
            "The three questions about the recorded clutter size can be put to either "
            "pipeline; the three about other clutter sizes have no columns in the flat "
            "table, so only a model that grounds itself for the queried objects can "
            "answer them.",
        ]
        return lines

    @staticmethod
    def _least_effective(outcome: QueryOutcome) -> Optional[InterventionalEffect]:
        """
        :param outcome: An answered outcome.
        :return: The cause region whose intervention gives the effect the lowest
            adjusted probability.
        """
        if not outcome.effects:
            return None
        return min(outcome.effects, key=lambda effect: effect.adjusted_probability)

    def _verdict(self, outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its answer put into words, or its refusal, in one cell.
        """
        if not outcome.answered:
            return f"{Verdict.REFUSED}: {outcome.refusal}."
        case = outcome.case
        best = outcome.most_effective
        worst = self._least_effective(outcome)
        return (
            f"{Verdict.ANSWERED}: with {case.describe_cause(best.cause_region)}, "
            f"{case.effect} with probability "
            f"{self._number(best.adjusted_probability, 2)}, the highest of any "
            f"setting; with {case.describe_cause(worst.cause_region)} it is only "
            f"{self._number(worst.adjusted_probability, 2)}."
        )

    def _quantities(self) -> List[str]:
        lines = [
            "## Fit and likelihood",
            "",
            "What each pipeline cost and how well it explains attempts it never saw. "
            "*Models fitted* counts the plain model plus one support-deterministic "
            "model per distinct cause the questions asked about; *training seconds* "
            "and the *nodes*/*edges* of every fitted circuit are summed over them. "
            "*Held-out coverage* is the share of held-out attempts that lie inside "
            "the plain model's support at all -- a tree's leaves span only the value "
            "ranges they were fitted on, so an attempt with any attribute outside "
            "every leaf's range has zero likelihood. The *mean log-likelihood* is "
            "over the covered attempts only, on an attempt's observed attributes (its "
            "own scalars and every neighbour's); the last column restricts it to the "
            "attempts both pipelines cover, so the two numbers are over the same "
            "rows.",
            "",
        ]
        rows = []
        for pipeline in self.report.pipelines:
            rows.append(
                [
                    pipeline.name,
                    str(pipeline.fit.model_count),
                    self._number(pipeline.fit.training_duration, 2),
                    str(pipeline.fit.size.node_count),
                    str(pipeline.fit.size.edge_count),
                    self._percent(pipeline.likelihood.coverage),
                    self._number(pipeline.likelihood.mean_log_likelihood, 2),
                    self._number(
                        self.report.shared_coverage_log_likelihoods[pipeline.name], 2
                    ),
                ]
            )
        lines += self._table(
            [
                "pipeline",
                "models fitted",
                "training seconds",
                "nodes",
                "edges",
                "held-out coverage",
                "mean log-likelihood (covered)",
                "mean log-likelihood (covered by both)",
            ],
            rows,
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
            "relational answer about a larger clutter grounds a larger circuit and "
            "takes longer.",
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
        lines += self._latency_findings()
        return lines

    def _agreement_findings(self) -> List[str]:
        """
        Where the two pipelines answered the same question, whether they agree.
        """
        lines = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            answers = [
                pipeline.outcomes[case_index]
                for pipeline in self.report.pipelines
                if pipeline.outcomes[case_index].answered
            ]
            if len(answers) < 2:
                continue
            best_regions = {answer.most_effective.cause_region for answer in answers}
            probabilities = ", ".join(
                self._number(answer.most_effective.adjusted_probability, 2)
                for answer in answers
            )
            if len(best_regions) == 1:
                lines.append(
                    f"- On `{outcome.case.name}`, both pipelines find "
                    f"{outcome.case.describe_cause(best_regions.pop())} the most "
                    f"effective setting (adjusted probabilities {probabilities})."
                )
                continue
            lines.append(
                f"- On `{outcome.case.name}`, the pipelines disagree on the most "
                "effective setting: "
                + "; ".join(
                    f"the {answer.pipeline_name} says "
                    f"{outcome.case.describe_cause(answer.most_effective.cause_region)} "
                    f"({self._number(answer.most_effective.adjusted_probability, 2)})"
                    for answer in answers
                )
                + "."
            )
        return lines

    def _likelihood_findings(self) -> List[str]:
        """
        Which pipeline covers more held-out attempts, and which explains the shared ones
        better.
        """
        by_coverage = max(
            self.report.pipelines, key=lambda pipeline: pipeline.likelihood.coverage
        )
        shared = self.report.shared_coverage_log_likelihoods
        by_likelihood = max(shared, key=shared.get)
        return [
            f"- The {by_coverage.name} covers the most held-out attempts "
            f"({self._percent(by_coverage.likelihood.coverage)}); on the attempts both "
            f"cover, the {by_likelihood} assigns the higher mean log-likelihood "
            f"({self._number(shared[by_likelihood], 2)}).",
        ]

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
            "One row per region of the cause the model distinguishes. *P(region)* is "
            "how much of the recorded population that region holds; *naive P(effect)* "
            "is the effect's probability simply conditioned on the region; *adjusted* "
            "is the interventional probability after summing out the question's "
            "confounders, which is what the question asks for. Where the two columns "
            "agree, the confounder carried no extra information within that region.",
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
                    "P(region)",
                    "naive P(effect)",
                    "adjusted P(effect | do(cause))",
                ],
                [
                    [
                        effect.cause_region,
                        self._number(effect.region_probability),
                        self._number(effect.naive_probability),
                        self._number(effect.adjusted_probability),
                    ]
                    for effect in sorted(
                        outcome.effects, key=lambda effect: effect.cause_region
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
