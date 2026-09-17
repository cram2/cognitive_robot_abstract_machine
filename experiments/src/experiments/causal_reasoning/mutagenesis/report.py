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
from typing_extensions import Iterable, List, Optional, Sequence, Tuple

from experiments.causal_reasoning.mutagenesis.evaluation import (
    EvaluationReport,
    LearningCurveReport,
    PermutationReport,
    PipelineReport,
    QueryOutcome,
    SplitReport,
)
from experiments.causal_reasoning.mutagenesis.flat_table import MoleculeView


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
    The comparison on one split, with every question asked and timed.
    """

    permutations: Optional[PermutationReport] = None
    """
    The same questions under random orderings of the atoms and bonds, if run.
    """

    splits: Optional[SplitReport] = None
    """
    The comparison repeated over several splits, if run.
    """

    curve: Optional[LearningCurveReport] = None
    """
    The likelihoods over growing training sets, if run.
    """

    def render(self) -> str:
        """
        :return: The whole document.
        """
        sections = [
            self._setup(),
            self._mutagenicity(),
            self._answerability(),
            self._quantities(),
            self._latencies(),
        ]
        if self.permutations is not None:
            sections.append(self._permutations())
        if self.splits is not None:
            sections.append(self._splits())
        if self.curve is not None:
            sections.append(self._learning_curve())
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
    def _region_order(region: str) -> Tuple[int, float, str]:
        """
        :param region: A cause region, written out.
        :return: A sort key putting numeric regions in numeric order ahead of the
            others in alphabetical order.
        """
        if re.fullmatch(r"-?\d+(\.\d+)?", region):
            return (0, float(region), region)
        return (1, 0.0, region)

    # %% sections

    def _setup(self) -> List[str]:
        report = self.report
        return [
            "# Mutagenesis: relational circuit against flat-table trees",
            "",
            "The CTU Mutagenesis dataset records 188 nitroaromatic molecules, each as "
            "its own attributes (the `ind1` structural indicator, `logp`, `lumo`, and "
            "whether it tested mutagenic) with one exchangeable part per atom "
            "(element, atom-type code, partial charge, number of bonds) and one per "
            "bond (its type). A molecule has between 14 and 40 atoms, and its atoms "
            "have no canonical order; the dataset lists heavy atoms first and "
            "hydrogens last, but nothing ties position to identity.",
            "",
            "Four pipelines were fitted on the same molecules and asked the same "
            "`cause`/`causes_effect` EQL queries:",
            "",
            "- **relational circuit**: a relational probabilistic circuit fitted on the "
            "molecules' relational structure, one circuit over the molecule's own "
            "attributes and its aggregation counts (chlorine atoms, branching atoms, "
            "double bonds, aromatic bonds), one template over an atom's attributes "
            "and one over a bond's, grounded per query into a circuit over exactly "
            "the queried molecule, atoms and bonds and registered as a causal "
            "circuit;",
            "- **propositional tree**: a joint probability tree fitted on the "
            "molecules flattened into one table of the molecule's own attributes and "
            "the same four counts, the classic propositional summary of a relational "
            "example, registered as a causal circuit the same way;",
            "- **unrolled tree**: the same tree on a table that also carries every "
            "atom's and bond's attributes under the part's position, padded with an "
            "absent marker past a molecule's last part, so that a column means "
            "whatever part a molecule happens to list at that position;",
            "- **scalars-only tree**: the same tree on the molecule's own attributes "
            "alone, what a flat learner sees without the relational feature "
            "extraction.",
            "",
            "Every flat tree answers a query by backdoor adjustment on a table column; "
            "the relational circuit does the same on the variable of a grounded "
            "circuit. In both, the model is stratified so it is support-deterministic "
            "over the cause, the effect's probability is read off every region of the "
            "cause, and any variable the query marks as a confounder is summed out of "
            "that reading. Every query lists one atom and one bond with all their "
            "attributes open, which is what makes grounding retain the molecule's "
            "counts as variables; a flat table ignores parts a query says nothing "
            "about and refuses a query that constrains a column it does not have.",
            "",
            "## Setup",
            "",
            f"- molecules: {report.training_molecule_count + report.test_molecule_count}"
            f" ({report.training_molecule_count} to fit on, "
            f"{report.test_molecule_count} held out)",
            f"- molecules that tested mutagenic: {self._percent(report.mutagenic_rate)}",
            f"- fewest training rows per leaf: {report.min_samples_per_leaf} in a "
            f"cause-specific model, {report.plain_min_samples_per_leaf} in the plain "
            "model that scores held-out molecules",
            f"- split seed: {report.random_seed}",
        ]

    def _mutagenicity(self) -> List[str]:
        lines = [
            "## How often a molecule is mutagenic",
            "",
            "The molecules themselves, before any model: the share that tested "
            "mutagenic, grouped by the `ind1` indicator, by how many branching atoms "
            "(atoms with three or four bonds, the ring-fusion and branch points of the "
            "molecular graph) the molecule has, and by how many of its bonds are "
            "aromatic. This is the signal the models are asked to explain.",
            "",
        ]
        for title, rates in (
            ("ind1", self.report.mutagenic_by_indicator),
            ("branching atoms", self.report.mutagenic_by_branching_atom_count),
            ("aromatic bonds", self.report.mutagenic_by_aromatic_bond_count),
        ):
            lines += self._table(
                [title, "molecules", "mutagenic"],
                [
                    [str(value), str(rate.molecule_count), self._percent(rate.rate)]
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
            "A question about counts needs the counts: the scalars-only tree refuses "
            "it. A question whose effect is one atom's own attribute needs the atoms: "
            "the propositional tree refuses it, the unrolled tree answers it about "
            "whatever atom the molecules list at that position, and the relational "
            "circuit answers it about an exchangeable atom. A question whose cause is "
            "one atom's own attribute is refused by the relational circuit: grounding "
            "with the molecule's counts left open mixes one copy of the atom template "
            "per sampled count, and those copies overlap on the atom's element "
            "without being identical (a copy for a molecule with no chlorine has no "
            "chlorine atom, a copy for one with some has), so the grounded circuit is "
            "not support-deterministic over the element. The unrolled tree answers "
            "it, since a column is a column; what that answer is worth is what the "
            "reordering below measures.",
        ]
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
            "What each pipeline cost. *Models fitted* counts the plain model plus one "
            "support-deterministic model per distinct cause the questions asked about "
            "and the pipeline could fit; *training seconds* and the *nodes*/*edges* "
            "of every fitted circuit are summed over them, which for the relational "
            "circuit includes the atom and bond templates.",
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
            "How well each explains molecules it never saw, on three views of a "
            "molecule: its own scalars, which every pipeline models; its scalars and "
            "counts; and the whole molecule, atoms and bonds included, which only the "
            "pipelines that model the parts can score. The relational circuit scores "
            "a whole molecule as its class circuit over the scalars and counts times "
            "each part template over one atom or bond given the counts; the unrolled "
            "tree scores it as one row. *Held-out coverage* is the share of held-out "
            "molecules that lie inside the plain model's support at all, since a "
            "tree's leaves span only the value ranges they were fitted on, and a "
            "whole molecule is covered only if every one of its atoms and bonds is. "
            "The *mean log-likelihood* is over the covered molecules only; the last "
            "column restricts it to the molecules every pipeline in the table covers, "
            "so the numbers are over the same rows.",
        ]
        for view in MoleculeView:
            scored = [
                pipeline
                for pipeline in self.report.pipelines
                if pipeline.likelihoods[view] is not None
            ]
            lines += ["", f"### {view}", ""]
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
            "leaves open and grounds one atom template per sampled value, which is "
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
        lines = [
            "## Does the order of the atoms matter?",
            "",
            f"Every molecule's atoms and bonds were put in a random order, "
            f"{permutations.ordering_count} times over, and each time the pipelines "
            "that model the parts were refitted on the same split and asked the "
            "questions about atoms again. A relational circuit treats the atoms as "
            "exchangeable, so nothing about it can depend on the order; an unrolled "
            "table's column `atoms[0]` holds a different atom of every molecule after "
            "each reordering. *Best regions* lists every most effective cause region "
            "found over the orderings; *largest difference* is, over the cause "
            "regions every answered ordering distinguishes, the widest gap between "
            "orderings in the effect's adjusted probability.",
            "",
        ]
        lines += self._table(
            [
                "question",
                "pipeline",
                "orderings answered",
                "best regions",
                "largest difference in adjusted P(effect)",
            ],
            [
                [
                    question.case.name,
                    question.pipeline_name,
                    f"{len(question.answered)} of {len(question.outcomes)}",
                    ", ".join(question.best_regions) or "-",
                    self._number(question.largest_adjusted_difference, 2),
                ]
                for question in permutations.questions
            ],
        )
        lines += [
            "",
            "The whole-molecule likelihood of the same held-out molecules under each "
            "ordering:",
            "",
        ]
        lines += self._table(
            ["pipeline"]
            + [
                f"ordering {ordering}, coverage / mean log-likelihood"
                for ordering in range(permutations.ordering_count)
            ]
            + ["largest difference"],
            [
                [name]
                + [
                    f"{self._percent(report.coverage)} / "
                    f"{self._number(report.mean_log_likelihood, 2)}"
                    for report in reports
                ]
                + [self._number(permutations.largest_likelihood_difference(name), 2)]
                for name, reports in permutations.whole_molecule_likelihoods.items()
            ],
        )
        return lines

    def _splits(self) -> List[str]:
        splits = self.splits
        lines = [
            "## Over several splits",
            "",
            f"The comparison repeated over {len(splits.reports)} random splits (seeds "
            f"{', '.join(str(report.random_seed) for report in splits.reports)}), "
            "mean ± standard deviation. The likelihoods are over the molecules every "
            "pipeline modelling the view covers.",
            "",
        ]
        for view in MoleculeView:
            names = [
                name
                for name in splits.pipeline_names
                if splits.reports[0].pipeline(name).likelihoods[view] is not None
            ]
            lines += [f"### {view}", ""]
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
            "Per question, how many splits each pipeline answered and which most "
            "effective cause regions it found across them:",
            "",
        ]
        header = ["question"] + splits.pipeline_names
        rows = []
        for case_index, outcome in enumerate(splits.reports[0].pipelines[0].outcomes):
            row = [outcome.case.name]
            for name in splits.pipeline_names:
                outcomes = splits.outcomes(name, case_index)
                answered = [one for one in outcomes if one.answered]
                regions = sorted({one.most_effective.cause_region for one in answered})
                row.append(
                    f"{len(answered)} of {len(outcomes)}"
                    + (f": {', '.join(regions)}" if regions else "")
                )
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _learning_curve(self) -> List[str]:
        curve = self.curve
        seeds = sorted({point.random_seed for point in curve.points})
        lines = [
            "## How much training data it takes",
            "",
            f"Every pipeline's plain model fitted on a growing share of the molecules "
            f"and scored on the same held-out fifth, over {len(seeds)} splits, mean ± "
            "standard deviation of the held-out coverage and of the mean "
            "log-likelihood over the covered molecules. The relational circuit's "
            "templates pool every atom of every training molecule, about 26 rows per "
            "molecule, where the unrolled tree sees one row per molecule.",
        ]
        for view in (MoleculeView.SCALARS_AND_COUNTS, MoleculeView.WHOLE_MOLECULE):
            names = [
                name
                for name in curve.pipeline_names
                if curve.points_of(name, curve.train_fractions[0])[0].likelihoods[view]
                is not None
            ]
            lines += ["", f"### {view}", ""]
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
                if pipeline.outcomes[case_index].answered
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
        Which pipeline explains the held-out molecules best, per view.
        """
        lines = []
        for view in MoleculeView:
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
                    f"- On the {view}, every pipeline modelling it assigns the same "
                    f"mean log-likelihood ({self._number(finite[best], 2)}) to the "
                    "held-out molecules: on those columns they are the same tree "
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
                f"- On the {view}, the {best} assigns the highest mean log-likelihood "
                f"({self._number(finite[best], 2)}, against {others}) to the "
                f"held-out molecules every pipeline covers; coverage: {coverage}."
            )
        return lines

    def _permutation_findings(self) -> List[str]:
        """
        Which pipelines' answers moved when the atoms were reordered.
        """
        lines = []
        by_pipeline = {}
        for question in self.permutations.questions:
            by_pipeline.setdefault(question.pipeline_name, []).append(question)
        for name, questions in by_pipeline.items():
            differences = [
                question.largest_adjusted_difference
                for question in questions
                if not math.isnan(question.largest_adjusted_difference)
            ]
            unstable = [
                question.case.name
                for question in questions
                if len(question.best_regions) > 1
            ]
            likelihood_difference = self.permutations.largest_likelihood_difference(
                name
            )
            sentence = (
                f"- Reordering the atoms moved the {name}'s adjusted effect "
                f"probabilities by up to {self._number(max(differences), 2) if differences else '-'}"
                f" and its whole-molecule mean log-likelihood by "
                f"{self._number(likelihood_difference, 2)}"
            )
            if unstable:
                sentence += "; it changed the most effective setting of " + ", ".join(
                    f"`{case_name}`" for case_name in unstable
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
            "how much of the training population that region holds; *naive P(effect)* "
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
