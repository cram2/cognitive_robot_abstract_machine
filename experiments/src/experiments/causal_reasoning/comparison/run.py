"""
Running one experiment end to end: the comparison on one split, then every study around
it, with the report written out after each so that a run stopped part-way leaves what
it finished on disk.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing_extensions import Any, Callable, List, Optional, Sequence

from experiments.causal_reasoning.comparison.dataset import ExampleDataset
from experiments.causal_reasoning.comparison.evaluation import (
    Comparison,
    KnownTruth,
    evaluate,
    ground_truth_study,
    learning_curve,
    monte_carlo_study,
    permutation_study,
    scaling_study,
    split_study,
)
from experiments.causal_reasoning.comparison.queries import CausalQueryCase
from experiments.causal_reasoning.comparison.report import MarkdownReport, ReportText

logger = logging.getLogger(__name__)


# %% what one experiment brings


@dataclass(frozen=True)
class ScalingSetup:
    """
    What the scaling study needs: examples of a chosen size and one question about a
    part to time on them.
    """

    examples_of_size: Callable[[int, int, np.random.Generator], List[Any]]
    """
    How to sample a number of examples with a typical number of parts.
    """

    case: CausalQueryCase
    """
    The part-level question to time.
    """

    sizes: Sequence[int] = (5, 10, 20, 50)
    """
    The typical numbers of parts per example to measure.
    """


@dataclass(frozen=True)
class Experiment:
    """
    One dataset's comparison: what is compared, the questions asked of it, and the
    optional studies it supports.
    """

    comparison: Comparison
    """
    The example, its positional parts and its effect summaries.
    """

    text: ReportText
    """
    What the report says about the dataset and the pipelines.
    """

    cases: Sequence[CausalQueryCase]
    """
    Every question of the catalogue, asked on the comparison's split.
    """

    part_cases: Sequence[CausalQueryCase] = ()
    """
    The questions about one part, asked again under every reordering of the parts.
    """

    monte_carlo_cases: Sequence[CausalQueryCase] = ()
    """
    The questions followed as grounding draws more samples.
    """

    truth: Optional[KnownTruth] = None
    """
    A model of the domain with known interventional probabilities, if there is one.
    """

    truth_cases: Sequence[CausalQueryCase] = ()
    """
    The questions scored against that model's truth.
    """

    scaling: Optional[ScalingSetup] = None
    """
    How to measure cost against the number of parts, if the experiment can sample
    examples of a chosen size.
    """

    results: Path = field(default_factory=Path)
    """
    Where the report is written by default.
    """


# %% run settings


@dataclass(frozen=True)
class RunSettings:
    """
    The knobs every experiment's run shares.
    """

    output: Path
    """
    The Markdown file to write.
    """

    train_fraction: float = 0.8
    """
    Share of examples to fit on.
    """

    seed: int = 0
    """
    Seed of the split and of the questions' Monte-Carlo grounding; the learning curve
    and any repeated splits use the seeds counting up from it.
    """

    min_samples_per_leaf: Optional[float] = None
    """
    The fewest training rows a leaf of a cause-specific model may hold; the pipelines'
    own default if not given.
    """

    plain_min_samples_per_leaf: Optional[float] = None
    """
    The fewest training rows a leaf of the plain model may hold; the pipelines' own
    default if not given.
    """

    ordering_count: int = 20
    """
    How many random orderings of the parts to try.
    """

    split_count: int = 0
    """
    How many random splits to repeat the comparison over; none skips the study.
    """

    learning_curve_splits: int = 3
    """
    How many splits the learning curve averages over.
    """

    min_region_support: int = 10
    """
    The fewest training examples a cause region may hold for its effect to be read as
    an answer.
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser, default_output: Path) -> None:
        """
        Add the shared command-line flags.

        :param parser: The experiment's parser.
        :param default_output: Where the report goes unless ``--output`` says otherwise.
        """
        parser.add_argument("--output", type=Path, default=default_output)
        parser.add_argument("--train-fraction", type=float, default=0.8)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--min-samples-per-leaf", type=float, default=None)
        parser.add_argument("--plain-min-samples-per-leaf", type=float, default=None)
        parser.add_argument("--orderings", type=int, default=20)
        parser.add_argument("--splits", type=int, default=0)
        parser.add_argument("--min-region-support", type=int, default=10)

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> RunSettings:
        """
        :param arguments: The parsed command line.
        :return: The settings it names.
        """
        return cls(
            output=arguments.output,
            train_fraction=arguments.train_fraction,
            seed=arguments.seed,
            min_samples_per_leaf=arguments.min_samples_per_leaf,
            plain_min_samples_per_leaf=arguments.plain_min_samples_per_leaf,
            ordering_count=arguments.orderings,
            split_count=arguments.splits,
            min_region_support=arguments.min_region_support,
        )


# %% running


@dataclass
class ReportWriter:
    """
    Writes the report out after every study, so that a run stopped part-way leaves what
    it has finished on disk and a reader can follow it as it goes.
    """

    output: Path
    """
    The Markdown file to write.
    """

    report: MarkdownReport
    """
    The report, which the studies fill in one by one.
    """

    def after(self, study: str) -> None:
        """
        Write the report as it stands.

        :param study: What was just finished, for the log.
        """
        self.output.write_text(self.report.render())
        logger.info("%s; wrote %s", study, self.output)


def run(experiment: Experiment, dataset: ExampleDataset, settings: RunSettings) -> None:
    """
    Run the comparison and every study the experiment supports, and write them out.

    :param experiment: What to compare and what to ask.
    :param dataset: The examples.
    :param settings: The run's knobs.
    """
    comparison = experiment.comparison
    leaf_settings = dict(
        min_samples_per_leaf=settings.min_samples_per_leaf,
        plain_min_samples_per_leaf=settings.plain_min_samples_per_leaf,
    )
    split_settings = dict(
        train_fraction=settings.train_fraction,
        min_region_support=settings.min_region_support,
        **leaf_settings,
    )
    rendered = MarkdownReport(
        domain=comparison.domain,
        text=experiment.text,
        report=evaluate(
            comparison,
            dataset,
            experiment.cases,
            random_seed=settings.seed,
            **split_settings,
        ),
    )
    write = ReportWriter(output=settings.output, report=rendered)
    write.after("Comparing on one split")
    if experiment.truth is not None:
        rendered.truth = ground_truth_study(
            comparison,
            experiment.truth,
            experiment.truth_cases,
            random_seed=settings.seed,
            min_region_support=settings.min_region_support,
            **leaf_settings,
        )
        write.after("Scoring against the synthetic model's truth")
    if experiment.part_cases:
        rendered.permutations = permutation_study(
            comparison,
            dataset,
            experiment.part_cases,
            ordering_count=settings.ordering_count,
            random_seed=settings.seed,
            **split_settings,
        )
        write.after(f"Reordering the parts {settings.ordering_count} times")
    if experiment.monte_carlo_cases:
        rendered.monte_carlo = monte_carlo_study(
            comparison,
            dataset,
            experiment.monte_carlo_cases,
            random_seed=settings.seed,
            **split_settings,
        )
        write.after("Following the answers as grounding draws more samples")
    rendered.curve = learning_curve(
        comparison,
        dataset,
        random_seeds=range(
            settings.seed, settings.seed + settings.learning_curve_splits
        ),
        plain_min_samples_per_leaf=settings.plain_min_samples_per_leaf,
    )
    write.after("Measuring the learning curve")
    if experiment.scaling is not None:
        rendered.scaling = scaling_study(
            comparison,
            experiment.scaling.examples_of_size,
            experiment.scaling.case,
            sizes=experiment.scaling.sizes,
            random_seed=settings.seed,
            **leaf_settings,
        )
        write.after("Measuring cost against the number of parts")
    if settings.split_count > 0:
        rendered.splits = split_study(
            comparison,
            dataset,
            experiment.cases,
            random_seeds=range(settings.seed, settings.seed + settings.split_count),
            **split_settings,
        )
        write.after(f"Repeating over {settings.split_count} splits")
