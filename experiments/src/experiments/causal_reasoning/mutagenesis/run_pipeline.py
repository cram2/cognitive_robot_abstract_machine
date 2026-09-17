"""
Fit every pipeline on the CTU Mutagenesis molecules, ask them every question of the
catalogue, repeat that over random atom orderings and random splits, measure how the
likelihoods grow with the training set, and write it all out as Markdown.

Run with::

    python -m experiments.causal_reasoning.mutagenesis.run_pipeline
        [--output RESULTS.md] [--train-fraction F] [--seed N]
        [--min-samples-per-leaf N] [--plain-min-samples-per-leaf N]
        [--orderings N] [--splits N]

The molecules are fetched from the CTU relational-dataset repository. The data access
objects the relational pipeline fits on come from the ``experiments`` package's
generated ORM interface; build it with ``scripts/regenerate_all_orm.py`` first.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Optional

from experiments.causal_reasoning.mutagenesis.dataset import (
    MutagenesisDataset,
    fetch_mutagenesis_molecules,
)
from experiments.causal_reasoning.mutagenesis.evaluation import (
    evaluate,
    learning_curve,
    permutation_study,
    split_study,
)
from experiments.causal_reasoning.mutagenesis.report import MarkdownReport


@dataclass(frozen=True)
class ExperimentFiles:
    """
    Where this experiment writes its comparison.
    """

    package_directory: Path = Path(__file__).parent
    """
    Where this experiment lives.
    """

    @property
    def results(self) -> Path:
        """
        Where the comparison is written.
        """
        return self.package_directory / "results.md"


logger = logging.getLogger(__name__)


def main(
    output: Path,
    train_fraction: float,
    seed: int,
    min_samples_per_leaf: Optional[int],
    plain_min_samples_per_leaf: Optional[int],
    ordering_count: int,
    split_count: int,
) -> None:
    """
    Run the comparison and every study around it, and write them out.

    :param output: The Markdown file to write.
    :param train_fraction: Share of molecules to fit on.
    :param seed: Seed of the split and of the questions' Monte-Carlo grounding; the
        repeated splits use the seeds counting up from it.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        tree may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain tree
        may hold; the pipelines' own default if not given.
    :param ordering_count: How many random atom orderings to try.
    :param split_count: How many random splits to repeat the comparison over.
    """
    import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes

    dataset = MutagenesisDataset(fetch_mutagenesis_molecules())
    settings = dict(
        train_fraction=train_fraction,
        min_samples_per_leaf=min_samples_per_leaf,
        plain_min_samples_per_leaf=plain_min_samples_per_leaf,
    )
    logger.info("Comparing on one split")
    report = evaluate(dataset, random_seed=seed, **settings)
    logger.info("Reordering the atoms %d times", ordering_count)
    permutations = permutation_study(
        dataset, ordering_count=ordering_count, random_seed=seed, **settings
    )
    logger.info("Repeating over %d splits", split_count)
    splits = split_study(
        dataset, random_seeds=range(seed, seed + split_count), **settings
    )
    logger.info("Measuring the learning curve")
    curve = learning_curve(
        dataset,
        random_seeds=range(seed, seed + min(split_count, 3)),
        plain_min_samples_per_leaf=plain_min_samples_per_leaf,
    )
    output.write_text(
        MarkdownReport(
            report, permutations=permutations, splits=splits, curve=curve
        ).render()
    )
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ExperimentFiles().results)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-samples-per-leaf", type=int, default=None)
    parser.add_argument("--plain-min-samples-per-leaf", type=int, default=None)
    parser.add_argument("--orderings", type=int, default=3)
    parser.add_argument("--splits", type=int, default=5)
    arguments = parser.parse_args()
    main(
        arguments.output,
        arguments.train_fraction,
        arguments.seed,
        arguments.min_samples_per_leaf,
        arguments.plain_min_samples_per_leaf,
        arguments.orderings,
        arguments.splits,
    )
