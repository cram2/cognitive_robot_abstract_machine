"""
Fit both pipelines on recorded attempts, ask them every question of the catalogue, and
write the comparison out as Markdown.

Run with::

    python -m experiments.causal_reasoning.tracy_clutter_picking.run_pipeline
        [--dataset ATTEMPTS.json] [--output RESULTS.md] [--train-fraction F]
        [--seed N] [--min-samples-per-leaf N]

The data access objects the relational pipeline fits on come from the ``experiments``
package's generated ORM interface; build it with ``scripts/regenerate_all_orm.py``
first.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Optional

from experiments.causal_reasoning.tracy_clutter_picking.dataset import (
    ClutterPickDataset,
    HostedDataset,
)
from experiments.causal_reasoning.tracy_clutter_picking.evaluation import evaluate
from experiments.causal_reasoning.tracy_clutter_picking.report import MarkdownReport


@dataclass(frozen=True)
class ExperimentFiles:
    """
    Where this experiment reads its recorded attempts from and writes its comparison.
    """

    package_directory: Path = Path(__file__).parent
    """
    Where this experiment lives.
    """

    recorded_attempts: HostedDataset = HostedDataset(
        "https://raw.githubusercontent.com/Narenvasant/tracy_clutter_picking_data/v2/milk_clutter_attempts.json"
    )
    """
    The attempts recorded with
    :mod:`~experiments.causal_reasoning.tracy_clutter_picking.collect_data`, hosted in
    their own repository and fetched on first use.
    """

    @property
    def results(self) -> Path:
        """
        Where the comparison is written.
        """
        return self.package_directory / "results.md"


logger = logging.getLogger(__name__)


def main(
    dataset: Optional[Path],
    output: Path,
    train_fraction: float,
    seed: int,
    min_samples_per_leaf: Optional[int],
    plain_min_samples_per_leaf: Optional[int],
) -> None:
    """
    Run the comparison and write it out.

    :param dataset: A local file of recorded attempts to read; the hosted ones if not
        given.
    :param output: The Markdown file to write.
    :param train_fraction: Share of attempts to fit on.
    :param seed: Seed of the split and of the questions' Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        tree may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain tree
        may hold; the pipelines' own default if not given.
    """
    import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes

    files = ExperimentFiles()
    attempts = (
        files.recorded_attempts.load()
        if dataset is None
        else ClutterPickDataset.load(dataset)
    )
    report = evaluate(
        attempts,
        train_fraction=train_fraction,
        random_seed=seed,
        min_samples_per_leaf=min_samples_per_leaf,
        plain_min_samples_per_leaf=plain_min_samples_per_leaf,
    )
    output.write_text(MarkdownReport(report).render())
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    files = ExperimentFiles()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=files.results)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-samples-per-leaf", type=int, default=None)
    parser.add_argument("--plain-min-samples-per-leaf", type=int, default=None)
    arguments = parser.parse_args()
    main(
        arguments.dataset,
        arguments.output,
        arguments.train_fraction,
        arguments.seed,
        arguments.min_samples_per_leaf,
        arguments.plain_min_samples_per_leaf,
    )
