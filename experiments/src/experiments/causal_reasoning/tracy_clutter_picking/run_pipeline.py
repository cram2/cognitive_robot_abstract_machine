"""
Fit every pipeline on the recorded attempts, ask them every question of the catalogue,
repeat that over random orderings of the neighbours, follow the relational answers as
grounding draws more samples, measure how the likelihoods grow with the training set and
how the cost grows with the size of the clutter, and write it all out as Markdown.

Run with::

    python -m experiments.causal_reasoning.tracy_clutter_picking.run_pipeline
        [--dataset ATTEMPTS.json] [--output RESULTS.md] [--train-fraction F]
        [--seed N] [--min-samples-per-leaf F] [--plain-min-samples-per-leaf F]
        [--orderings N] [--splits N] [--min-region-support N]

The data access objects the relational pipeline fits on come from the ``experiments``
package's generated ORM interface; build it with ``scripts/regenerate_all_orm.py``
first.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing_extensions import List, Optional

from experiments.causal_reasoning.comparison.evaluation import Comparison
from experiments.causal_reasoning.comparison.report import ReportText
from experiments.causal_reasoning.comparison.run import (
    Experiment,
    RunSettings,
    ScalingSetup,
    run,
)
from experiments.causal_reasoning.tracy_clutter_picking.dataset import (
    ClutterPickDataset,
    HostedDataset,
    lift_summaries,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterPickScene,
    attempt_domain,
)
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    ClosingAxisSideCausesDisturbance,
    CrowdingCausesLift,
    FrictionCausesLift,
    monte_carlo_cases,
    neighbour_level_cases,
    query_catalogue,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    ClutterTruth,
    synthetic_clutter_pick_scenes,
)

logger = logging.getLogger(__name__)


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


def report_text(recorded_neighbour_count: int) -> ReportText:
    """
    :param recorded_neighbour_count: How many neighbours the recorded attempts have.
    :return: What the report says about the dataset and the pipelines.
    """
    return ReportText(
        title="Tracy clutter picking: relational circuit against flat-table trees",
        introduction=(
            "Tracy's left arm picks one milk carton out of a ten-carton clutter in "
            "MuJoCo, holding it by contact friction alone. Every attempt is recorded "
            "as a relational scene: the attempt's own attributes (environment, grasp "
            "friction, grasp yaw, whether the target came up) and one exchangeable "
            "part per neighbouring carton (its position relative to the target, its "
            "distance band, which side of the fingers' closing axis it stands on, and "
            f"how far the pick shoved it). Every recorded attempt has "
            f"{recorded_neighbour_count} neighbours, and they have no canonical "
            "order; the recording lists them in the order the layout was drawn, and "
            "nothing ties a position to an identity.",
            "Four pipelines were fitted on the same recorded attempts and asked the "
            "same `cause`/`causes_effect` EQL queries:",
            "- **relational circuit**: a relational probabilistic circuit fitted on "
            "the attempts' relational structure, one circuit over the attempt's own "
            "attributes and its aggregation count (neighbours adjacent to the "
            "target), one template over a neighbour's attributes, grounded per query "
            "into a circuit over exactly the queried attempt and neighbours and "
            "registered as a causal circuit;\n"
            "- **propositional tree**: a joint probability tree fitted on the "
            "attempts flattened into one table of the attempt's own attributes and "
            "the same count, the classic propositional summary of a relational "
            "example, registered as a causal circuit the same way;\n"
            "- **unrolled tree**: the same tree on a table that also carries every "
            "neighbour's attributes under the neighbour's position, so that a column "
            "means whatever neighbour an attempt happens to list at that position;\n"
            "- **scalars-only tree**: the same tree on the attempt's own attributes "
            "alone, what a flat learner sees without the relational feature "
            "extraction.",
            "Every flat tree answers a query by backdoor adjustment on a table column; "
            "the relational circuit does the same on the variable of a grounded "
            "circuit. In both, the model is stratified so it is support-deterministic "
            "over the cause, the effect's probability is read off every region of the "
            "cause, and any variable the query marks as a confounder is summed out of "
            "that reading. A query lists as many neighbours as the clutter it asks "
            "about has, with all their attributes open, which is what the relational "
            "circuit grounds itself for; a flat table ignores parts a query says "
            "nothing about, so it answers a question about a clutter of another size "
            "with the numbers it has for the recorded one, and refuses a query that "
            "constrains a column it does not have.",
        ),
        effect_summary=(
            "the environment the clutter stood in, by the grasp's friction "
            "coefficient, and by how many neighbours stood adjacent to the target "
            "(closer than the fingers' sweep)"
        ),
        answerability_note=(
            "A question about the crowding count needs the count: the scalars-only "
            "tree refuses it. A question whose cause and effect live on one neighbour "
            "needs the neighbours: the propositional tree refuses it, the unrolled "
            "tree answers it about whatever neighbour the attempts list at that "
            "position, and the relational circuit answers it about an exchangeable "
            "neighbour. The questions about clutters of other sizes are answered by "
            "the flat trees with the same numbers as for the recorded size, since "
            "nothing in a flat table tells the sizes apart; only a model that grounds "
            "itself for the queried objects gives a size its own answer, and only it "
            "can be asked about a neighbour beyond the last column of the unrolled "
            "table."
        ),
        reordering_note=(
            "The recording's order is the order the layout sampler drew the "
            "neighbours in, which carries nothing about where they stand."
        ),
        ground_truth_note=(
            "The mechanism the synthetic attempts are drawn from gives the hold's "
            "probability in closed form, so forcing the friction or the crowding "
            "leaves an expectation over layouts and the truth is exact in the "
            "outcome. The environment is the confounder, since a bin packs the "
            "cartons more tightly and holds only the slippery ones, and one setting "
            "removes that by letting both environments draw from the whole friction "
            "ladder."
        ),
        learning_curve_note=(
            "Every neighbour of every training attempt goes into the template."
        ),
        scaling_note=(
            "The synthetic attempts come from the same layout sampler as the recorded "
            "ones, with a random outcome in place of the simulator."
        ),
    )


def attempts_of_size(
    neighbour_count: int, attempt_count: int, random_state: np.random.Generator
) -> List[ClutterPickScene]:
    """
    :param neighbour_count: How many neighbours each attempt has.
    :param attempt_count: How many attempts to sample.
    :param random_state: Source of randomness.
    :return: Synthetic attempts on clutters of that size.
    """
    return synthetic_clutter_pick_scenes(
        random_state, attempt_count, object_count=neighbour_count + 1
    )


def tracy_experiment(recorded_neighbour_count: int) -> Experiment:
    """
    :param recorded_neighbour_count: How many neighbours the recorded attempts have.
    :return: The Tracy clutter-picking comparison: what is compared and what is asked.
    """
    return Experiment(
        comparison=Comparison(domain=attempt_domain(), summaries=lift_summaries),
        text=report_text(recorded_neighbour_count),
        cases=query_catalogue(recorded_neighbour_count),
        part_cases=neighbour_level_cases(recorded_neighbour_count),
        monte_carlo_cases=monte_carlo_cases(recorded_neighbour_count),
        truth=ClutterTruth(),
        truth_cases=[
            FrictionCausesLift(open_part_count=recorded_neighbour_count),
            CrowdingCausesLift(open_part_count=recorded_neighbour_count),
        ],
        scaling=ScalingSetup(
            examples_of_size=attempts_of_size,
            case=ClosingAxisSideCausesDisturbance(open_part_count=1, neighbour_index=0),
            sizes=(4, 9, 16, 25),
        ),
        results=ExperimentFiles().results,
    )


def main(settings: RunSettings, dataset: Optional[Path]) -> None:
    """
    Run the comparison and every study around it, and write them out.

    :param settings: The run's knobs.
    :param dataset: A local file of recorded attempts to read; the hosted ones if not
        given.
    """
    import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes

    files = ExperimentFiles()
    attempts = (
        files.recorded_attempts.load()
        if dataset is None
        else ClutterPickDataset.load(dataset)
    )
    logger.info("Read %d attempts", len(attempts.scenes))
    run(
        tracy_experiment(attempts.recorded_neighbour_count),
        attempts.examples(),
        settings,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    RunSettings.add_arguments(parser, ExperimentFiles().results)
    parser.add_argument("--dataset", type=Path, default=None)
    arguments = parser.parse_args()
    main(RunSettings.from_arguments(arguments), arguments.dataset)
