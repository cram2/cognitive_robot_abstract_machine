"""
Fit every pipeline on the CTU Mutagenesis molecules, ask them every question of the
catalogue, repeat that over random orderings of the parts, follow the relational answers
as grounding draws more samples, measure how the likelihoods grow with the training set,
and write it all out as Markdown.

Run with::

    python -m experiments.causal_reasoning.mutagenesis.run_pipeline
        [--output RESULTS.md] [--train-fraction F] [--seed N]
        [--min-samples-per-leaf F] [--plain-min-samples-per-leaf F]
        [--orderings N] [--splits N] [--min-region-support N]

The molecules are fetched from the CTU relational-dataset repository. The data access
objects the relational pipeline fits on come from the ``experiments`` package's
generated ORM interface; build it with ``scripts/regenerate_all_orm.py`` first.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from experiments.causal_reasoning.comparison.evaluation import Comparison
from experiments.causal_reasoning.comparison.report import ReportText
from experiments.causal_reasoning.comparison.run import Experiment, RunSettings, run
from experiments.causal_reasoning.mutagenesis.dataset import (
    fetch_mutagenesis_molecules,
    mutagenesis_dataset,
    mutagenicity_summaries,
)
from experiments.causal_reasoning.mutagenesis.domain import molecule_domain
from experiments.causal_reasoning.mutagenesis.queries import (
    atom_level_cases,
    monte_carlo_cases,
    query_catalogue,
)

logger = logging.getLogger(__name__)


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


def report_text() -> ReportText:
    """
    :return: What the report says about the dataset and the pipelines.
    """
    return ReportText(
        title="Mutagenesis: relational circuit against flat-table trees",
        introduction=(
            "The CTU Mutagenesis dataset records 188 nitroaromatic molecules, each as "
            "its own attributes (the `ind1` structural indicator, `logp`, `lumo`, and "
            "whether it tested mutagenic) with one exchangeable part per atom "
            "(element, atom-type code, partial charge, number of bonds) and one per "
            "bond (its type). A molecule has between 14 and 40 atoms, and its atoms "
            "have no canonical order; the dataset lists heavy atoms first and "
            "hydrogens last, but nothing ties position to identity.",
            "Four pipelines were fitted on the same molecules and asked the same "
            "`cause`/`causes_effect` EQL queries:",
            "- **relational circuit**: a relational probabilistic circuit fitted on "
            "the molecules' relational structure, one circuit over the molecule's own "
            "attributes and its aggregation counts (atoms, chlorine atoms, branching "
            "atoms, double bonds, aromatic bonds), one template over an atom's "
            "attributes and one over a bond's, grounded per query into a circuit over "
            "exactly the queried molecule, atoms and bonds and registered as a causal "
            "circuit;\n"
            "- **propositional tree**: a joint probability tree fitted on the "
            "molecules flattened into one table of the molecule's own attributes and "
            "the same five counts, the classic propositional summary of a relational "
            "example, registered as a causal circuit the same way;\n"
            "- **unrolled tree**: the same tree on a table that also carries every "
            "atom's and bond's attributes under the part's position, padded with an "
            "absent marker past a molecule's last part, so that a column means "
            "whatever part a molecule happens to list at that position;\n"
            "- **scalars-only tree**: the same tree on the molecule's own attributes "
            "alone, what a flat learner sees without the relational feature "
            "extraction.",
            "Every flat tree answers a query by backdoor adjustment on a table column; "
            "the relational circuit does the same on the variable of a grounded "
            "circuit. In both, the model is stratified so it is support-deterministic "
            "over the cause, the effect's probability is read off every region of the "
            "cause, and any variable the query marks as a confounder is summed out of "
            "that reading. Every query lists one atom and one bond with all their "
            "attributes open, which is what makes grounding retain the molecule's "
            "counts as variables; a flat table ignores parts a query says nothing "
            "about and refuses a query that constrains a column it does not have.",
        ),
        effect_summary=(
            "the `ind1` indicator, by how many branching atoms (atoms with three or "
            "four bonds, the ring-fusion and branch points of the molecular graph) the "
            "molecule has, by how many of its bonds are aromatic, and by how many "
            "atoms it holds at all"
        ),
        answerability_note=(
            "A question about counts needs the counts: the scalars-only tree refuses "
            "it. A question whose effect is one atom's own attribute needs the atoms: "
            "the propositional tree refuses it, the unrolled tree answers it about "
            "whatever atom the molecules list at that position, and the relational "
            "circuit answers it about an exchangeable atom. What an answer about "
            '"atom 0" is worth is what the reordering below measures.'
        ),
        reordering_note=(
            "The dataset's order is the order the molecules were drawn in, heavy atoms "
            "first and hydrogens last, so a column that addresses an atom by position "
            "addresses an element more often than chance, but never the same atom."
        ),
        learning_curve_note=(
            "Every atom and every bond of every training molecule goes into the "
            "templates."
        ),
    )


def mutagenesis_experiment() -> Experiment:
    """
    :return: The Mutagenesis comparison: what is compared and what is asked.
    """
    return Experiment(
        comparison=Comparison(
            domain=molecule_domain(), summaries=mutagenicity_summaries
        ),
        text=report_text(),
        cases=query_catalogue(),
        part_cases=atom_level_cases(),
        monte_carlo_cases=monte_carlo_cases(),
        results=ExperimentFiles().results,
    )


def main(settings: RunSettings) -> None:
    """
    Run the comparison and every study around it, and write them out.

    :param settings: The run's knobs.
    """
    import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes

    dataset = mutagenesis_dataset(fetch_mutagenesis_molecules())
    logger.info("Read %d molecules", len(dataset.examples))
    run(mutagenesis_experiment(), dataset, settings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    RunSettings.add_arguments(parser, ExperimentFiles().results)
    main(RunSettings.from_arguments(parser.parse_args()))
