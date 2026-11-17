"""
Summarizes the rows of a circuit learning benchmark into markdown tables: how every
method fits every dataset, what training cost, how fast the queries are, and how the
runs ended.
"""

from __future__ import annotations

# %% imports
import argparse
import enum
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.circuit_learning.runner import ResultFile

# %% names


class RowType(enum.Enum):
    """
    The row types of the benchmark, as the rows files name them.
    """

    TRAINING = "TrainingResult"
    EXACT_QUERIES = "ExactQueriesResult"
    LIKELIHOOD = "LikelihoodSpeedResult"
    SAMPLING = "SamplingSpeedResult"
    CONDITIONING = "ConditioningSpeedResult"
    TRUNCATION = "TruncationSpeedResult"


# %% summary


@dataclass(kw_only=True)
class BenchmarkSummary:
    """
    The tables of the rows in an output directory.
    """

    output: Path
    """
    The output directory of the benchmark.
    """

    def rows(self) -> dict[RowType, pd.DataFrame]:
        """
        :return: Every row, by its type.
        """
        records: dict[RowType, list[dict]] = {row_type: [] for row_type in RowType}
        for path in sorted((self.output / ResultFile.ROWS.value).glob("*.jsonl")):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                records[RowType(record.pop("row_type"))].append(record)
        return {
            row_type: pd.DataFrame(entries) for row_type, entries in records.items()
        }

    def runs(self) -> pd.DataFrame:
        """
        :return: The end of every run, the latest record of a run winning.
        """
        path = self.output / ResultFile.RUNS.value
        records = [json.loads(line) for line in path.read_text().splitlines()]
        return pd.DataFrame(records).drop_duplicates("key", keep="last")

    @staticmethod
    def setting_column(*, frame: pd.DataFrame) -> pd.Series:
        """
        :return: The setting of every row as one short string.
        """
        return (
            frame["array_library"].str.lower()
            + "/"
            + frame["device"].str.lower()
            + "/"
            + frame["floating_point_type"].str.lower().str.replace("float", "f")
            + "/"
            + frame["edge_representation"].str.lower()
        )

    def markdown(self) -> str:
        """
        :return: Every table.
        """
        rows = self.rows()
        sections = []
        training = rows[RowType.TRAINING]
        if not training.empty:
            training = training.assign(setting=self.setting_column(frame=training))
            first = training.sort_values("training_seconds").drop_duplicates(
                ["learning_method", "dataset", "setting"]
            )
            sections.append("## Test log-likelihood, best setting\n")
            sections.append(
                first.pivot_table(
                    index="learning_method",
                    columns="dataset",
                    values="average_test_log_likelihood",
                    aggfunc="max",
                )
                .round(2)
                .to_markdown()
            )
            sections.append(
                "\n## Test log-likelihood of the supported rows, and percentage of "
                "impossible test rows\n"
            )
            supported = first.pivot_table(
                index="learning_method",
                columns="dataset",
                values="average_test_log_likelihood_of_supported_rows",
                aggfunc="max",
            ).round(2)
            impossible = (
                first.pivot_table(
                    index="learning_method",
                    columns="dataset",
                    values="fraction_of_test_rows_without_support",
                    aggfunc="min",
                )
                * 100
            ).round(2)
            sections.append(
                (
                    supported.astype(str) + " (" + impossible.astype(str) + "%)"
                ).to_markdown()
            )
            sections.append("\n## Training, every setting\n")
            sections.append(
                first[
                    [
                        "dataset",
                        "learning_method",
                        "setting",
                        "average_train_log_likelihood",
                        "average_test_log_likelihood",
                        "fraction_of_test_rows_without_support",
                        "average_test_log_likelihood_of_supported_rows",
                        "training_seconds",
                        "epochs_until_convergence",
                        "converged",
                        "peak_memory_bytes",
                        "number_of_parameters",
                        "number_of_edges",
                    ]
                ]
                .sort_values(["dataset", "learning_method", "setting"])
                .round(3)
                .to_markdown(index=False)
            )
        likelihood = rows[RowType.LIKELIHOOD]
        if not likelihood.empty:
            likelihood = likelihood.assign(
                setting=self.setting_column(frame=likelihood)
            )
            sections.append("\n## Likelihood, rows per second\n")
            sections.append(
                likelihood.pivot_table(
                    index=["dataset", "learning_method"],
                    columns="setting",
                    values="rows_per_second",
                )
                .round(0)
                .to_markdown()
            )
        for row_type, size, value in (
            (
                RowType.CONDITIONING,
                "number_of_conditioned_variables",
                "conditioning_time",
            ),
            (RowType.TRUNCATION, "number_of_simple_sets", "truncation_time"),
        ):
            frame = rows[row_type]
            if frame.empty:
                continue
            frame = frame.assign(
                setting=self.setting_column(frame=frame),
                milliseconds=frame[value].map(lambda measured: measured["mean"]),
            )
            sections.append(f"\n## {row_type.name.lower()}, mean milliseconds\n")
            sections.append(
                frame.pivot_table(
                    index=["dataset", "learning_method", "setting"],
                    columns=size,
                    values="milliseconds",
                )
                .round(2)
                .to_markdown()
            )
        sampling = rows[RowType.SAMPLING]
        if not sampling.empty:
            sampling = sampling.assign(setting=self.setting_column(frame=sampling))
            sections.append("\n## Sampling, samples per second\n")
            sections.append(
                sampling.pivot_table(
                    index=["dataset", "learning_method"],
                    columns="setting",
                    values="samples_per_second",
                )
                .round(0)
                .to_markdown()
            )
        exact = rows[RowType.EXACT_QUERIES]
        if not exact.empty:
            sections.append("\n## Exact queries\n")
            sections.append(
                exact.assign(exact_queries=exact["exact_queries"].map(", ".join))[
                    ["dataset", "learning_method", "exact_queries"]
                ]
                .sort_values(["dataset", "learning_method"])
                .to_markdown(index=False)
            )
        runs = self.runs()
        sections.append("\n## Runs\n")
        sections.append(
            runs.groupby(["kind", "status"]).size().unstack(fill_value=0).to_markdown()
        )
        return "\n".join(sections)


# %% command line


def main() -> None:
    """
    Print the tables of an output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(BenchmarkSummary(output=parser.parse_args().output).markdown())


if __name__ == "__main__":
    main()
