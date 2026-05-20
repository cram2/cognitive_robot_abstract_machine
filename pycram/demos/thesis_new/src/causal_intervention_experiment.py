#!/usr/bin/env python3
"""
Controlled interventional experiment for thesis causal learning.

Goal
----
Create data where causal questions are identifiable by design, not only by
observational assumptions.

Robot substitution intervention:

    do(robot_name = r)

For each environment and seed, the same bread-cutting OAAT instances are run
for every robot. Existing cutting logs contain one row per bread object. After
execution, rows are paired by:

    environment_name + seed + bread_name

This creates a proper interventional block:

    causal_instance_id, robot_name, final_success

where the scene is held fixed by the seed/environment and only the robot is
changed. That is the dataset you can use to claim genuine simulated
interventions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

RECORDS_DIR = Path(__file__).resolve().parents[1] / "records"
EXPERIMENT_DIR = RECORDS_DIR / "causal_intervention"
RAW_INTERVENTION_RESULTS = EXPERIMENT_DIR / "raw_cutting_intervention_results.csv"
MANIFEST_CSV = EXPERIMENT_DIR / "robot_substitution_manifest.csv"
PAIRED_DATASET_CSV = EXPERIMENT_DIR / "paired_robot_substitution_dataset.csv"
PAIRWISE_EFFECTS_CSV = EXPERIMENT_DIR / "paired_robot_substitution_effects.csv"
SUMMARY_JSON = EXPERIMENT_DIR / "robot_substitution_summary.json"

DEFAULT_ROBOTS = (
    "pr2",
    "hsrb",
    "tiago",
    "stretch",
    "armar7",
    "rollin_justin",
    "unitree_g1",
    "garmi"
)
DEFAULT_ENVIRONMENTS = ("apartment", "kitchen", "isr")
DEFAULT_SEEDS = tuple(range(910001, 910011))
TASK_NAME = "cut"


@dataclass(frozen=True)
class InterventionRun:
    causal_experiment_id: str
    intervention_family: str
    treatment_variable: str
    treatment_value: str
    task_name: str
    environment_name: str
    seed: int
    robot_name: str
    expected_instance_key: str


def normalize_robot_name(robot_name: object) -> str:
    mapping = {
        "justin": "rollin_justin",
        "g1": "unitree_g1",
    }
    value = str(robot_name).strip().lower()
    return mapping.get(value, value)


def normalize_environment_name(environment_name: object) -> str:
    return str(environment_name).strip().lower()


def build_manifest(
    *,
    robots: tuple[str, ...] = DEFAULT_ROBOTS,
    environments: tuple[str, ...] = DEFAULT_ENVIRONMENTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    runs = []
    for environment_name in environments:
        normalized_environment = normalize_environment_name(environment_name)
        for seed in seeds:
            causal_experiment_id = f"robot_substitution:{normalized_environment}:{seed}"
            for robot_name in robots:
                normalized_robot = normalize_robot_name(robot_name)
                runs.append(
                    InterventionRun(
                        causal_experiment_id=causal_experiment_id,
                        intervention_family="robot_substitution",
                        treatment_variable="robot_name",
                        treatment_value=normalized_robot,
                        task_name=TASK_NAME,
                        environment_name=normalized_environment,
                        seed=int(seed),
                        robot_name=normalized_robot,
                        expected_instance_key=f"{normalized_environment}:{seed}:<bread_name>",
                    )
                )
    return pd.DataFrame([asdict(run) for run in runs])


def write_manifest(manifest: pd.DataFrame) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(f"Wrote intervention manifest: {MANIFEST_CSV}")
    print(f"Planned runs: {len(manifest)}")


def execute_manifest(manifest: pd.DataFrame) -> None:
    for index, row in manifest.iterrows():
        print(
            "[causal-run] "
            f"{index + 1}/{len(manifest)} "
            f"env={row.environment_name} seed={row.seed} robot={row.robot_name}"
        )
        run_cutting_intervention_isolated(
            seed=int(row.seed),
            robot_name=row.robot_name,
            environment_name=row.environment_name,
        )


def run_cutting_intervention_isolated(
    *, seed: int, robot_name: str, environment_name: str
) -> subprocess.CompletedProcess:
    thesis_new_dir = Path(__file__).resolve().parents[1]
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(thesis_new_dir)!r}); "
        "from src.demo_cut_all_breads_retry import main_cutting; "
        "main_cutting("
        f"seed={int(seed)!r}, "
        f"robot_name={str(robot_name)!r}, "
        f"environment_name={str(environment_name)!r}"
        ")"
    )
    env = os.environ.copy()
    env["THESIS_CUT_RESULTS_CSV_PATH"] = str(RAW_INTERVENTION_RESULTS)
    result = subprocess.run([sys.executable, "-c", code], env=env)
    if result.returncode:
        print(
            "[causal-run] child failed; continuing "
            f"(seed={seed}, robot={robot_name}, environment={environment_name}, "
            f"returncode={result.returncode})"
        )
    return result


def load_raw_results(raw_results_path: Path = RAW_INTERVENTION_RESULTS) -> pd.DataFrame:
    if not raw_results_path.exists():
        raise FileNotFoundError(
            f"Raw cutting results do not exist yet: {raw_results_path}"
        )
    results = pd.read_csv(raw_results_path)
    results["seed"] = pd.to_numeric(results["seed"], errors="coerce").astype("Int64")
    results["robot_name_normalized"] = results["robot_name"].map(normalize_robot_name)
    results["environment_name_normalized"] = results["world_name"].map(
        normalize_environment_name
    )
    return results


def build_paired_dataset(
    *,
    manifest_path: Path = MANIFEST_CSV,
    raw_results_path: Path = RAW_INTERVENTION_RESULTS,
) -> tuple[pd.DataFrame, dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist yet: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest["seed"] = pd.to_numeric(manifest["seed"], errors="coerce").astype("Int64")
    manifest["robot_name_normalized"] = manifest["robot_name"].map(normalize_robot_name)
    manifest["environment_name_normalized"] = manifest["environment_name"].map(
        normalize_environment_name
    )

    results = load_raw_results(raw_results_path)
    merged = results.merge(
        manifest[
            [
                "causal_experiment_id",
                "intervention_family",
                "treatment_variable",
                "treatment_value",
                "seed",
                "robot_name_normalized",
                "environment_name_normalized",
            ]
        ],
        on=["seed", "robot_name_normalized", "environment_name_normalized"],
        how="inner",
    )

    if merged.empty:
        raise RuntimeError(
            "No rows matched the manifest. Run the experiment first or check seeds/robots/environments."
        )

    merged["causal_instance_id"] = (
        merged["environment_name_normalized"].astype(str)
        + ":"
        + merged["seed"].astype(str)
        + ":"
        + merged["bread_name"].astype(str)
    )
    merged["do_variable"] = merged["treatment_variable"]
    merged["do_value"] = merged["treatment_value"]
    merged["is_interventional"] = True
    merged["final_success_numeric"] = merged["final_success"].astype(bool).astype(int)

    block_sizes = merged.groupby("causal_instance_id")["do_value"].nunique()
    complete_robot_count = manifest["robot_name_normalized"].nunique()
    complete_blocks = block_sizes[block_sizes == complete_robot_count].index
    merged["complete_intervention_block"] = merged["causal_instance_id"].isin(
        complete_blocks
    )

    complete = merged[merged["complete_intervention_block"]].copy()
    pairwise_effects = pd.DataFrame()
    if not complete.empty:
        outcome_table = complete.pivot_table(
            index="causal_instance_id",
            columns="do_value",
            values="final_success_numeric",
            aggfunc="max",
        )
        rows = []
        robots = sorted(manifest["robot_name_normalized"].dropna().unique().tolist())
        for source_robot in robots:
            for target_robot in robots:
                if source_robot == target_robot:
                    continue
                if (
                    source_robot not in outcome_table.columns
                    or target_robot not in outcome_table.columns
                ):
                    continue
                delta = outcome_table[target_robot] - outcome_table[source_robot]
                rows.append(
                    {
                        "source_robot": source_robot,
                        "target_robot": target_robot,
                        "causal_estimand": f"E[Y(do({target_robot})) - Y(do({source_robot}))]",
                        "paired_ate": float(delta.mean()),
                        "instances": int(delta.notna().sum()),
                        "source_success_rate": float(
                            outcome_table[source_robot].mean()
                        ),
                        "target_success_rate": float(
                            outcome_table[target_robot].mean()
                        ),
                        "target_better_count": int((delta > 0).sum()),
                        "source_better_count": int((delta < 0).sum()),
                        "same_outcome_count": int((delta == 0).sum()),
                    }
                )
        pairwise_effects = pd.DataFrame(rows)

    summary = {
        "manifest_runs": int(len(manifest)),
        "matched_rows": int(len(merged)),
        "causal_instances": int(merged["causal_instance_id"].nunique()),
        "complete_causal_instances": int(len(complete_blocks)),
        "robots_per_complete_block": int(complete_robot_count),
        "robots": sorted(manifest["robot_name_normalized"].dropna().unique().tolist()),
        "environments": sorted(
            manifest["environment_name_normalized"].dropna().unique().tolist()
        ),
        "seeds": [
            int(seed) for seed in sorted(manifest["seed"].dropna().unique().tolist())
        ],
        "success_rate_by_robot": {
            str(robot): float(rate)
            for robot, rate in merged.groupby("do_value")["final_success_numeric"]
            .mean()
            .items()
        },
    }
    if not pairwise_effects.empty:
        summary["paired_ate_by_substitution"] = {
            f"{row.source_robot}->{row.target_robot}": float(row.paired_ate)
            for row in pairwise_effects.itertuples()
        }

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PAIRED_DATASET_CSV, index=False)
    if not pairwise_effects.empty:
        pairwise_effects.to_csv(PAIRWISE_EFFECTS_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merged, summary


def parse_csv_list(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_seed_range(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    if ":" in value:
        start, stop = value.split(":", 1)
        return tuple(range(int(start), int(stop)))
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robots",
        default="pr2,hsrb",
        help="Comma-separated treatment robots. Default keeps the experiment small: pr2,hsrb.",
    )
    parser.add_argument(
        "--environments",
        default="apartment",
        help="Comma-separated environments. Default: apartment.",
    )
    parser.add_argument(
        "--seeds",
        default="910001:910006",
        help="Comma-separated seeds or Python-style start:stop range. Default: 910001:910006.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the intervention manifest. Without this, only the manifest is written.",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Build paired dataset from existing raw results and manifest.",
    )
    args = parser.parse_args()

    manifest = build_manifest(
        robots=tuple(
            normalize_robot_name(robot)
            for robot in parse_csv_list(args.robots, DEFAULT_ROBOTS)
        ),
        environments=tuple(
            normalize_environment_name(environment)
            for environment in parse_csv_list(args.environments, DEFAULT_ENVIRONMENTS)
        ),
        seeds=parse_seed_range(args.seeds, DEFAULT_SEEDS),
    )
    write_manifest(manifest)

    if args.execute:
        execute_manifest(manifest)

    if args.build_dataset:
        try:
            _, summary = build_paired_dataset()
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"Could not build paired dataset yet: {exc}")
            print("Run with --execute first, then rerun with --build-dataset.")
            print(f"Expected raw intervention results at: {RAW_INTERVENTION_RESULTS}")
        else:
            print(json.dumps(summary, indent=2, sort_keys=True))
            print(f"Wrote paired interventional dataset: {PAIRED_DATASET_CSV}")
            print(f"Wrote paired causal effects: {PAIRWISE_EFFECTS_CSV}")
            print(f"Wrote summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
