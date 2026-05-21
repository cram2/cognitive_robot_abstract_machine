"""
Compare the broad randomized cutting experiment with the paired causal
robot-substitution experiment.

Outputs:
  - random_vs_paired_robot_correlation.csv
  - random_vs_paired_robot_environment_correlation.csv
  - random_vs_paired_mechanism_correlation.csv
  - random_vs_paired_correlation_report.md
  - figures/*.png

Run from anywhere:
    python pycram/demos/thesis_new/analysis_reports/compare_random_vs_paired_cutting.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RECORDS_DIR = SCRIPT_DIR.parent / "records"
RANDOM_CSV = RECORDS_DIR / "cut_all_breads_results.csv"
PAIRED_CSV = RECORDS_DIR / "causal_intervention" / "raw_cutting_intervention_results2.csv"

OUT_DIR = SCRIPT_DIR / "random_vs_paired"
FIG_DIR = OUT_DIR / "figures"


METRICS = {
    "success_rate": ("final_success", "mean"),
    "collision_failure_count": ("collision_failure_count", "mean"),
    "retry_count": ("retry_count", "mean"),
    "recovery_rate": ("recovery_used", "mean"),
    "perturbation_rate": ("perturbation_applied", "mean"),
    "execution_time_s": ("execution_time_s", "mean"),
}


def bool_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .astype(int)
    )


def load_csv(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["source"] = source
    df["robot_name"] = df["robot_name"].astype(str).str.lower()
    df["world_name"] = df["world_name"].astype(str).str.lower()
    for column in ["final_success", "recovery_used", "perturbation_applied"]:
        if column in df.columns:
            df[column] = bool_to_int(df[column])
    for column in [
        "collision_failure_count",
        "retry_count",
        "execution_time_s",
        "motion_stopped_waypoint_fraction",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg_kwargs = {
        metric_name: pd.NamedAgg(column=column, aggfunc=aggfunc)
        for metric_name, (column, aggfunc) in METRICS.items()
        if column in df.columns
    }
    result = df.groupby(group_cols).agg(**agg_kwargs)
    result["n"] = df.groupby(group_cols).size()
    return result.reset_index()


def paired_aggregate(random_df: pd.DataFrame, paired_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    random_agg = aggregate(random_df, group_cols).rename(
        columns={name: f"random_{name}" for name in METRICS}
    )
    paired_agg = aggregate(paired_df, group_cols).rename(
        columns={name: f"paired_{name}" for name in METRICS}
    )
    merged = random_agg.merge(paired_agg, on=group_cols, how="inner")
    for metric in METRICS:
        r_col = f"random_{metric}"
        p_col = f"paired_{metric}"
        if r_col in merged.columns and p_col in merged.columns:
            merged[f"delta_{metric}"] = merged[p_col] - merged[r_col]
    return merged


def correlation_table(merged: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        r_col = f"random_{metric}"
        p_col = f"paired_{metric}"
        if r_col not in merged.columns or p_col not in merged.columns:
            continue
        valid = merged[[r_col, p_col]].dropna()
        if len(valid) < 3 or valid[r_col].nunique() < 2 or valid[p_col].nunique() < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = valid[r_col].corr(valid[p_col], method="pearson")
            spearman = valid[r_col].corr(valid[p_col], method="spearman")
        rows.append(
            {
                "grouping": group_name,
                "metric": metric,
                "n_groups": len(valid),
                "pearson": pearson,
                "spearman": spearman,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if pd.isna(value) else format(value, floatfmt))
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def scatter_plot(merged: pd.DataFrame, metric: str, label_col: str, title: str, path: Path) -> None:
    r_col = f"random_{metric}"
    p_col = f"paired_{metric}"
    data = merged[[label_col, r_col, p_col]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(data[r_col], data[p_col], s=60, color="#4c78a8")
    for row in data.itertuples():
        ax.annotate(str(getattr(row, label_col)), (getattr(row, r_col), getattr(row, p_col)), fontsize=8)
    lo = float(np.nanmin([data[r_col].min(), data[p_col].min()]))
    hi = float(np.nanmax([data[r_col].max(), data[p_col].max()]))
    pad = 0.05 * max(hi - lo, 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"random experiment {metric}")
    ax.set_ylabel(f"paired experiment {metric}")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(robot_cmp: pd.DataFrame, env_cmp: pd.DataFrame, corr: pd.DataFrame) -> None:
    report_path = OUT_DIR / "random_vs_paired_correlation_report.md"
    success_robot = corr[(corr["grouping"] == "robot") & (corr["metric"] == "success_rate")]
    success_env = corr[(corr["grouping"] == "robot_environment") & (corr["metric"] == "success_rate")]

    lines = [
        "# Randomized vs Paired Cutting Experiment Robustness Check",
        "",
        "## Purpose",
        "",
        "This analysis compares the broad randomized cutting experiment with the paired robot-substitution experiment. The randomized experiment estimates overall robustness across a broad scene distribution. The paired experiment controls scene variation by reusing the same environment, seed, and bread identifiers across robots.",
        "",
        "The comparison asks whether robot rankings and mechanism metrics remain stable across both sampling schemes.",
        "",
        "## Correlation Summary",
        "",
        dataframe_to_markdown(corr),
        "",
    ]
    if not success_robot.empty:
        r = success_robot.iloc[0]
        lines.extend([
            "## Main Interpretation",
            "",
            f"At robot level, success-rate correlation is Pearson={r['pearson']:.3f}, Spearman={r['spearman']:.3f}.",
            "",
        ])
    if not success_env.empty:
        r = success_env.iloc[0]
        lines.extend([
            f"At robot-environment level, success-rate correlation is Pearson={r['pearson']:.3f}, Spearman={r['spearman']:.3f}.",
            "",
        ])
    lines.extend([
        "If these correlations are high, the paired experiment preserves the broad robot ranking observed under random sampling. If they are low, the paired subset should be interpreted as a controlled but narrower scene distribution rather than a replacement for the randomized experiment.",
        "",
        "## Robot-Level Comparison",
        "",
        dataframe_to_markdown(robot_cmp),
        "",
        "## Robot-Environment Comparison",
        "",
        dataframe_to_markdown(env_cmp),
        "",
        "## Figures",
        "",
        "![Robot success correlation](figures/robot_success_rate_scatter.png)",
        "",
        "![Robot collision failure correlation](figures/robot_collision_failure_count_scatter.png)",
        "",
        "![Robot-environment success correlation](figures/robot_environment_success_rate_scatter.png)",
        "",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    random_df = load_csv(RANDOM_CSV, "random")
    paired_df = load_csv(PAIRED_CSV, "paired")

    robot_cmp = paired_aggregate(random_df, paired_df, ["robot_name"])
    env_cmp = paired_aggregate(random_df, paired_df, ["robot_name", "world_name"])

    robot_cmp.to_csv(OUT_DIR / "random_vs_paired_robot_correlation.csv", index=False)
    env_cmp.to_csv(OUT_DIR / "random_vs_paired_robot_environment_correlation.csv", index=False)

    corr = pd.concat(
        [
            correlation_table(robot_cmp, "robot"),
            correlation_table(env_cmp, "robot_environment"),
        ],
        ignore_index=True,
    )
    corr.to_csv(OUT_DIR / "random_vs_paired_correlation_summary.csv", index=False)

    scatter_plot(
        robot_cmp,
        "success_rate",
        "robot_name",
        "Robot success-rate correlation",
        FIG_DIR / "robot_success_rate_scatter.png",
    )
    scatter_plot(
        robot_cmp,
        "collision_failure_count",
        "robot_name",
        "Robot collision-failure correlation",
        FIG_DIR / "robot_collision_failure_count_scatter.png",
    )
    env_cmp = env_cmp.copy()
    env_cmp["robot_environment"] = env_cmp["robot_name"] + "@" + env_cmp["world_name"]
    scatter_plot(
        env_cmp,
        "success_rate",
        "robot_environment",
        "Robot-environment success-rate correlation",
        FIG_DIR / "robot_environment_success_rate_scatter.png",
    )
    write_report(robot_cmp, env_cmp, corr)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
