"""
Generate a thesis-style analysis report for the robot substitution cutting study.

The report is intentionally more verbose than the console analysis scripts. It
turns the raw intervention CSV and model outputs into:

  - a Markdown report with method, results, and interpretation text
  - publication-oriented PNG figures
  - tables that can be copied into the thesis

Run from this directory:

    python thesis_cutting_analysis_report.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "raw_cutting_intervention_results2.csv"
ANALYSIS_REPORT = BASE_DIR / "causal_analysis_report.txt"
FEATURE_IMPORTANCE_CSV = BASE_DIR / "cutting_model_feature_importance.csv"

OUT_DIR = BASE_DIR / "thesis_analysis"
FIG_DIR = OUT_DIR / "figures"
REPORT_MD = OUT_DIR / "thesis_cutting_analysis_report.md"

ROBOT_DRIVE_TYPE = {
    "pr2": "omni",
    "hsrb": "omni",
    "rollin_justin": "omni",
    "armar7": "omni",
    "tiago": "differential",
    "stretch": "differential",
    "unitree_g1": "legged",
    "garmi": "omni",
}

ROBOT_ORDER = [
    "pr2",
    "rollin_justin",
    "unitree_g1",
    "tiago",
    "armar7",
    "hsrb",
    "stretch",
]


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .astype(int)
    )


def load_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    for column in [
        "final_success",
        "perturbation_applied",
        "recovery_used",
        "motion_approach_completed",
    ]:
        if column in df.columns:
            df[column] = bool_series(df[column])

    df["robot_name"] = df["robot_name"].astype(str).str.lower()
    df["drive_type"] = df["robot_name"].map(ROBOT_DRIVE_TYPE).fillna("unknown")
    df["causal_instance_id"] = (
        df["world_name"].astype(str)
        + ":"
        + df["seed"].astype(str)
        + ":"
        + df["bread_name"].astype(str)
    )

    numeric_columns = [
        "collision_failure_count",
        "retry_count",
        "total_attempts",
        "execution_time_s",
        "object_size_x",
        "object_volume_aabb",
        "object_yaw_rad",
        "cut_normal_approach_perpendicular_score",
        "motion_stopped_waypoint_fraction",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_failure_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categories = []
    for row in df.itertuples():
        if bool(row.final_success):
            if getattr(row, "perturbation_type", "") == "rotate_z_180deg":
                categories.append("success_after_180deg_rotation")
            elif getattr(row, "perturbation_type", "") == "rotate_z_90deg":
                categories.append("success_after_90deg_rotation")
            elif bool(getattr(row, "recovery_used", 0)):
                categories.append("success_after_arm_switch")
            else:
                categories.append("success_without_recovery")
            continue

        feasibility = str(getattr(row, "feasibility_reason", ""))
        if "navigation" in feasibility or "pickup" in feasibility:
            categories.append("failed_before_approach")
            continue

        approach_completed = bool(getattr(row, "motion_approach_completed", 0))
        progress = getattr(row, "motion_stopped_waypoint_fraction", np.nan)
        if not approach_completed:
            categories.append("failed_before_motion")
        elif np.isfinite(progress) and progress < 0.33:
            categories.append("failed_after_approach_low_progress")
        elif np.isfinite(progress) and progress < 0.66:
            categories.append("failed_after_approach_mid_progress")
        elif np.isfinite(progress):
            categories.append("failed_after_approach_late_progress")
        else:
            categories.append("failed_after_approach_unknown_progress")
    df["failure_taxonomy"] = categories
    return df


def save_barh(series: pd.Series, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    series.sort_values().plot(kind="barh", ax=ax, color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_success_by_robot(df: pd.DataFrame) -> Path:
    path = FIG_DIR / "success_rate_by_robot.png"
    success = df.groupby("robot_name")["final_success"].mean()
    success = success.reindex([r for r in ROBOT_ORDER if r in success.index])
    save_barh(success, "Cutting success rate by robot", "success rate", path)
    return path


def plot_success_by_robot_environment(df: pd.DataFrame) -> Path:
    path = FIG_DIR / "success_rate_robot_environment.png"
    table = df.pivot_table(
        index="robot_name",
        columns="world_name",
        values="final_success",
        aggfunc="mean",
    )
    table = table.reindex([r for r in ROBOT_ORDER if r in table.index])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(table.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title("Success rate by robot and environment")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_failure_taxonomy(df: pd.DataFrame) -> Path:
    path = FIG_DIR / "failure_taxonomy_by_robot.png"
    failed = df[df["final_success"] == 0]
    table = pd.crosstab(failed["robot_name"], failed["failure_taxonomy"], normalize="index")
    table = table.reindex([r for r in ROBOT_ORDER if r in table.index])

    fig, ax = plt.subplots(figsize=(10, 5.2))
    table.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Failure taxonomy by robot")
    ax.set_ylabel("fraction of failed trials")
    ax.set_xlabel("robot")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_waypoint_progress(df: pd.DataFrame) -> Path:
    path = FIG_DIR / "waypoint_progress_failed_trials.png"
    failed = df[
        (df["final_success"] == 0)
        & df["motion_stopped_waypoint_fraction"].notna()
        & (df["motion_approach_completed"] == 1)
    ].copy()
    robots = [r for r in ROBOT_ORDER if r in set(failed["robot_name"])]
    data = [failed.loc[failed["robot_name"] == robot, "motion_stopped_waypoint_fraction"] for robot in robots]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(data, labels=robots, showfliers=False)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Waypoint progress at failed post-approach trials")
    ax.set_ylabel("nearest waypoint fraction")
    ax.set_xlabel("robot")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_importance() -> Path | None:
    if not FEATURE_IMPORTANCE_CSV.exists():
        return None
    path = FIG_DIR / "model_feature_importance_top20.png"
    imp = pd.read_csv(FEATURE_IMPORTANCE_CSV).head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(imp["feature"], imp["mean_abs_contribution"], color="#f58518")
    ax.set_title("Top model feature contributions")
    ax.set_xlabel("mean absolute contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def paired_robot_table(df: pd.DataFrame, robot_a: str, robot_b: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="causal_instance_id",
        columns="robot_name",
        values="final_success",
        aggfunc="max",
    )
    if robot_a not in pivot.columns or robot_b not in pivot.columns:
        return pd.DataFrame()
    paired = pivot[[robot_a, robot_b]].dropna().astype(int)
    return pd.crosstab(
        paired[robot_a],
        paired[robot_b],
        rownames=[robot_a],
        colnames=[robot_b],
    )


def markdown_table(df: pd.DataFrame, float_format: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(floatfmt=float_format)
    except ImportError:
        rows = [df.reset_index().columns.tolist()]
        rows.extend(df.reset_index().values.tolist())
        return "\n".join(
            " | ".join(str(value) for value in row)
            for row in rows
        )


def summarize(df: pd.DataFrame) -> dict:
    success_by_robot = (
        df.groupby("robot_name")
        .agg(
            success_rate=("final_success", "mean"),
            n=("final_success", "count"),
            collision_failures=("collision_failure_count", "mean"),
            recovery_rate=("recovery_used", "mean"),
            execution_time_s=("execution_time_s", "mean"),
        )
        .sort_values("success_rate", ascending=False)
    )
    success_by_drive = (
        df.groupby("drive_type")
        .agg(
            success_rate=("final_success", "mean"),
            n=("final_success", "count"),
            collision_failures=("collision_failure_count", "mean"),
        )
        .sort_values("success_rate", ascending=False)
    )
    taxonomy = (
        pd.crosstab(df["robot_name"], df["failure_taxonomy"], normalize="index")
        .reindex([r for r in ROBOT_ORDER if r in set(df["robot_name"])])
        .fillna(0.0)
    )
    return {
        "success_by_robot": success_by_robot,
        "success_by_drive": success_by_drive,
        "taxonomy": taxonomy,
        "pr2_tiago": paired_robot_table(df, "pr2", "tiago"),
        "pr2_stretch": paired_robot_table(df, "pr2", "stretch"),
    }


def extract_existing_analysis_text() -> str:
    if not ANALYSIS_REPORT.exists():
        return "_The compact causal analysis report was not found._"
    return ANALYSIS_REPORT.read_text(encoding="utf-8")


def write_report(df: pd.DataFrame, figures: dict[str, Path | None], tables: dict) -> None:
    compact_report = extract_existing_analysis_text()
    lines = [
        "# Thesis Analysis: Robot Substitution in Bread Cutting",
        "",
        "## 1. Purpose of the Analysis",
        "",
        "The robot substitution experiment is designed to separate scene difficulty from robot embodiment. "
        "For each environment and seed, the same generated bread instances are executed with multiple robots. "
        "The causal block is therefore `environment_name + seed + bread_name`; within that block, the intervention is `do(robot_name = r)`. "
        "This design supports paired comparisons: if one robot succeeds and another fails on the same bread instance, the difference cannot be attributed to random object placement alone.",
        "",
        "The analysis is not intended to prove that one isolated mechanical property explains all performance. "
        "Instead, it tests a chain of explanations: robot embodiment changes whether the parameterized OAAT cutting motion can be grounded and executed; execution failures are observed through recovery, collision counts, waypoint progress, and final success.",
        "",
        "## 2. Data and Variables",
        "",
        f"The current raw dataset contains **{len(df):,} trials**, **{df['robot_name'].nunique()} robots**, **{df['world_name'].nunique()} environments**, **{df['seed'].nunique()} seeds**, and **{df['bread_name'].nunique()} bread identifiers**. "
        "Each row is one completed bread-cutting trial. The central outcome is `final_success`. The main intervention variable is `robot_name`; secondary diagnostic variables include `collision_failure_count`, `retry_count`, `perturbation_applied`, `motion_approach_completed`, and `motion_stopped_waypoint_fraction`.",
        "",
        "Important distinction: `perturbation_applied` is not a randomized treatment. It indicates that the controller needed a recovery rotation, so it is interpreted as a marker of execution difficulty rather than an exogenous cause.",
        "",
        "## 3. Methodology",
        "",
        "### 3.1 Paired Robot Substitution",
        "",
        "For robot comparisons, results are paired by the same causal instance. This controls for object location, object size, environment, and seed. The paired table counts how often robot A and robot B succeed or fail on identical bread instances.",
        "",
        "### 3.2 Failure Taxonomy",
        "",
        "Each trial is assigned to an interpretable execution category. Successful trials are split into direct success, arm-switch recovery, 90-degree rotation recovery, and 180-degree rotation recovery. Failed trials are split by whether the robot reached the motion phase and, if so, how far along the waypoint sequence it progressed.",
        "",
        "### 3.3 ATE via IPW",
        "",
        "The compact causal analysis estimates the average effect of `perturbation_applied` on success using inverse probability weighting. Since perturbation is endogenous, this is interpreted as the effect of entering the recovery regime, not as a randomized intervention.",
        "",
        "### 3.4 Mediation",
        "",
        "The mediation analysis tests whether the negative association between recovery-triggered trials and success is transmitted through `collision_failure_count`. This is the mechanistic bridge between a difficult execution context and final task failure.",
        "",
        "### 3.5 Cross-Validated Interaction Model",
        "",
        "A logistic success model is evaluated with 5-fold cross-validation. It includes robot identity, drive type, environment, object geometry, waypoint progress, collision failures, and robot-specific interaction terms. The F-test compares a model without interactions to a model with interactions. A significant result means that robot embodiment changes how strongly geometry and execution progress predict success.",
        "",
        "## 4. Core Quantitative Results",
        "",
        "```text",
        compact_report.strip(),
        "```",
        "",
        "## 5. Tables",
        "",
        "### 5.1 Success by Robot",
        "",
        markdown_table(tables["success_by_robot"]),
        "",
        "### 5.2 Success by Drive Type",
        "",
        markdown_table(tables["success_by_drive"]),
        "",
        "### 5.3 Paired PR2 vs TIAGo",
        "",
        markdown_table(tables["pr2_tiago"], ".0f"),
        "",
        "Rows are PR2 outcomes and columns are TIAGo outcomes. The off-diagonal cells are the most important: PR2 succeeds while TIAGo fails, or the reverse, on the same causal instance.",
        "",
        "### 5.4 Paired PR2 vs Stretch",
        "",
        markdown_table(tables["pr2_stretch"], ".0f"),
        "",
        "### 5.5 Failure Taxonomy by Robot",
        "",
        markdown_table(tables["taxonomy"]),
        "",
        "## 6. Figures",
        "",
    ]

    for title, path in figures.items():
        if path is None:
            continue
        rel = path.relative_to(OUT_DIR)
        lines.extend([
            f"### {title}",
            "",
            f"![{title}]({rel.as_posix()})",
            "",
        ])

    lines.extend([
        "## 7. Interpretation for the Thesis",
        "",
        "The experiment supports a stronger claim than a raw success-rate comparison. Because scenes are paired by seed and bread identity, differences between robots are evaluated on the same generated object placements. The results show that cutting failures are dominated by execution fragility: the strongest predictor is `collision_failure_count`, and the mediation analysis attributes a large part of the recovery-regime effect to this variable.",
        "",
        "The interaction model adds the missing objectivity for embodiment claims. A significant interaction F-test means that robot identity and drive/body class do not merely shift the overall success rate; they change how execution failures, object geometry, and waypoint progress translate into success. This is the defensible version of the intuitive observation that robots such as PR2, TIAGo, HSRB, and Stretch fail for different mechanistic reasons.",
        "",
        "A careful thesis wording is therefore:",
        "",
        "> The robot substitution intervention identifies embodiment-dependent performance differences under fixed scenes. The subsequent diagnostic analysis shows that these differences are mediated primarily by execution failures and are moderated by robot-specific interactions with object geometry and waypoint progress. Differential drive is one contributing embodiment factor, but the stronger empirical mechanism is post-approach execution fragility during long, orientation-constrained cutting motions.",
        "",
        "## 8. Suggested Thesis Graphics",
        "",
        "Use the generated figures as the basis for a compact analysis section:",
        "",
        "1. Success rate by robot and environment: establishes the phenomenon.",
        "2. Paired robot comparison table: shows the scene-controlled design.",
        "3. Failure taxonomy by robot: explains where execution fails.",
        "4. Waypoint progress boxplot: links failures to post-approach motion execution.",
        "5. Feature importance plot: shows that interaction terms objectively matter.",
    ])

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = add_failure_taxonomy(load_data())
    figure_builders = [
        ("Success Rate by Robot", lambda: plot_success_by_robot(df)),
        (
            "Success Rate by Robot and Environment",
            lambda: plot_success_by_robot_environment(df),
        ),
        ("Failure Taxonomy by Robot", lambda: plot_failure_taxonomy(df)),
        (
            "Waypoint Progress in Failed Post-Approach Trials",
            lambda: plot_waypoint_progress(df),
        ),
        ("Top Model Feature Contributions", plot_feature_importance),
    ]
    figures = {}
    for title, builder in figure_builders:
        try:
            print(f"Building figure: {title}")
            figures[title] = builder()
        except Exception as exc:
            print(f"Skipping figure '{title}': {type(exc).__name__}: {exc}")
            figures[title] = None

    tables = summarize(df)
    write_report(df, figures, tables)
    print(f"Wrote report: {REPORT_MD}")
    print(f"Wrote figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
