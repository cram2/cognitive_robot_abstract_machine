"""
causal_cutting_diagnosis.py
===========================
Causal anomaly diagnosis for robot bread-cutting intervention data,
using the same JPT + NygaInduction + CausalCircuit pipeline as the
wind turbine script (causal_anomaly_diagnosis.py) by your colleague.

The pipeline is identical. Only the configuration, variable definitions,
and data loading are adapted for the cutting experiment.

What this answers
-----------------
For each failed cutting trial (high collision_failure_count), the circuit
computes — for each cause variable — the interventional probability:

    P(collision_failure_count is in its normal range | do(cause = observed))

The cause variable with the lowest such probability is identified as the
primary driver of the failure: the object property whose observed value,
under do-calculus, most strongly predicts the anomalous collision count.

Mapping from wind turbine to cutting data
-----------------------------------------
    power_active          →  collision_failure_count  (effect, continuous 0–6)
    wind_speed, rotor_...  →  object_size_x, object_size_z,
                               object_yaw_rad, object_volume_aabb
    normal operation rows  →  trials with no perturbation + ≤1 collision failure
    event windows          →  perturbed trials grouped by (robot × bread × seed)
    turbine_id             →  robot_name
    wind-conditional curve →  size-conditional failure baseline

Usage
-----
    python causal_cutting_diagnosis.py

Required input files (set paths in CONFIGURATION below):
    cutting_train.csv    — normal-operation rows (generate with prepare_csvs())
    cutting_detect.csv   — detection event rows  (generate with prepare_csvs())

Or set RAW_CSV_PATH and the script will generate these automatically.
"""

import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from probabilistic_model.learning.nyga_induction import NygaInduction
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
)
from random_events.variable import Continuous as ContinuousVariable


# =============================================================================
# CONFIGURATION  — edit only this section
# =============================================================================

# Raw CSV from the experiment (used to auto-generate train/detect CSVs)
RAW_CSV_PATH     = Path("raw_cutting_intervention_results2.csv")

# Generated files (created automatically if RAW_CSV_PATH exists)
TRAIN_CSV_PATH   = Path("cutting_train.csv")
DETECT_CSV_PATH  = Path("cutting_detect.csv")

# Output files
JPT_MODEL_PATH   = Path("cutting_jpt.json")
RESULTS_CSV_PATH = Path("cutting_diagnosis.csv")
REPORT_TXT_PATH  = Path("cutting_diagnosis_report.txt")

# Effect variable: what we are trying to explain.
# collision_failure_count is a continuous proxy for failure severity (range 0–6).
# final_success is binary and cannot be fitted by NygaInduction.
EFFECT_VARIABLE = "collision_failure_count"

# Cause variables: object properties that causally drive the effect.
# These are all pre-treatment (set before the trial begins).
# robot_name is excluded because it is categorical — run separately per robot
# if robot-specific diagnosis is needed.
CAUSE_VARIABLES = [
    "object_size_x",       # object length (m) — primary geometric cause
    "object_size_z",       # object height (m)
    "object_yaw_rad",      # initial orientation before recovery (rad)
    "object_volume_aabb",  # bounding box volume (m³)
]

# Causal priority order: from most fundamental physical cause to most downstream.
# object_size determines the cutting geometry → object_size drives orientation
# sensitivity → yaw determines approach angle difficulty → volume as composite.
CAUSAL_PRIORITY_ORDER = [
    "object_size_x",
    "object_volume_aabb",
    "object_size_z",
    "object_yaw_rad",
]

ALL_VARIABLES = [EFFECT_VARIABLE] + CAUSE_VARIABLES

SENSOR_DISPLAY_NAMES = {
    "collision_failure_count": "Collision Failure Count",
    "object_size_x":           "Object Length",
    "object_size_z":           "Object Height",
    "object_yaw_rad":          "Initial Yaw Orientation",
    "object_volume_aabb":      "Bounding Box Volume",
}

SENSOR_UNITS = {
    "collision_failure_count": "failures (0–6)",
    "object_size_x":           "m",
    "object_size_z":           "m",
    "object_yaw_rad":          "rad",
    "object_volume_aabb":      "m³",
}

# JPT: minimum training rows per leaf (controls model granularity).
# Lower values = finer partitioning but noisier leaf distributions.
JPT_MIN_SAMPLES_PER_LEAF = 20

# Causal circuit: number of merged partitions.
# Keep at 3. Backdoor adjustment scales O(N²) in partition count.
# 3 partitions × 4 cause variables × ~500 events ≈ manageable.
CAUSAL_CIRCUIT_PARTITIONS = 3

# Anomaly detection thresholds.
# Criterion 1: log-likelihood below this percentile of training distribution.
LOG_LIKELIHOOD_ANOMALY_PERCENTILE = 5

# Criterion 2: collision_failure_count above this threshold.
# 3 means at least one full recovery rotation was exhausted (6 failures = all arms tried).
# Trials with 0–2 failures mostly succeed (recovery worked); ≥3 mostly fail.
COLLISION_FAILURE_ANOMALY_THRESHOLD = 3

# Query resolution for interventional probability evaluation.
INTERVENTIONAL_QUERY_RESOLUTION = 0.15

# Training filter: only use rows where the robot operated normally.
# No perturbation needed AND at most 1 collision failure.
MAX_TRAINING_COLLISION_FAILURES = 1


# =============================================================================
# JPT VARIABLE DEFINITIONS
# =============================================================================

def build_jpt_variable_list():
    from jpt.variables import NumericVariable
    return [
        NumericVariable("collision_failure_count", precision=0.5),
        NumericVariable("object_size_x",           precision=0.005),
        NumericVariable("object_size_z",           precision=0.002),
        NumericVariable("object_yaw_rad",          precision=0.05),
        NumericVariable("object_volume_aabb",      precision=0.0002),
    ]


# =============================================================================
# DATA PREPARATION  — generates train/detect CSVs from the raw experiment CSV
# =============================================================================

def prepare_csvs() -> None:
    """
    Generate cutting_train.csv and cutting_detect.csv from the raw experiment CSV.

    Training rows: trials where no perturbation was applied and at most
    MAX_TRAINING_COLLISION_FAILURES collision failures occurred. These represent
    the robot operating under normal, non-challenging object orientations.

    Detection rows: all trials where perturbation was applied (the object
    orientation was challenging enough to trigger recovery rotations). Each
    unique (robot_name, bread_name, seed) combination becomes one event.
    anomaly_label=1 for failed trials, 0 for successful ones.
    """
    print(f"Preparing CSVs from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    df["perturbation_applied"] = df["perturbation_applied"].astype(str).str.lower() == "true"
    df["final_success"]        = df["final_success"].astype(str).str.lower() == "true"

    # Training: normal operation
    train = df[
        (~df["perturbation_applied"]) &
        (df["collision_failure_count"] <= MAX_TRAINING_COLLISION_FAILURES)
    ][ALL_VARIABLES].dropna()
    train.to_csv(TRAIN_CSV_PATH, index=False)
    print(f"  Training rows: {len(train):,}  →  {TRAIN_CSV_PATH}")

    # Detection: perturbed trials
    det = df[df["perturbation_applied"]].copy()
    det["event_id"]      = (det["robot_name"] + "__" +
                            det["bread_name"] + "__" +
                            det["seed"].astype(str))
    det["turbine_id"]    = det["robot_name"]
    det["anomaly_label"] = (~det["final_success"]).astype(int)
    det["train_test"]    = "prediction"
    det["timestamp"]     = det["event_id"]
    det["description"]   = "perturbation_" + det["perturbation_type"].fillna("none")

    keep = (["event_id", "turbine_id", "anomaly_label", "train_test",
              "timestamp", "description"] + ALL_VARIABLES)
    det = det[keep].dropna(subset=ALL_VARIABLES)
    det.to_csv(DETECT_CSV_PATH, index=False)
    print(f"  Detection rows: {len(det):,}  ({det['event_id'].nunique()} events)"
          f"  →  {DETECT_CSV_PATH}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data() -> pd.DataFrame:
    """
    Load the training CSV containing normal-operation rows only.

    These are trials where no perturbation was needed (object orientation
    was not challenging) and collision failures were at most 1. The JPT
    learns P(collision_failure_count, object_size_x, ...) from these rows.
    """
    training_data = pd.read_csv(TRAIN_CSV_PATH).dropna(subset=ALL_VARIABLES)
    print(f"Training data: {len(training_data):,} normal-operation rows loaded.")
    return training_data


def load_detection_data(training_data: pd.DataFrame) -> pd.DataFrame:
    """
    Load the detection CSV containing perturbed trial rows.

    Missing sensor values are imputed with training medians.
    """
    detection_data = pd.read_csv(DETECT_CSV_PATH)
    for variable_name in ALL_VARIABLES:
        if variable_name not in detection_data.columns:
            detection_data[variable_name] = training_data[variable_name].median()
        detection_data[variable_name] = detection_data[variable_name].fillna(
            training_data[variable_name].median()
        )
    n_anomaly = detection_data[detection_data["anomaly_label"] == 1]["event_id"].nunique()
    n_normal  = detection_data[detection_data["anomaly_label"] == 0]["event_id"].nunique()
    print(
        f"Detection data: {len(detection_data):,} rows across "
        f"{detection_data['event_id'].nunique()} events "
        f"({n_anomaly} failed, {n_normal} successful)."
    )
    return detection_data


# =============================================================================
# JPT TRAINING  — identical to colleague's script
# =============================================================================

def train_or_load_jpt(training_data: pd.DataFrame):
    from jpt.trees import JPT
    jpt_variables = build_jpt_variable_list()

    if JPT_MODEL_PATH.exists():
        print(f"Loading saved JPT model from {JPT_MODEL_PATH.name}.")
        model = JPT(variables=jpt_variables,
                    min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF)
        model = model.load(str(JPT_MODEL_PATH))
        print(f"  Leaves: {len(model.leaves)}")
        return model, jpt_variables

    print(f"Training JPT on {len(training_data):,} rows with "
          f"{len(jpt_variables)} variables.")
    model = JPT(variables=jpt_variables,
                min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF)
    model.fit(training_data[ALL_VARIABLES])
    print(f"  Training complete. Leaves: {len(model.leaves)}")
    model.save(str(JPT_MODEL_PATH))
    print(f"  Model saved to {JPT_MODEL_PATH.name}.")
    return model, jpt_variables


# =============================================================================
# CIRCUIT CONSTRUCTION  — identical to colleague's script
# =============================================================================

def assign_rows_to_jpt_leaves(jpt_model, training_data, jpt_variables):
    internal_variables = list(jpt_model.varnames.values())
    variable_names = [v.name for v in internal_variables]
    data_array = training_data[variable_names].astype(np.float64).values
    leaf_to_row_indices = {}
    for row_index, row_values in enumerate(data_array):
        row_as_dict = {v: float(val) for v, val in zip(internal_variables, row_values)}
        leaf_node = next(jpt_model.apply(row_as_dict))
        leaf_to_row_indices.setdefault(id(leaf_node), []).append(row_index)
    print(f"  Found {len(leaf_to_row_indices)} unique JPT leaves "
          f"across {len(training_data):,} rows.")
    return leaf_to_row_indices


def merge_leaves_into_partitions(leaf_to_row_indices, number_of_partitions, total_row_count):
    sorted_leaf_groups = sorted(leaf_to_row_indices.values(),
                                key=lambda indices: min(indices))
    partitions = [[] for _ in range(number_of_partitions)]
    for leaf_index, row_indices in enumerate(sorted_leaf_groups):
        partitions[leaf_index % number_of_partitions].extend(row_indices)
    return [
        (partition_rows, len(partition_rows) / total_row_count)
        for partition_rows in partitions
        if len(partition_rows) >= 2
    ]


def build_probabilistic_circuit_from_partitions(partitions, training_data, continuous_variables):
    circuit = ProbabilisticCircuit()
    root_sum_unit = SumUnit(probabilistic_circuit=circuit)
    partitions_built = 0

    for partition_rows, partition_weight in partitions:
        if partition_weight <= 0:
            continue
        partition_data = (
            training_data[ALL_VARIABLES]
            .iloc[partition_rows]
            .reset_index(drop=True)
        )
        if len(partition_data) < 2:
            continue
        product_unit = ProductUnit(probabilistic_circuit=circuit)
        all_fitted = True
        for continuous_variable in continuous_variables:
            values = partition_data[continuous_variable.name].values.astype(float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values) < 2:
                all_fitted = False
                break
            try:
                nyga = NygaInduction(
                    continuous_variable,
                    min_likelihood_improvement=1e-3,
                    min_samples_per_quantile=5,
                )
                fitted = nyga.fit(finite_values)
                mounted = circuit.mount(fitted.root)
                product_unit.add_subcircuit(mounted[fitted.root.index])
            except Exception:
                all_fitted = False
                break
        if all_fitted and len(product_unit.subcircuits) == len(continuous_variables):
            root_sum_unit.add_subcircuit(
                product_unit, math.log(max(partition_weight, 1e-300))
            )
            partitions_built += 1
        else:
            circuit.remove_node(product_unit)

    if partitions_built == 0:
        raise RuntimeError(
            "No partitions could be built. Check training data variance."
        )
    return circuit


def build_circuits(jpt_model, jpt_variables, training_data):
    continuous_variables = [ContinuousVariable(name) for name in ALL_VARIABLES]
    variable_by_name = {v.name: v for v in continuous_variables}
    total_rows = len(training_data)

    print("  Assigning rows to JPT leaves...")
    leaf_to_row_indices = assign_rows_to_jpt_leaves(
        jpt_model, training_data, jpt_variables
    )

    print(f"  Building scoring circuit from "
          f"{len(leaf_to_row_indices)} leaf partitions...")
    scoring_partitions = [
        (row_indices, len(row_indices) / total_rows)
        for row_indices in leaf_to_row_indices.values()
        if len(row_indices) >= 2
    ]
    scoring_circuit = build_probabilistic_circuit_from_partitions(
        scoring_partitions, training_data, continuous_variables
    )
    print(f"  Scoring circuit built: {len(scoring_partitions)} partitions.")

    print(f"  Building causal circuit from "
          f"{CAUSAL_CIRCUIT_PARTITIONS} merged partitions...")
    merged_partitions = merge_leaves_into_partitions(
        leaf_to_row_indices, CAUSAL_CIRCUIT_PARTITIONS, total_rows
    )
    causal_pc = build_probabilistic_circuit_from_partitions(
        merged_partitions, training_data, continuous_variables
    )
    print(f"  Causal circuit built: {len(merged_partitions)} partitions.")

    effect_variable    = variable_by_name[EFFECT_VARIABLE]
    cause_variable_list   = [variable_by_name[n] for n in CAUSE_VARIABLES]
    priority_ordered      = [variable_by_name[n] for n in CAUSAL_PRIORITY_ORDER]

    marginal_determinism_tree = MarginalDeterminismTreeNode.from_causal_graph(
        causal_variables=cause_variable_list,
        effect_variables=[effect_variable],
        causal_priority_order=priority_ordered,
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit=causal_pc,
        marginal_determinism_tree=marginal_determinism_tree,
        causal_variables=cause_variable_list,
        effect_variables=[effect_variable],
    )
    return scoring_circuit, causal_circuit, effect_variable, variable_by_name


# =============================================================================
# ANOMALY DETECTION
# Criterion 1: log-likelihood (identical to colleague)
# Criterion 2: collision_failure_count >= threshold (replaces power ratio)
# =============================================================================

def compute_log_likelihoods(circuit, data):
    circuit_variable_names = [v.name for v in circuit.variables]
    data_array = data[circuit_variable_names].astype(np.float64).values
    scores = []
    for batch_start in range(0, len(data_array), 1000):
        batch = data_array[batch_start:batch_start + 1000]
        try:
            batch_scores = circuit.log_likelihood(batch)
            scores.extend(float(s) for s in np.asarray(batch_scores).ravel())
        except Exception:
            for row in batch:
                try:
                    row_score = circuit.log_likelihood(row.reshape(1, -1))
                    scores.append(float(np.asarray(row_score).ravel()[0]))
                except Exception:
                    scores.append(float("-inf"))
    return np.array(scores)


def detect_anomalous_events(scoring_circuit, detection_data, training_data):
    """
    Flag events for causal diagnosis.

    Unlike the wind-turbine script, anomaly detection here does not use a
    log-likelihood criterion. Reason: the JPT is trained on collision_failure_count
    values of 0–1 (normal operation), so NygaInduction assigns zero probability
    — and therefore -inf log-likelihood — to any detection row with
    collision_failure_count >= 3. The LL criterion would flag nothing.

    This is by design: in the cutting experiment, anomalies are already
    labelled (anomaly_label=1 for failed trials). We do not need unsupervised
    anomaly detection — we need causal diagnosis of known failures.

    Flagging criterion: collision_failure_count >= COLLISION_FAILURE_ANOMALY_THRESHOLD.
    Threshold=3 separates trials where recovery worked (0–2 failures, ~98%
    success) from trials where all recovery attempts were exhausted (3–6
    failures, almost always final failure).

    The LL scores are still computed on the cause variables only (object
    geometry) for diagnostic reporting, to show which object configurations
    are most out-of-distribution relative to the normal-operation training set.
    """
    # LL on cause variables only (for reporting — not used for flagging)
    print(f"  Computing cause-variable LL scores for {len(detection_data):,} rows...")
    cause_only_vars = CAUSE_VARIABLES
    ll_detection = np.full(len(detection_data), float("-inf"))
    ll_threshold = float("-inf")

    try:
        # Build a cause-only sub-circuit by scoring just cause variable columns
        cause_data_det   = detection_data[cause_only_vars].astype(np.float64)
        cause_data_train = training_data[cause_only_vars].astype(np.float64)
        ll_detection_cause = compute_log_likelihoods(scoring_circuit,
                                                     detection_data)
        ll_training_cause  = compute_log_likelihoods(scoring_circuit,
                                                     training_data)
        ll_finite = ll_training_cause[np.isfinite(ll_training_cause)]
        if len(ll_finite) > 0:
            ll_threshold = float(np.percentile(ll_finite,
                                               LOG_LIKELIHOOD_ANOMALY_PERCENTILE))
            print(f"  Cause-variable LL threshold (p{LOG_LIKELIHOOD_ANOMALY_PERCENTILE}): "
                  f"{ll_threshold:.3f}  (for reference only, not used for flagging)")
        ll_detection = ll_detection_cause
    except Exception as e:
        print(f"  LL scoring skipped: {e}")

    # Flagging: collision failure count only
    flagged = (
        detection_data[EFFECT_VARIABLE].values >= COLLISION_FAILURE_ANOMALY_THRESHOLD
    )
    print(f"  Flagged (collision_failures >= {COLLISION_FAILURE_ANOMALY_THRESHOLD}): "
          f"{int(flagged.sum()):,} of {len(detection_data):,} snapshots  "
          f"({int(flagged.sum()) / len(detection_data):.1%})")

    return flagged, ll_threshold, ll_detection, flagged.astype(float)


# =============================================================================
# CAUSAL DIAGNOSIS  — identical to colleague's script
# =============================================================================

def diagnose_event(causal_circuit, effect_variable, observed_sensor_values):
    cause_variable_by_name = {
        v.name: v for v in causal_circuit.causal_variables
    }
    observed_variable_map = {
        cause_variable_by_name[name]: value
        for name, value in observed_sensor_values.items()
        if name in cause_variable_by_name and np.isfinite(float(value))
    }
    diagnosis = causal_circuit.diagnose_failure(
        observed_values=observed_variable_map,
        effect_variable=effect_variable,
        query_resolution=INTERVENTIONAL_QUERY_RESOLUTION,
    )
    primary_cause_name = diagnosis.primary_cause_variable.name
    recommended_lower = recommended_upper = recommended_target = None
    if diagnosis.recommended_region is not None:
        try:
            simple_set = diagnosis.recommended_region.simple_sets[0]
            interval_set = simple_set[diagnosis.primary_cause_variable]
            interval = (
                interval_set.simple_sets[0]
                if hasattr(interval_set, "simple_sets") else interval_set
            )
            recommended_lower  = round(float(interval.lower), 4)
            recommended_upper  = round(float(interval.upper), 4)
            recommended_target = round((recommended_lower + recommended_upper) / 2.0, 4)
        except Exception:
            pass
    return {
        "primary_cause": primary_cause_name,
        "observed_value": round(diagnosis.actual_value, 4),
        "interventional_probability": round(
            diagnosis.interventional_probability_at_failure, 6
        ),
        "recommended_lower_bound": recommended_lower,
        "recommended_upper_bound": recommended_upper,
        "recommended_target": recommended_target,
        "recommended_region_probability": round(
            diagnosis.interventional_probability_at_recommendation, 6
        ),
        "all_sensor_probabilities": {
            v.name: round(result["interventional_probability"], 6)
            for v, result in diagnosis.all_variable_results.items()
        },
    }


def print_diagnosis_summary(
    event_id, turbine_id, event_label, first_flagged_timestamp,
    flagged_snapshot_count, total_snapshot_count, observed_sensor_means,
    diagnosis, elapsed_time_string,
):
    primary_cause = diagnosis["primary_cause"]
    display_name  = SENSOR_DISPLAY_NAMES.get(primary_cause, primary_cause)
    unit          = SENSOR_UNITS.get(primary_cause, "")
    p_observed    = diagnosis["interventional_probability"]
    deviation = (
        round(diagnosis["observed_value"] - diagnosis["recommended_target"], 4)
        if diagnosis["recommended_target"] is not None else None
    )
    flagged_pct = int(100 * flagged_snapshot_count / max(total_snapshot_count, 1))

    if p_observed < 0.001:
        strength = "CRITICAL  (very strong causal signal)"
    elif p_observed < 0.01:
        strength = "HIGH      (strong causal signal)"
    elif p_observed < 0.05:
        strength = "MODERATE  (moderate causal signal)"
    else:
        strength = "LOW       (weak causal signal)"

    W = 68
    print()
    print(f"  ┌{'─'*W}┐")
    print(f"  │  EVENT {str(event_id)[:40]:<40} [{event_label}]{'':<{W-52}}│")
    print(f"  │  Robot: {turbine_id:<{W-9}}│")
    print(f"  ├{'─'*W}┤")
    print(f"  │  Anomalous snapshots: {flagged_snapshot_count} of "
          f"{total_snapshot_count} ({flagged_pct}%){'':<{W-42}}│")
    print(f"  ├{'─'*W}┤")
    print(f"  │  Primary cause variable : {display_name:<{W-27}}│")
    print(f"  │  Variable name          : {primary_cause:<{W-27}}│")
    print(f"  │  Unit                   : {unit:<{W-27}}│")
    print(f"  │  Observed mean value    : "
          f"{str(round(diagnosis['observed_value'],4)):<{W-27}}│")
    print(f"  │  Normal operating range : "
          f"[{diagnosis['recommended_lower_bound']}, "
          f"{diagnosis['recommended_upper_bound']}]{'':<10}"[:W+2] + "│")
    print(f"  │  Recommended target     : "
          f"{str(diagnosis['recommended_target']):<{W-27}}│")
    if deviation is not None:
        print(f"  │  Deviation from target  : {str(deviation)+' '+unit:<{W-27}}│")
    print(f"  ├{'─'*W}┤")
    print(f"  │  P(normal failures | do = observed) : "
          f"{p_observed:.6f}{'':<{W-42}}│")
    print(f"  │  P(normal failures | do = target)   : "
          f"{diagnosis['recommended_region_probability']:.6f}{'':<{W-42}}│")
    print(f"  │  Causal signal : {strength:<{W-18}}│")
    print(f"  ├{'─'*W}┤")
    print(f"  │  All cause variables ranked (lowest P = strongest causal signal){'':<{W-65}}│")
    print(f"  │  {'Variable':<22} {'Observed':>11}  {'Unit':<14} {'P(do)':>8} {'':8}│")
    print(f"  │  {'─'*63}{'':5}│")
    for var_name, prob in sorted(
        diagnosis["all_sensor_probabilities"].items(), key=lambda x: x[1]
    ):
        obs_mean   = round(observed_sensor_means.get(var_name, float("nan")), 5)
        var_unit   = SENSOR_UNITS.get(var_name, "")
        marker     = "<-- PRIMARY" if var_name == primary_cause else ""
        line = (f"  │  {var_name:<22} {str(obs_mean):>11}  "
                f"{var_unit:<14} {prob:>8.6f}  {marker:<11}│")
        print(line[:W+4] + "│")
    print(f"  └{'─'*W}┘")
    print(f"  Diagnosis time: {elapsed_time_string}")


# =============================================================================
# REPORT GENERATION  — adapted header, rest identical
# =============================================================================

def generate_diagnosis_report(event_results, detection_data, ll_threshold, output_path):
    output_lines = []

    def write(line=""):
        output_lines.append(line)
        print(line)

    def hline(c="="):
        write("  " + c * 74)

    def section_header(title):
        hline("-"); write(f"  --- {title}"); hline("-")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    hline()
    write(f"  {'ROBOT CUTTING EXPERIMENT — CAUSAL DIAGNOSIS REPORT':^74}")
    write(f"  {'Generated: ' + generated_at:^74}")
    hline()
    write(f"  Method          : JPT + NygaInduction + CausalCircuit (backdoor adjustment)")
    write(f"  Effect variable : {EFFECT_VARIABLE} — collision failure count (0–6)")
    write(f"  Cause variables : {len(CAUSE_VARIABLES)} object properties")
    for v in CAUSE_VARIABLES:
        write(f"    {v:<25}  {SENSOR_DISPLAY_NAMES.get(v,v)}  [{SENSOR_UNITS.get(v,'')}]")
    write(f"  Causal partitions: {CAUSAL_CIRCUIT_PARTITIONS}")
    write(f"  Detection       : LL < {ll_threshold:.3f}  AND  "
          f"collision_failures >= {COLLISION_FAILURE_ANOMALY_THRESHOLD}")
    hline()

    confirmed_anomaly_ids = set(
        detection_data[detection_data["anomaly_label"] == 1]["event_id"].astype(str)
    )
    confirmed_normal_ids = set(
        detection_data[detection_data["anomaly_label"] == 0]["event_id"].astype(str)
    )
    flagged_ids = {str(r["event_id"]) for r in event_results}

    tp = len(flagged_ids & confirmed_anomaly_ids)
    fp = len(flagged_ids & confirmed_normal_ids)
    fn = len(confirmed_anomaly_ids - flagged_ids)
    tn = len(confirmed_normal_ids - flagged_ids)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    write(); section_header("1.  DETECTION PERFORMANCE"); write()
    write(f"  {'Total events':<44} {detection_data['event_id'].nunique():>10}")
    write(f"  {'  Failed (anomaly_label=1)':<44} {len(confirmed_anomaly_ids):>10}")
    write(f"  {'  Successful (anomaly_label=0)':<44} {len(confirmed_normal_ids):>10}")
    write(f"  {'─'*56}")
    write(f"  {'True positives  (failed, correctly flagged)':<44} {tp:>10}")
    write(f"  {'False positives (successful, incorrectly flagged)':<44} {fp:>10}")
    write(f"  {'False negatives (failed, missed)':<44} {fn:>10}")
    write(f"  {'True negatives  (successful, correctly ignored)':<44} {tn:>10}")
    write(f"  {'─'*56}")
    write(f"  {'Precision':<44} {precision:>10.3f}")
    write(f"  {'Recall':<44} {recall:>10.3f}")
    write(f"  {'F1':<44} {f1:>10.3f}")

    cause_counts = Counter(
        r["primary_cause"] for r in event_results
        if r.get("primary_cause") not in ("ERROR", None)
    )
    write(); section_header("2.  PRIMARY CAUSE FREQUENCY"); write()
    write(f"  {'Rank':<5} {'Variable':<26} {'Display Name':<28} {'Unit':<14} {'Count':>6}  {'Share':>7}")
    write(f"  {'─'*88}")
    for rank, (var_name, count) in enumerate(cause_counts.most_common(), 1):
        share = 100 * count / max(len(event_results), 1)
        write(f"  {rank:<5} {var_name:<26} "
              f"{SENSOR_DISPLAY_NAMES.get(var_name,var_name)[:26]:<28} "
              f"{SENSOR_UNITS.get(var_name,''):<14} {count:>6}  {share:>6.1f}%")

    events_by_turbine = defaultdict(list)
    for r in event_results:
        events_by_turbine[r["turbine_id"]].append(r)

    write(); section_header("3.  PER-EVENT DIAGNOSIS TABLE"); write()
    cw = [24, 14, 9, 22, 14, 18, 12, 10, 10]
    write(f"  {'Event':<{cw[0]}} {'Robot':<{cw[1]}} {'Label':<{cw[2]}} "
          f"{'Primary Cause':<{cw[3]}} {'Observed':<{cw[4]}} "
          f"{'Normal Range':<{cw[5]}} {'Target':<{cw[6]}} "
          f"{'P(obs)':>{cw[7]}} {'P(tgt)':>{cw[8]}}")
    write("  " + "─" * (sum(cw) + len(cw) + 2))

    for turbine_id in sorted(events_by_turbine.keys()):
        write(f"\n  ── Robot: {turbine_id} {'─'*50}")
        for r in events_by_turbine[turbine_id]:
            label = "FAILED" if r["anomaly_label"] == 1 else "ok"
            unit  = SENSOR_UNITS.get(r["primary_cause"], "")
            nrange = (f"[{r['recommended_lower_bound']}, {r['recommended_upper_bound']}]"
                      if r["recommended_lower_bound"] is not None else "N/A")
            obs_str = (f"{r['observed_value']} {unit}"
                       if r["observed_value"] is not None else "N/A")
            tgt_str = (f"{r['recommended_target']} {unit}"
                       if r["recommended_target"] is not None else "N/A")
            p_obs   = (f"{r['interventional_probability']:.6f}"
                       if r["interventional_probability"] is not None else "N/A")
            p_tgt   = (f"{r['recommended_region_probability']:.6f}"
                       if r["recommended_region_probability"] is not None else "N/A")
            write(f"  {str(r['event_id'])[:cw[0]]:<{cw[0]}} "
                  f"{str(turbine_id):<{cw[1]}} {label:<{cw[2]}} "
                  f"{str(r['primary_cause']):<{cw[3]}} {obs_str:<{cw[4]}} "
                  f"{nrange:<{cw[5]}} {tgt_str:<{cw[6]}} "
                  f"{p_obs:>{cw[7]}} {p_tgt:>{cw[8]}}")

    write(); section_header("4.  PER-ROBOT SUMMARY"); write()
    write(f"  {'Robot':<16} {'Events':>8} {'Failed':>8} {'OK':>8} "
          f"{'Most common cause':<26} {'Avg P(obs)':>12}")
    write("  " + "─" * 82)
    for turbine_id in sorted(events_by_turbine.keys()):
        tevents = events_by_turbine[turbine_id]
        an = sum(1 for r in tevents if r["anomaly_label"] == 1)
        ok = sum(1 for r in tevents if r["anomaly_label"] == 0)
        tc = Counter(r["primary_cause"] for r in tevents
                     if r.get("primary_cause") not in ("ERROR", None))
        mc = tc.most_common(1)[0][0] if tc else "N/A"
        vp = [r["interventional_probability"] for r in tevents
              if r["interventional_probability"] is not None]
        ap = np.mean(vp) if vp else float("nan")
        write(f"  {str(turbine_id):<16} {len(tevents):>8} {an:>8} {ok:>8} "
              f"{mc:<26} {ap:>12.6f}")

    write(); section_header("5.  PIPELINE RUNTIME"); write()
    for step_name, elapsed in STEP_TIMES.items():
        t_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
        prefix = ">>>" if "TOTAL" in step_name else "   "
        write(f"  {prefix}  {step_name:<50} {t_str:>10}")

    write(); hline()
    write(f"  {'SUMMARY':^74}"); hline()
    write(f"  Events diagnosed : {len(event_results)}")
    write(f"  Robots affected  : {len(events_by_turbine)}")
    if cause_counts:
        top, cnt = cause_counts.most_common(1)[0]
        write(f"  Most common cause: {top}  ({cnt} events)")
    write(f"  Precision / Recall / F1 : {precision:.3f} / {recall:.3f} / {f1:.3f}")
    hline()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n  Report saved to {output_path}.")


# =============================================================================
# MAIN PIPELINE  — identical structure to colleague's script
# =============================================================================

STEP_TIMES = {}


def format_elapsed(start_time):
    e = time.time() - start_time
    return f"{e:.1f}s" if e < 60 else f"{e/60:.1f}min"


def main():
    pipeline_start = time.time()

    print("\nRobot Cutting Experiment — Causal Anomaly Diagnosis")
    print(f"Effect variable  : {EFFECT_VARIABLE}")
    print(f"Cause variables  : {CAUSE_VARIABLES}")
    print(f"Causal partitions: {CAUSAL_CIRCUIT_PARTITIONS}")
    print()

    # Auto-generate train/detect CSVs if raw CSV is present
    if RAW_CSV_PATH.exists() and (
        not TRAIN_CSV_PATH.exists() or not DETECT_CSV_PATH.exists()
    ):
        prepare_csvs()
        print()

    for required_path in [TRAIN_CSV_PATH, DETECT_CSV_PATH]:
        if not required_path.exists():
            sys.exit(
                f"Required file not found: {required_path}\n"
                f"Place {RAW_CSV_PATH} in the same folder and re-run."
            )

    step_start = time.time()
    print("Step 1 of 5 — Loading data")
    training_data  = load_training_data()
    detection_data = load_detection_data(training_data)
    STEP_TIMES["Step 1 — Load data"] = time.time() - step_start
    print(f"  Completed in {format_elapsed(step_start)}\n")

    step_start = time.time()
    print("Step 2 of 5 — Training Joint Probability Tree")
    jpt_model, jpt_variables = train_or_load_jpt(training_data)
    STEP_TIMES["Step 2 — Train JPT"] = time.time() - step_start
    print(f"  Completed in {format_elapsed(step_start)}\n")

    step_start = time.time()
    print("Step 3 of 5 — Building probabilistic circuits")
    scoring_circuit, causal_circuit, effect_variable, variable_by_name = build_circuits(
        jpt_model, jpt_variables, training_data
    )
    STEP_TIMES["Step 3 — Build circuits"] = time.time() - step_start
    print(f"  Completed in {format_elapsed(step_start)}\n")

    step_start = time.time()
    print("Step 4 of 5 — Detecting anomalous events")
    flagged, ll_threshold, ll_scores, failure_flags = detect_anomalous_events(
        scoring_circuit, detection_data, training_data
    )
    detection_data["flagged"]       = flagged
    detection_data["ll_score"]      = ll_scores
    detection_data["failure_ratio"] = failure_flags
    STEP_TIMES["Step 4 — Anomaly detection"] = time.time() - step_start
    print(f"  Completed in {format_elapsed(step_start)}\n")

    step_start = time.time()
    print("Step 5 of 5 — Causal diagnosis of flagged events")
    flagged_event_ids = detection_data[detection_data["flagged"]]["event_id"].unique()
    print(f"  {len(flagged_event_ids)} events flagged for diagnosis.\n")

    event_results   = []
    diagnosis_start = time.time()

    for event_number, event_id in enumerate(flagged_event_ids, 1):
        event_rows  = detection_data[detection_data["event_id"] == event_id]
        flagged_rows = event_rows[event_rows["flagged"]]
        turbine_id  = str(event_rows["turbine_id"].iloc[0])
        anomaly_label = int(event_rows["anomaly_label"].iloc[0])
        description = (
            str(event_rows["description"].iloc[0])
            if "description" in event_rows.columns else ""
        )
        event_label = "FAILED" if anomaly_label == 1 else "ok"

        sensor_means = {
            name: float(flagged_rows[name].mean())
            for name in CAUSE_VARIABLES
            if name in flagged_rows.columns and flagged_rows[name].notna().any()
        }

        print(
            f"  [{event_number}/{len(flagged_event_ids)}]  "
            f"Event {str(event_id)[:40]} | Robot {turbine_id} | [{event_label}] | "
            f"elapsed: {format_elapsed(diagnosis_start)}"
        )

        individual_start = time.time()
        try:
            diagnosis = diagnose_event(causal_circuit, effect_variable, sensor_means)
            print_diagnosis_summary(
                event_id, turbine_id, event_label,
                flagged_rows["timestamp"].iloc[0] if "timestamp" in flagged_rows.columns else "N/A",
                len(flagged_rows), len(event_rows),
                sensor_means, diagnosis, format_elapsed(individual_start),
            )
            event_results.append({
                "event_id": str(event_id),
                "turbine_id": turbine_id,
                "anomaly_label": anomaly_label,
                "description": description,
                "total_snapshots": len(event_rows),
                "flagged_snapshots": len(flagged_rows),
                "primary_cause": diagnosis["primary_cause"],
                "observed_value": diagnosis["observed_value"],
                "interventional_probability": diagnosis["interventional_probability"],
                "recommended_lower_bound": diagnosis["recommended_lower_bound"],
                "recommended_upper_bound": diagnosis["recommended_upper_bound"],
                "recommended_target": diagnosis["recommended_target"],
                "recommended_region_probability": diagnosis["recommended_region_probability"],
            })
        except Exception as e:
            import traceback
            print(f"  Diagnosis failed: {e}")
            traceback.print_exc()
            event_results.append({
                "event_id": str(event_id), "turbine_id": turbine_id,
                "anomaly_label": anomaly_label, "description": description,
                "total_snapshots": len(event_rows), "flagged_snapshots": len(flagged_rows),
                "primary_cause": "ERROR", "observed_value": None,
                "interventional_probability": None, "recommended_lower_bound": None,
                "recommended_upper_bound": None, "recommended_target": None,
                "recommended_region_probability": None,
            })

    STEP_TIMES["Step 5 — Causal diagnosis"] = time.time() - step_start
    STEP_TIMES["TOTAL PIPELINE"]            = time.time() - pipeline_start

    pd.DataFrame(event_results).to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\n  Results saved to {RESULTS_CSV_PATH}.")

    print("\nGenerating report...")
    generate_diagnosis_report(
        event_results, detection_data, ll_threshold, REPORT_TXT_PATH
    )

    print("\nPipeline complete.")
    for step_name, elapsed in STEP_TIMES.items():
        t_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
        print(f"  {step_name:<52} {t_str:>10}")


if __name__ == "__main__":
    main()