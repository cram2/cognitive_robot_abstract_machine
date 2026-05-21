# Thesis Analysis: Robot Substitution in Bread Cutting

## 1. Purpose of the Analysis

The robot substitution experiment is designed to separate scene difficulty from robot embodiment. For each environment and seed, the same generated bread instances are executed with multiple robots. The causal block is therefore `environment_name + seed + bread_name`; within that block, the intervention is `do(robot_name = r)`. This design supports paired comparisons: if one robot succeeds and another fails on the same bread instance, the difference cannot be attributed to random object placement alone.

The analysis is not intended to prove that one isolated mechanical property explains all performance. Instead, it tests a chain of explanations: robot embodiment changes whether the parameterized OAAT cutting motion can be grounded and executed; execution failures are observed through recovery, collision counts, waypoint progress, and final success.

## 2. Data and Variables

The current raw dataset contains **1,302 trials**, **7 robots**, **3 environments**, **5 seeds**, and **19 bread identifiers**. Each row is one completed bread-cutting trial. The central outcome is `final_success`. The main intervention variable is `robot_name`; secondary diagnostic variables include `collision_failure_count`, `retry_count`, `perturbation_applied`, `motion_approach_completed`, and `motion_stopped_waypoint_fraction`.

Important distinction: `perturbation_applied` is not a randomized treatment. It indicates that the controller needed a recovery rotation, so it is interpreted as a marker of execution difficulty rather than an exogenous cause.

## 3. Methodology

### 3.1 Paired Robot Substitution

For robot comparisons, results are paired by the same causal instance. This controls for object location, object size, environment, and seed. The paired table counts how often robot A and robot B succeed or fail on identical bread instances.

### 3.2 Failure Taxonomy

Each trial is assigned to an interpretable execution category. Successful trials are split into direct success, arm-switch recovery, 90-degree rotation recovery, and 180-degree rotation recovery. Failed trials are split by whether the robot reached the motion phase and, if so, how far along the waypoint sequence it progressed.

### 3.3 ATE via IPW

The compact causal analysis estimates the average effect of `perturbation_applied` on success using inverse probability weighting. Since perturbation is endogenous, this is interpreted as the effect of entering the recovery regime, not as a randomized intervention.

### 3.4 Mediation

The mediation analysis tests whether the negative association between recovery-triggered trials and success is transmitted through `collision_failure_count`. This is the mechanistic bridge between a difficult execution context and final task failure.

### 3.5 Cross-Validated Interaction Model

A logistic success model is evaluated with 5-fold cross-validation. It includes robot identity, drive type, environment, object geometry, waypoint progress, collision failures, and robot-specific interaction terms. The F-test compares a model without interactions to a model with interactions. A significant result means that robot embodiment changes how strongly geometry and execution progress predict success.

## 4. Core Quantitative Results

```text
KAUSALE ANALYSE — ROBOTER SCHNEID-INTERVENTIONSDATEN
============================================================
Generiert: 2026-05-20 19:29
Datei:     raw_cutting_intervention_results.csv

DESIGN
  N=1,302 Trials | 7 Roboter | 19 Objekte | 5 Seeds
  Balanciertes Experiment: jeder Roboter hat dieselben Objekte gesehen.

ATE (AVERAGE TREATMENT EFFECT)
  Treatment:  perturbation_applied (Objekt brauchte Recovery-Rotation)
  Outcome:    final_success
  Methode:    Inverse Probability Weighting (Hajek), B=2000
  ATE:        -0.6828  (95% CI [-0.7255, -0.6388])
  Bedeutung:  Perturbation reduziert Erfolgsrate um ~68%.

MEDIATIONSANALYSE
  Pfad:       perturbation_applied → collision_failure_count → final_success
  TE:         -0.6556
  NDE:        -0.2025  (direkter Effekt)
  NIE:        -0.4531  (über collision_failure_count)
  Proportion: 69.1% des Effekts ist mediiert

CROSS-VALIDATED REGRESSION + INTERAKTIONEN
  ROC-AUC:    0.996 ± 0.004
  Accuracy:   0.984 ± 0.007
  F-Test:     F=11.7609, p=3.947e-49
  Importance: linear log-odds contribution fallback
  Top features:
    collision_failure_count                                 3.69425
    robot_stretch:x:collision_failure_count                 1.88684
    drive_omni:x:collision_failure_count                    1.44441
    robot_hsrb:x:collision_failure_count                    1.40085
    motion_approach_completed                               1.02541
    retry_count                                             0.82095
    robot_stretch                                           0.66253
    robot_unitree_g1:x:collision_failure_count              0.62857
    drive_legged:x:collision_failure_count                  0.62857
    robot_tiago:x:collision_failure_count                   0.62370

ROBOTER-ERFOLGSRATEN
  armar7           gesamt=25.3%  baseline=66.7%  perturbed=18.2%
  hsrb             gesamt=20.4%  baseline=95.2%  perturbed=10.9%
  pr2              gesamt=58.6%  baseline=95.3%  perturbed=27.0%
  rollin_justin    gesamt=63.4%  baseline=80.2%  perturbed=43.5%
  stretch          gesamt=16.7%  baseline=100.0%  perturbed=7.2%
  tiago            gesamt=43.5%  baseline=98.1%  perturbed=21.8%
  unitree_g1       gesamt=49.5%  baseline=100.0%  perturbed=30.4%

OBJEKTGRÖSSE → ERFOLG
  Größere Objekte scheitern häufiger (Korrelation: -0.307).
  XS (<0.30m): ~49%  |  S: ~30%  |  M: ~17%  |  L (>0.38m): ~1%

KAUSALE HINWEISE
  - perturbation_type (90°/180°) ist ENDOGEN, kein Treatment.
  - PC-Algorithmus bestätigt: perturbation → collision_failures → Erfolg.
  - Geometrie (size) hat direkten Pfad auf Erfolg (PC-Befund).
```

## 5. Tables

### 5.1 Success by Robot

| robot_name    |   success_rate |       n |   collision_failures |   recovery_rate |   execution_time_s |
|:--------------|---------------:|--------:|---------------------:|----------------:|-------------------:|
| rollin_justin |          0.634 | 186.000 |                2.113 |           0.500 |             39.707 |
| pr2           |          0.586 | 186.000 |                2.780 |           0.570 |             40.556 |
| unitree_g1    |          0.495 | 186.000 |                3.753 |           0.785 |             45.857 |
| tiago         |          0.435 | 186.000 |                3.855 |           0.742 |             42.708 |
| armar7        |          0.253 | 186.000 |                4.720 |           0.887 |             53.422 |
| hsrb          |          0.204 | 186.000 |                2.500 |           0.887 |             13.466 |
| stretch       |          0.167 | 186.000 |                2.591 |           0.898 |             18.695 |

### 5.2 Success by Drive Type

| drive_type   |   success_rate |       n |   collision_failures |
|:-------------|---------------:|--------:|---------------------:|
| legged       |          0.495 | 186.000 |                3.753 |
| omni         |          0.419 | 744.000 |                3.028 |
| differential |          0.301 | 372.000 |                3.223 |

### 5.3 Paired PR2 vs TIAGo

|   pr2 |   0 |   1 |
|------:|----:|----:|
|     0 |  66 |  11 |
|     1 |  39 |  70 |

Rows are PR2 outcomes and columns are TIAGo outcomes. The off-diagonal cells are the most important: PR2 succeeds while TIAGo fails, or the reverse, on the same causal instance.

### 5.4 Paired PR2 vs Stretch

|   pr2 |   0 |   1 |
|------:|----:|----:|
|     0 |  71 |   6 |
|     1 |  84 |  25 |

### 5.5 Failure Taxonomy by Robot

| robot_name    |   failed_after_approach_late_progress |   failed_after_approach_low_progress |   failed_after_approach_mid_progress |   failed_before_approach |   success_after_180deg_rotation |   success_after_90deg_rotation |   success_after_arm_switch |   success_without_recovery |
|:--------------|--------------------------------------:|-------------------------------------:|-------------------------------------:|-------------------------:|--------------------------------:|-------------------------------:|---------------------------:|---------------------------:|
| pr2           |                                 0.194 |                                0.183 |                                0.016 |                    0.022 |                           0.048 |                          0.097 |                      0.032 |                      0.409 |
| rollin_justin |                                 0.156 |                                0.091 |                                0.011 |                    0.108 |                           0.102 |                          0.097 |                      0.043 |                      0.392 |
| unitree_g1    |                                 0.280 |                                0.177 |                                0.048 |                    0.000 |                           0.091 |                          0.129 |                      0.059 |                      0.215 |
| tiago         |                                 0.242 |                                0.274 |                                0.043 |                    0.005 |                           0.059 |                          0.097 |                      0.027 |                      0.253 |
| armar7        |                                 0.387 |                                0.285 |                                0.027 |                    0.048 |                           0.065 |                          0.091 |                      0.032 |                      0.065 |
| hsrb          |                                 0.387 |                                0.376 |                                0.027 |                    0.005 |                           0.032 |                          0.065 |                      0.000 |                      0.108 |
| stretch       |                                 0.414 |                                0.253 |                                0.167 |                    0.000 |                           0.027 |                          0.038 |                      0.000 |                      0.102 |

## 6. Figures

### Success Rate by Robot

![Success Rate by Robot](figures/success_rate_by_robot.png)

### Success Rate by Robot and Environment

![Success Rate by Robot and Environment](figures/success_rate_robot_environment.png)

### Failure Taxonomy by Robot

![Failure Taxonomy by Robot](figures/failure_taxonomy_by_robot.png)

### Waypoint Progress in Failed Post-Approach Trials

![Waypoint Progress in Failed Post-Approach Trials](figures/waypoint_progress_failed_trials.png)

### Top Model Feature Contributions

![Top Model Feature Contributions](figures/model_feature_importance_top20.png)

## 7. Interpretation for the Thesis

The experiment supports a stronger claim than a raw success-rate comparison. Because scenes are paired by seed and bread identity, differences between robots are evaluated on the same generated object placements. The results show that cutting failures are dominated by execution fragility: the strongest predictor is `collision_failure_count`, and the mediation analysis attributes a large part of the recovery-regime effect to this variable.

The interaction model adds the missing objectivity for embodiment claims. A significant interaction F-test means that robot identity and drive/body class do not merely shift the overall success rate; they change how execution failures, object geometry, and waypoint progress translate into success. This is the defensible version of the intuitive observation that robots such as PR2, TIAGo, HSRB, and Stretch fail for different mechanistic reasons.

A careful thesis wording is therefore:

> The robot substitution intervention identifies embodiment-dependent performance differences under fixed scenes. The subsequent diagnostic analysis shows that these differences are mediated primarily by execution failures and are moderated by robot-specific interactions with object geometry and waypoint progress. Differential drive is one contributing embodiment factor, but the stronger empirical mechanism is post-approach execution fragility during long, orientation-constrained cutting motions.

## 8. Suggested Thesis Graphics

Use the generated figures as the basis for a compact analysis section:

1. Success rate by robot and environment: establishes the phenomenon.
2. Paired robot comparison table: shows the scene-controlled design.
3. Failure taxonomy by robot: explains where execution fails.
4. Waypoint progress boxplot: links failures to post-approach motion execution.
5. Feature importance plot: shows that interaction terms objectively matter.
