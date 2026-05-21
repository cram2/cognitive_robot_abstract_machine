# Randomized vs Paired Cutting Experiment Robustness Check

## Purpose

This analysis compares the broad randomized cutting experiment with the paired robot-substitution experiment. The randomized experiment estimates overall robustness across a broad scene distribution. The paired experiment controls scene variation by reusing the same environment, seed, and bread identifiers across robots.

The comparison asks whether robot rankings and mechanism metrics remain stable across both sampling schemes.

## Correlation Summary

| grouping | metric | n_groups | pearson | spearman |
| --- | --- | --- | --- | --- |
| robot | success_rate | 7 | 0.968 | 0.893 |
| robot | collision_failure_count | 7 | 0.984 | 1.000 |
| robot | retry_count | 7 | 0.324 | 0.408 |
| robot | recovery_rate | 7 | 0.270 | -0.144 |
| robot | perturbation_rate | 7 |  |  |
| robot | execution_time_s | 7 | 0.904 | 0.679 |
| robot_environment | success_rate | 21 | 0.890 | 0.928 |
| robot_environment | collision_failure_count | 21 | 0.864 | 0.769 |
| robot_environment | retry_count | 21 | 0.219 | 0.185 |
| robot_environment | recovery_rate | 21 | 0.389 | 0.315 |
| robot_environment | perturbation_rate | 21 |  |  |
| robot_environment | execution_time_s | 21 | 0.484 | 0.474 |

## Main Interpretation

At robot level, success-rate correlation is Pearson=0.968, Spearman=0.893.

At robot-environment level, success-rate correlation is Pearson=0.890, Spearman=0.928.

If these correlations are high, the paired experiment preserves the broad robot ranking observed under random sampling. If they are low, the paired subset should be interpreted as a controlled but narrower scene distribution rather than a replacement for the randomized experiment.

## Robot-Level Comparison

| robot_name | random_success_rate | random_collision_failure_count | random_retry_count | random_recovery_rate | random_perturbation_rate | random_execution_time_s | n_x | paired_success_rate | paired_collision_failure_count | paired_retry_count | paired_recovery_rate | paired_perturbation_rate | paired_execution_time_s | n_y | delta_success_rate | delta_collision_failure_count | delta_retry_count | delta_recovery_rate | delta_perturbation_rate | delta_execution_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| armar7 | 0.268 | 0.732 | 0.000 | 0.743 | 0.000 | 37.960 | 343 | 0.253 | 4.720 | 4.043 | 0.887 | 0.855 | 53.422 | 186 | -0.016 | 3.989 | 4.043 | 0.144 | 0.855 | 15.463 |
| hsrb | 0.305 | 0.205 | 0.000 | 0.214 | 0.000 | 7.154 | 341 | 0.204 | 2.500 | 1.710 | 0.887 | 0.887 | 13.466 | 186 | -0.101 | 2.295 | 1.710 | 0.673 | 0.887 | 6.312 |
| pr2 | 0.733 | 0.267 | 0.000 | 0.293 | 0.000 | 36.604 | 352 | 0.586 | 2.780 | 2.392 | 0.570 | 0.538 | 40.556 | 186 | -0.147 | 2.513 | 2.392 | 0.277 | 0.538 | 3.953 |
| rollin_justin | 0.790 | 0.199 | 0.000 | 0.259 | 0.000 | 42.264 | 347 | 0.634 | 2.113 | 1.978 | 0.500 | 0.457 | 39.707 | 186 | -0.155 | 1.914 | 1.978 | 0.241 | 0.457 | -2.557 |
| stretch | 0.280 | 0.207 | 0.000 | 0.216 | 0.000 | 12.814 | 328 | 0.167 | 2.591 | 1.758 | 0.898 | 0.898 | 18.695 | 186 | -0.114 | 2.384 | 1.758 | 0.681 | 0.898 | 5.881 |
| tiago | 0.462 | 0.484 | 0.066 | 0.509 | 0.000 | 29.822 | 407 | 0.435 | 3.855 | 3.296 | 0.742 | 0.715 | 42.708 | 186 | -0.026 | 3.371 | 3.229 | 0.233 | 0.715 | 12.887 |
| unitree_g1 | 0.537 | 0.463 | 0.000 | 0.549 | 0.000 | 45.986 | 315 | 0.495 | 3.753 | 3.247 | 0.785 | 0.726 | 45.857 | 186 | -0.042 | 3.289 | 3.247 | 0.236 | 0.726 | -0.129 |

## Robot-Environment Comparison

| robot_name | world_name | random_success_rate | random_collision_failure_count | random_retry_count | random_recovery_rate | random_perturbation_rate | random_execution_time_s | n_x | paired_success_rate | paired_collision_failure_count | paired_retry_count | paired_recovery_rate | paired_perturbation_rate | paired_execution_time_s | n_y | delta_success_rate | delta_collision_failure_count | delta_retry_count | delta_recovery_rate | delta_perturbation_rate | delta_execution_time_s | robot_environment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| armar7 | apartment | 0.319 | 0.681 | 0.000 | 0.681 | 0.000 | 57.350 | 91 | 0.350 | 4.450 | 3.850 | 0.900 | 0.850 | 49.434 | 20 | 0.031 | 3.769 | 3.850 | 0.219 | 0.850 | -7.916 | armar7@apartment |
| armar7 | isr | 0.295 | 0.705 | 0.000 | 0.735 | 0.000 | 28.630 | 132 | 0.329 | 4.634 | 3.988 | 0.878 | 0.854 | 49.585 | 82 | 0.034 | 3.930 | 3.988 | 0.143 | 0.854 | 20.955 | armar7@isr |
| armar7 | kitchen | 0.200 | 0.800 | 0.000 | 0.800 | 0.000 | 33.519 | 120 | 0.155 | 4.869 | 4.143 | 0.893 | 0.857 | 58.118 | 84 | -0.045 | 4.069 | 4.143 | 0.093 | 0.857 | 24.599 | armar7@kitchen |
| hsrb | apartment | 0.321 | 0.333 | 0.000 | 0.333 | 0.000 | 9.499 | 81 | 0.500 | 1.850 | 1.350 | 0.750 | 0.750 | 13.075 | 20 | 0.179 | 1.517 | 1.350 | 0.417 | 0.750 | 3.576 | hsrb@apartment |
| hsrb | isr | 0.398 | 0.164 | 0.000 | 0.188 | 0.000 | 6.025 | 128 | 0.341 | 2.183 | 1.524 | 0.817 | 0.817 | 12.141 | 82 | -0.057 | 2.019 | 1.524 | 0.630 | 0.817 | 6.117 | hsrb@isr |
| hsrb | kitchen | 0.205 | 0.167 | 0.000 | 0.167 | 0.000 | 6.810 | 132 | 0.000 | 2.964 | 1.976 | 0.988 | 0.988 | 14.852 | 84 | -0.205 | 2.798 | 1.976 | 0.821 | 0.988 | 8.042 | hsrb@kitchen |
| pr2 | apartment | 0.975 | 0.025 | 0.000 | 0.074 | 0.000 | 59.836 | 81 | 0.850 | 1.100 | 0.950 | 0.300 | 0.200 | 35.722 | 20 | -0.125 | 1.075 | 0.950 | 0.226 | 0.200 | -24.115 | pr2@apartment |
| pr2 | isr | 0.641 | 0.359 | 0.000 | 0.373 | 0.000 | 30.440 | 142 | 0.537 | 3.293 | 2.841 | 0.683 | 0.659 | 40.758 | 82 | -0.104 | 2.934 | 2.841 | 0.310 | 0.659 | 10.318 | pr2@isr |
| pr2 | kitchen | 0.682 | 0.318 | 0.000 | 0.341 | 0.000 | 28.801 | 129 | 0.571 | 2.679 | 2.298 | 0.524 | 0.500 | 41.511 | 84 | -0.111 | 2.361 | 2.298 | 0.183 | 0.500 | 12.710 | pr2@kitchen |
| rollin_justin | apartment | 0.979 | 0.021 | 0.000 | 0.063 | 0.000 | 68.320 | 95 | 0.700 | 2.050 | 1.750 | 0.400 | 0.350 | 40.439 | 20 | -0.279 | 2.029 | 1.750 | 0.337 | 0.350 | -27.881 | rollin_justin@apartment |
| rollin_justin | isr | 0.731 | 0.262 | 0.000 | 0.354 | 0.000 | 32.978 | 130 | 0.707 | 2.329 | 2.305 | 0.598 | 0.549 | 40.164 | 82 | -0.023 | 2.068 | 2.305 | 0.244 | 0.549 | 7.187 | rollin_justin@isr |
| rollin_justin | kitchen | 0.705 | 0.270 | 0.000 | 0.311 | 0.000 | 31.869 | 122 | 0.548 | 1.917 | 1.714 | 0.429 | 0.393 | 39.085 | 84 | -0.157 | 1.646 | 1.714 | 0.117 | 0.393 | 7.217 | rollin_justin@kitchen |
| stretch | apartment | 0.271 | 0.376 | 0.000 | 0.388 | 0.000 | 17.790 | 85 | 0.250 | 2.350 | 1.600 | 0.800 | 0.800 | 18.087 | 20 | -0.021 | 1.974 | 1.600 | 0.412 | 0.800 | 0.297 | stretch@apartment |
| stretch | isr | 0.307 | 0.189 | 0.000 | 0.205 | 0.000 | 10.463 | 127 | 0.110 | 2.732 | 1.841 | 0.939 | 0.939 | 17.439 | 82 | -0.197 | 2.543 | 1.841 | 0.734 | 0.939 | 6.976 | stretch@isr |
| stretch | kitchen | 0.259 | 0.103 | 0.000 | 0.103 | 0.000 | 11.742 | 116 | 0.202 | 2.512 | 1.714 | 0.881 | 0.881 | 20.066 | 84 | -0.056 | 2.408 | 1.714 | 0.778 | 0.881 | 8.324 | stretch@kitchen |
| tiago | apartment | 0.703 | 0.297 | 0.000 | 0.341 | 0.000 | 46.455 | 91 | 0.600 | 2.600 | 2.200 | 0.450 | 0.450 | 31.941 | 20 | -0.103 | 2.303 | 2.200 | 0.109 | 0.450 | -14.514 | tiago@apartment |
| tiago | isr | 0.507 | 0.478 | 0.000 | 0.558 | 0.000 | 26.837 | 138 | 0.439 | 4.037 | 3.476 | 0.817 | 0.768 | 41.400 | 82 | -0.068 | 3.558 | 3.476 | 0.259 | 0.768 | 14.563 | tiago@isr |
| tiago | kitchen | 0.303 | 0.584 | 0.152 | 0.556 | 0.000 | 23.632 | 178 | 0.393 | 3.976 | 3.381 | 0.738 | 0.726 | 46.549 | 84 | 0.089 | 3.392 | 3.229 | 0.182 | 0.726 | 22.918 | tiago@kitchen |
| unitree_g1 | apartment | 0.595 | 0.405 | 0.000 | 0.476 | 0.000 | 88.660 | 84 | 0.700 | 2.200 | 1.900 | 0.550 | 0.450 | 35.155 | 20 | 0.105 | 1.795 | 1.900 | 0.074 | 0.450 | -53.505 | unitree_g1@apartment |
| unitree_g1 | isr | 0.554 | 0.446 | 0.000 | 0.515 | 0.000 | 29.040 | 130 | 0.488 | 3.841 | 3.329 | 0.817 | 0.756 | 42.420 | 82 | -0.066 | 3.395 | 3.329 | 0.302 | 0.756 | 13.380 | unitree_g1@isr |
| unitree_g1 | kitchen | 0.465 | 0.535 | 0.000 | 0.653 | 0.000 | 32.305 | 101 | 0.452 | 4.036 | 3.488 | 0.810 | 0.762 | 51.760 | 84 | -0.013 | 3.501 | 3.488 | 0.156 | 0.762 | 19.456 | unitree_g1@kitchen |

## Figures

![Robot success correlation](figures/robot_success_rate_scatter.png)

![Robot collision failure correlation](figures/robot_collision_failure_count_scatter.png)

![Robot-environment success correlation](figures/robot_environment_success_rate_scatter.png)

