#!/bin/bash
# Executes every self-assessment exercise as a notebook, cell by cell, in order.
#
# The exercises drive a simulated robot, so a run costs minutes rather than seconds and the
# per-notebook budget is far larger than the one the example notebooks get. Override it with
# EXERCISE_TIMEOUT_SECONDS.
set -euo pipefail

# The ROS setup scripts read variables they do not set themselves, which `set -u` treats
# as an error.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
export AMENT_PYTHON_EXECUTABLE="$(command -v python3)"
source /opt/ros/jazzy/setup.bash
# The PR2 model is resolved as package://iai_pr2_description, which only the overlay knows.
if [[ -f /opt/ros/overlay_ws/install/setup.bash ]]; then
    source /opt/ros/overlay_ws/install/setup.bash
fi

EXERCISE_TIMEOUT_SECONDS="${EXERCISE_TIMEOUT_SECONDS:-3600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISES_DIR="$(cd "$SCRIPT_DIR/../self_assessment/exercises" && pwd)"
cd "$EXERCISES_DIR"

rm -rf test_tmp
mkdir test_tmp
jupytext --to notebook *.md
mv *.ipynb test_tmp
cd test_tmp

for notebook in *.ipynb; do
    echo "Running $notebook ..."
    timeout "$EXERCISE_TIMEOUT_SECONDS" treon --thread 1 -v "$notebook"
done
