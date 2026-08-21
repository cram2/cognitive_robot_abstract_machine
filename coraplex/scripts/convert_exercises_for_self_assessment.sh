#!/bin/bash
# Turns the exercise sources into student notebooks: the example solutions are stripped out,
# leaving the task descriptions, the stubs and the checks that grade them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISES_DIR="$(cd "$SCRIPT_DIR/../self_assessment/exercises" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/configs/nb_remove_solutions.json"

echo "Using config file: $CONFIG_FILE"

cd "$EXERCISES_DIR"

rm -rf converted_exercises
mkdir converted_exercises

jupytext --to notebook *.md
mv *.ipynb converted_exercises
cd converted_exercises

for notebook in *.ipynb; do
    echo "Converting $notebook ..."
    jupyter nbconvert --config "$CONFIG_FILE" --to notebook --inplace "$notebook"
done
