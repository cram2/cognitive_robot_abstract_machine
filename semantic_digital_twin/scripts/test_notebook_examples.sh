#!/usr/bin/env bash
set -euo pipefail

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
export AMENT_PYTHON_EXECUTABLE="$(command -v python3)"
source /opt/ros/jazzy/setup.bash

# Determine the directory of this script and change to the examples directory relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../examples" && pwd)"
cd "$EXAMPLES_DIR"

rm -rf test_tmp
mkdir -p test_tmp

# Convert markdown examples to notebooks
jupytext --to notebook *.md
mv ./*.ipynb test_tmp/
cd test_tmp

# Debug: show interpreter + key packages (helps catch env/kernel mismatches)
python3 -c "import sys; print('python:', sys.executable)"
python3 -c "import ipykernel; print('ipykernel:', ipykernel.__version__)" || true
python3 -c "import nbclient; print('nbclient:', nbclient.__version__)" || true

# Run notebooks sequentially to reduce flakiness / resource spikes and to pinpoint failures.
# Also add a timeout so a stuck cell doesn't hang CI forever.
NOTEBOOK_TIMEOUT_SECONDS="${NOTEBOOK_TIMEOUT_SECONDS:-600}"

shopt -s nullglob
notebooks=( *.ipynb )
if [[ ${#notebooks[@]} -eq 0 ]]; then
  echo "No notebooks found in $(pwd)"
  exit 0
fi

for nb in "${notebooks[@]}"; do
  echo "============================================================"
  echo "Executing notebook: $nb"
  echo "============================================================"
  timeout "$NOTEBOOK_TIMEOUT_SECONDS" treon --thread 1 -v "$nb"
done