#!/bin/bash
# USAGE: ./test_notebook_examples.sh [--notebook <file.md>]
# If no notebook is specified, all notebooks in the examples directory will be tested.
source /opt/ros/jazzy/setup.bash

NOTEBOOK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --notebook)
            NOTEBOOK="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--notebook <file.md>]" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../examples" && pwd)"
cd "$EXAMPLES_DIR"
rm -rf tmp
mkdir tmp

if [[ -n "$NOTEBOOK" ]]; then
    if [[ ! -f "$NOTEBOOK" ]]; then
        echo "Notebook not found: $NOTEBOOK" >&2
        exit 1
    fi
    jupytext --to notebook "$NOTEBOOK"
else
    jupytext --to notebook *.md
fi

mv *.ipynb tmp
cd tmp
treon -v --exclude=migrate_neems.ipynb --exclude=improving_actions.ipynb