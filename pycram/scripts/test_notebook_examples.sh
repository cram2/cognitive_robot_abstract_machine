#!/bin/bash
source /opt/ros/jazzy/setup.bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../examples" && pwd)"
cd "$EXAMPLES_DIR"
rm -rf tmp
mkdir tmp
jupytext --to notebook *.md
mv *.ipynb tmp
cd tmp
treon -v --exclude=migrate_neems.ipynb --exclude=improving_actions.ipynb