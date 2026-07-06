#!/bin/bash

# Determine the directory of this script and change to the examples directory relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../doc/eql/user" && pwd)"
cd "$EXAMPLES_DIR"
rm -rf test_tmp
mkdir test_tmp
jupytext --to notebook *.md
mv *.ipynb test_tmp
cp _*.py test_tmp/ 2>/dev/null || true
cd test_tmp
treon --thread 1 -v