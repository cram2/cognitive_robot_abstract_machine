#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the parent directory of the script (project root)
cd "$SCRIPT_DIR/.." || exit

# Build documentation using jupyter-book (config lives in doc/source)
jupyter-book build doc/source

echo "Documentation built successfully!"
