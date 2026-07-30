#!/bin/bash
set -e

PORT=8080
ADDR="0.0.0.0:$PORT"
DATA_DIR="/root/.local/share/code-server"

# Activate your Python virtual environment
source /root/.virtualenvs/pycram-segmind/bin/activate

# Set the default Python interpreter for VS Code
DEFAULT_PYTHON_PATH=$(which python)
export DEFAULT_PYTHON_PATH

# Start code-server
echo "🚀 Starting code-server on $ADDR"
exec code-server --bind-addr $ADDR --user-data-dir $DATA_DIR --auth none "$@"
