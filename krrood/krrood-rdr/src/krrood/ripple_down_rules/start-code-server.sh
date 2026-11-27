#!/bin/bash
set -e
if [ -z "$RDR_EDITOR_PORT" ]; then
    echo "RDR_EDITOR_PORT is not set. Using default port 8080."
    RDR_EDITOR_PORT=8080
fi
ADDR="0.0.0.0:$RDR_EDITOR_PORT"
# DATA_DIR="/root/.local/share/code-server"
echo "🚀 Starting code-server on $ADDR"
# Activate your Python virtual environment if exists else ignore
if [ -z "$RDR_VENV_PATH" ]; then
    echo "No virtual environment found. Skipping activation."
else
  source "$RDR_VENV_PATH/bin/activate"
  # Set the default Python interpreter for VS Code
  export DEFAULT_PYTHON_PATH=$(which python)
fi

# Start code-server.
echo "🚀 Starting code-server on $ADDR"
if [ -z "$CODE_SERVER_USER_DATA_DIR" ]; then
    echo "No user data directory found. Using default"
    code-server --bind-addr $ADDR --auth none "$@"
else
    echo "Using user data directory: $CODE_SERVER_USER_DATA_DIR"
    code-server --bind-addr $ADDR --user-data-dir $CODE_SERVER_USER_DATA_DIR --auth none "$@"
fi
