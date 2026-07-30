#!/bin/bash
set -euxo pipefail

# workspace directory
WORKSPACE_DIR="/root/workspace/src/Segmind"
LIBS_DIR="/root/libs"

# Wait for files to be mounted
while [ ! -f "$WORKSPACE_DIR/requirements.txt" ]; do
  echo "Waiting for repo to mount..."
  sleep 1
done

# Your setup commands
source /root/.virtualenvs/pycram-segmind/bin/activate
pip install -r "$WORKSPACE_DIR/requirements.txt"
pip install -e "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR/../ripple_down_rules" && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$WORKSPACE_DIR/../semantic_world" && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$WORKSPACE_DIR/../pycram" && git checkout icub_demo_ros2 && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$LIBS_DIR/giskardpy" && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$LIBS_DIR/Multiverse-Parser" && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$LIBS_DIR/Multiverse-Resources" && git pull && git submodule update --init --recursive