#!/bin/bash
# Updated setup script to be idempotent and support real machines.

# Workspace directory (override via OVERLAY_WS env var)
export OVERLAY_WS="${OVERLAY_WS:-$HOME/ros2_ws}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure python3 is available
if ! command -v python3 &> /dev/null; then
    apt update && apt install -y python3
fi

# Run the idempotent Python setup tool
python3 "$SCRIPT_DIR/setup_workspace.py"