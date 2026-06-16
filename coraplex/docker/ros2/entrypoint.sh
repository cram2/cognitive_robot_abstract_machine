#!/bin/bash

set -e

source /opt/ros/overlay_ws/install/setup.bash
source /opt/ros/overlay_ws/src/coraplex/coraplex-venv/bin/activate

exec "$@"