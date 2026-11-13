#!/usr/bin/env bash
# %% Standard ROS workspace build
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /opt/ros/overlay_ws/install/setup.bash
source /opt/ros/cram-env/bin/activate
cd /workspace/ros_ws
python -m colcon build --merge-install --parallel-workers "${BUILD_PARALLEL_WORKERS:-2}"
