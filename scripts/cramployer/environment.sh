#!/usr/bin/env bash
# %% ROS and Python environment
source /opt/ros/jazzy/setup.bash
source /opt/ros/overlay_ws/install/setup.bash
if [[ -f /workspace/ros_ws/install/setup.bash ]]; then
    source /workspace/ros_ws/install/setup.bash
fi
source /opt/ros/cram-env/bin/activate
