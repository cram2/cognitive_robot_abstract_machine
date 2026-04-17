#!/bin/bash

# Workspace directory (override via OVERLAY_WS env var)
OVERLAY_WS="${OVERLAY_WS:-$HOME/ros2_ws}"

# ── System dependencies ────────────────────────────────────────────────────────
apt update
apt install -y \
    python3.12-venv \
    ros-jazzy-xacro \
    ros-jazzy-navigation2 \
    ros-jazzy-py-trees-ros \
    python3-vcstool \
    git \
    ros-dev-tools \
    default-jre \
    graphviz \
    graphviz-dev \
    ros-jazzy-rclpy-message-converter \
    pip \
    python3-colcon-common-extensions

# ── ROS 2 workspace – robot description packages ──────────────────────────────
mkdir -p "$OVERLAY_WS/src"
cd "$OVERLAY_WS/src"

git clone -b ros-jazzy --single-branch https://github.com/code-iai/iai_maps.git
git clone -b ros-jazzy-main --single-branch https://github.com/code-iai/iai_pr2.git
git clone -b ros2-jazzy --single-branch https://github.com/code-iai/hsr_description.git
git clone -b ros2-jazzy --single-branch https://github.com/code-iai/iai_tracy.git \
    && rm -rf iai_tracy/iai_tracy_bringup iai_tracy/iai_tracy_ur
git clone --single-branch https://github.com/maltehue/ros2_robotiq_gripper.git \
    && rm -rf ros2_robotiq_gripper/robotiq_controllers \
               ros2_robotiq_gripper/robotiq_drivers \
               ros2_robotiq_gripper/robotiq_hardware_tests \
               ros2_robotiq_gripper/robotiq_driver
git clone -b jazzy --single-branch https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
git clone -b ros2-jazzy --single-branch https://github.com/code-iai/iai_stretch.git \
    && rm -rf iai_stretch/iai_stretch_bringup
git clone -b ros2-master --single-branch https://github.com/realsenseai/realsense-ros.git \
    && rm -rf realsense-ros/realsense2_camera \
               realsense-ros/realsense2_camera_msgs \
               realsense-ros/realsense2_rgbd_plugin \
               realsense-ros/realsense2_ros_mqtt_bridge
git clone -b ros2-main --single-branch https://github.com/code-iai/iai_tiago_description.git
git clone -b humble-devel --single-branch https://github.com/pal-robotics/pmb2_robot.git \
    && rm -rf pmb2_robot/pmb2_bringup pmb2_robot/pmb2_controller_configuration
git clone -b humble-devel --single-branch https://github.com/pal-robotics/pal_gripper.git \
    && rm -rf pal_gripper/pal_grippper_controller_configuration \
               pal_gripper/pal_gripper_gazeboo \
               pal_gripper/pal_gripper_simulation \
               pal_gripper/pal_parallel_gripper_wrapper

# ── Build the workspace ───────────────────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
cd "$OVERLAY_WS"
colcon build --merge-install

# ── Source the workspace ──────────────────────────────────────────────────────
grep -qxF "source $OVERLAY_WS/install/setup.bash" ~/.bashrc || \
    echo "source $OVERLAY_WS/install/setup.bash" >> ~/.bashrc
