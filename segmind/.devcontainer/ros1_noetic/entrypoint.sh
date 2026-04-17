#!/bin/bash

set -e

source /opt/ros/noetic/setup.bash
source /root/workspace/devel/setup.bash
source /root/.virtualenvs/pycram-segmind/bin/activate

# workspace directory
WORKSPACE_DIR="/root/workspace/src/Segmind"
LIBS_DIR="/root/libs"

# Your setup commands
pip install -U catkin_pkg rospkg rosdep
pip install -r "$WORKSPACE_DIR/requirements.txt"
pip install -e "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR/../ripple_down_rules" && git pull && pip install -r "requirements.txt" && pip install -e .
#cd "$WORKSPACE_DIR/../semantic_world" && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$WORKSPACE_DIR/../pycram" && git checkout icub_demo && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$LIBS_DIR/giskardpy" && git pull && pip install -r "requirements.txt" && pip install -e .
cd "$WORKSPACE_DIR/../giskardpy_ros" && git pull
cd "$LIBS_DIR/Multiverse" && git pull && git submodule update --init --recursive
cd "$LIBS_DIR/Multiverse/Multiverse-Launch/src/multiverse_connectors/multiverse_simulators_connector/src/mujoco_connector" && pip install -r "requirements.txt"
cd "$LIBS_DIR/Multiverse/Multiverse-Launch/src/multiverse_connectors/multiverse_ros_connector/ros_ws/multiverse_ws" && source /opt/ros/${ROS_DISTRO}/setup.bash && catkin clean -y && catkin build
cd "$WORKSPACE_DIR"
cp -r "$WORKSPACE_DIR/resources/multiverse_episodes/icub_montessori_no_hands/models/iCub" "$WORKSPACE_DIR/../pycram/resources/robots/iCub"

exec "$@"