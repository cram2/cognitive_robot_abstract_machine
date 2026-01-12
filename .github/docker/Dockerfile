ARG FROM_IMAGE=ros:jazzy

FROM $FROM_IMAGE AS builder

ARG OVERLAY_WS=/opt/ros/overlay_ws

SHELL ["/bin/bash", "-c"]

WORKDIR /opt/ros
RUN apt update  \
    && apt install python3.12-venv ros-jazzy-xacro ros-jazzy-navigation2 python3-vcstool git ros-dev-tools default-jre graphviz graphviz-dev ros-jazzy-rclpy-message-converter pip -y  \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p $OVERLAY_WS/src \
    && cd $OVERLAY_WS/src \
    && git clone -b ros-jazzy --single-branch https://github.com/code-iai/iai_maps.git \
    && git clone -b ros-jazzy-main --single-branch https://github.com/code-iai/iai_pr2.git \
    && git clone -b ros2-jazzy --single-branch https://github.com/code-iai/hsr_description.git

# Pre-install deps
RUN source /opt/ros/jazzy/setup.bash \
    && python3 -m venv cram-env --system-site-packages \
    && source /opt/ros/cram-env/bin/activate \
    && git clone https://github.com/cram2/cognitive_robot_abstract_machine.git \
    && pip install poetry \
    && rm -rf cognitive_robot_abstract_machine

# Setup ROS2 ws
RUN source /opt/ros/jazzy/setup.bash \
    && cd $OVERLAY_WS \
    && colcon build --merge-install

RUN echo "source $OVERLAY_WS/install/setup.bash" >> ~/.bashrc

COPY entrypoint.sh /
ENTRYPOINT ["bash", "/entrypoint.sh"]