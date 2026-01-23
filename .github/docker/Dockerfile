# To build in repo root:
# sudo docker buildx build .github/docker/
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
    && git clone -b ros2-jazzy --single-branch https://github.com/code-iai/hsr_description.git \
    && git clone -b ros2-jazzy --single-branch https://github.com/code-iai/iai_tracy.git && rm -rf iai_tracy/iai_tracy_bringup iai_tracy_ur \
    && git clone --single-branch https://github.com/maltehue/ros2_robotiq_gripper.git && rm -rf ros2_robotiq_gripper/robotiq_controllers ros2_robotiq_gripper/robotiq_drivers ros2_robotiq_gripper/robotiq_hardware_tests ros2_robotiq_gripper/robotiq_driver \
    && git clone -b jazzy --single-branch https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git \
    && git clone -b ros2-jazzy --single-branch https://github.com/code-iai/iai_stretch.git && rm -rf iai_stretch/iai_stretch_bringup \
    && git clone -b ros2-master --single-branch https://github.com/realsenseai/realsense-ros.git && rm -rf realsense-ros/realsense2_camera realsense-ros/realsense2_camera_msgs realsense-ros/realsense2_rgbd_plugin realsense-ros/realsense2_ros_mqtt_bridge \
    && git clone -b ros2-main --single-branch https://github.com/code-iai/iai_tiago_description.git \
    && git clone -b humble-devel --single-branch https://github.com/pal-robotics/pmb2_robot.git && rm -rf pmb2_robot/pmb2_bringup pmb2_robot/pmb2_controller_configuration \
    && git clone -b humble-devel --single-branch https://github.com/pal-robotics/pal_gripper.git && rm -rf pal_gripper/pal_grippper_controller_configuration pal_gripper/pal_gripper_gazeboo pal_gripper/pal_gripper_simulation pal_gripper/pal_parallel_gripper_wrapper

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