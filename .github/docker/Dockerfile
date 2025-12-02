ARG FROM_IMAGE=ros:jazzy
ARG OVERLAY_WS=/opt/ros/overlay_ws

FROM $FROM_IMAGE AS builder
ARG OVERLAY_WS=/opt/ros/overlay_ws
WORKDIR $OVERLAY_WS/src
SHELL ["/bin/bash", "-c"]

WORKDIR /opt/ros
RUN apt update  \
    && apt install python3.12-venv ros-jazzy-xacro python3-vcstool git ros-dev-tools default-jre graphviz graphviz-dev -y  \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/* \

RUN cd $OVERLAY_WS/src \
    && git clone -b ros-jazzy --single-branch https://github.com/code-iai/iai_maps.git \
    && git clone -b ros-jazzy-main --single-branch https://github.com/code-iai/iai_pr2.git \
    && git clone -b ros2-jazzy --single-branch https://github.com/code-iai/hsr_description.git \

RUN source /opt/ros/jazzy/setup.bash  \
    && cd $OVERLAY_WS  \
    && colcon build --merge-install

RUN python3 -m venv cram-env --system-site-packages  \
    && source cram-env/bin/activate  \
    && pip install poetry

RUN source /opt/ros/jazzy/setup.bash  \
    && cd $OVERLAY_WS  \
    && colcon build --symlink-install
RUN echo "source $OVERLAY_WS/install/setup.bash" >> ~/.bashrc

COPY entrypoint.sh /
ENTRYPOINT ["bash", "/entrypoint.sh"]