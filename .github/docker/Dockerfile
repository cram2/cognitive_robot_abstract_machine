# To build in repo root:
# sudo docker buildx build .github/docker/
ARG FROM_IMAGE=ros:jazzy

FROM $FROM_IMAGE AS builder

ARG OVERLAY_WS=/opt/ros/overlay_ws

SHELL ["/bin/bash", "-c"]

WORKDIR /opt/ros
COPY setup_ros_workspace.sh /tmp/setup_ros_workspace.sh
RUN OVERLAY_WS=$OVERLAY_WS bash /tmp/setup_ros_workspace.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-install deps
RUN source /opt/ros/jazzy/setup.bash \
    && python3 -m venv cram-env --system-site-packages \
    && source /opt/ros/cram-env/bin/activate \
    && git clone https://github.com/cram2/cognitive_robot_abstract_machine.git \
    && pip install poetry objgraph treon jupytext jupyterquiz jupyter-book \
    && rm -rf cognitive_robot_abstract_machine


COPY entrypoint.sh /
ENTRYPOINT ["bash", "/entrypoint.sh"]
