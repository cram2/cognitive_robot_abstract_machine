ARG FROM_IMAGE=ros:jazzy
ARG OVERLAY_WS=/opt/ros/overlay_ws

FROM $FROM_IMAGE AS builder
ARG OVERLAY_WS=/opt/ros/overlay_ws
WORKDIR $OVERLAY_WS/src
SHELL ["/bin/bash", "-c"]

WORKDIR /opt/ros
RUN apt update && apt install python3.12-venv ros-jazzy-xacro python3-vcstool git ros-dev-tools default-jre graphviz graphviz-dev -y && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN vcs import --input https://raw.githubusercontent.com/cram2/semantic_digital_twin/main/rosinstall/semantic_digital_twin-https.rosinstall
RUN mv iai_maps /opt/ros/overlay_ws/src/iai_maps
RUN mv iai_pr2 /opt/ros/overlay_ws/src/iai_pr2
RUN source /opt/ros/jazzy/setup.bash && cd $OVERLAY_WS && colcon build --symlink-install
RUN echo "source $OVERLAY_WS/install/setup.bash" >> ~/.bashrc

RUN python3 -m venv semantic_digital_twin-venv --system-site-packages  && source semantic_digital_twin-venv/bin/activate && pip install -U pip && pip install -U setuptools && pip install -r semantic_digital_twin/requirements.txt && pip install https://nc.uni-bremen.de/public.php/dav/files/nxxi4QSBcYmmxba/?accept=zip && pip install -e semantic_digital_twin

# Install the Multiverse Parser outside of the overlay workspace

RUN source /opt/ros/jazzy/setup.bash && cd $OVERLAY_WS && colcon build --symlink-install
RUN echo "source $OVERLAY_WS/install/setup.bash" >> ~/.bashrc

COPY entrypoint.sh /
ENTRYPOINT ["bash", "/entrypoint.sh"]
