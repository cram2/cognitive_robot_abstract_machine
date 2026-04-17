FROM osrf/ros:noetic-desktop-full-focal as base

SHELL [ "/bin/bash" , "-c" ]

# Remove old ROS key and install updated one
RUN sudo apt-get install -y curl gnupg lsb-release && \
    sudo rm -f /etc/apt/sources.list.d/ros-latest.list && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros.list && \
    sudo apt-get update

# Now safe to do full upgrade
RUN sudo apt-get upgrade -y

RUN sudo apt update && sudo apt upgrade -y

RUN sudo apt install software-properties-common -y && \
    sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt update && \
    sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Install pycram dependencies
RUN sudo apt install wget ros-noetic-xacro python3-vcstool git ros-dev-tools default-jre python3-catkin-tools -y

RUN sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    sudo apt-get update && \
    sudo apt-get upgrade -y

RUN sudo apt install ros-noetic-joint-trajectory-controller ros-noetic-ros-control ros-noetic-ros-controllers -y && \
    sudo apt install ros-noetic-py-trees -y && \
    sudo apt install ros-noetic-rospy-message-converter -y && \
    sudo apt install ros-noetic-py-trees-ros -y

# Install RDR dependencies
RUN sudo apt-get install graphviz graphviz-dev -y

# Install additional dependencies for audio processing
RUN sudo apt-get update && apt-get install -y \
    libasound2 \
    libsdl2-mixer-2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the user to the non-root user
ARG USERNAME=root

# Switch to non-root user
USER $USERNAME
ENV HOME=/root

# Create overlay workspace
WORKDIR $HOME/workspace/src
COPY ./packages.repos .

ARG LIBSDIR=$HOME/libs

RUN mkdir -p $LIBSDIR && \
    cd $LIBSDIR && \
    git clone -b giskard_library https://github.com/SemRoCo/giskardpy.git && \
    git clone https://github.com/qpSWIFT/qpSWIFT

RUN cd $LIBSDIR && \
    git clone -b main https://github.com/Multiverse-Framework/Multiverse.git --depth 1 && \
    cd $LIBSDIR/Multiverse && \
    git submodule update --init --recursive

RUN cd $HOME/workspace/src \
    && git clone https://github.com/orocos/orocos_kinematics_dynamics.git \
    && sudo add-apt-repository universe \
    && sudo apt update \
    && sudo apt-get install libeigen3-dev libcppunit-dev -y \
    && sudo apt-get install python3-psutil python3-future -y \
    && cd orocos_kinematics_dynamics \
    && git submodule update --init

RUN vcs import --input https://raw.githubusercontent.com/AbdelrhmanBassiouny/pycram/icub_demo/rosinstall/pycram-https.rosinstall\
    && source /opt/ros/$ROS_DISTRO/setup.bash \
    && cd .. \
    && catkin build

RUN vcs import --input ./packages.repos --recursive
RUN cd $HOME/workspace/src/ripple_down_rules && \
    git pull

# Add sourcing ROS setup.bash to .bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> $HOME/.bashrc

# Install virtualenv and virtualenvwrapper
RUN python3.10 -m venv $HOME/.virtualenvs/venv && \
    source $HOME/.virtualenvs/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install virtualenvwrapper

# Add virtualenvwrapper to bashrc
RUN echo "export WORKON_HOME=$HOME/.virtualenvs" >> $HOME/.bashrc && \
    echo "export VIRTUALENVWRAPPER_PYTHON=$HOME/.virtualenvs/venv/bin/python" >> $HOME/.bashrc && \
    echo "export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.virtualenvs/venv/bin/virtualenv" >> $HOME/.bashrc && \
    echo "source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh" >> $HOME/.bashrc

# Set up Multiverse Python connector
RUN echo "export PYTHONPATH=$PYTHONPATH:$LIBSDIR/Multiverse/Multiverse-Launch/src/multiverse_connectors/multiverse_python_connector" >> $HOME/.bashrc

# Set up aliases for convenience
RUN echo "alias ease25_demo='$LIBSDIR/Multiverse/Multiverse-Launch/bin/multiverse_launch $LIBSDIR/Multiverse/Multiverse-Resources/muv/EASE2025/iCub_with_montessori_toys_2.muv'" >> $HOME/.bashrc && \
    echo "alias giskard_icup='roslaunch giskardpy_ros giskardpy_icub_mujoco_vel.launch'" >> $HOME/.bashrc && \
    echo "alias replay_icup='python $LIBSDIR/Multiverse/Multiverse-Utilities/scripts/multiverse_replay.py --data_path=~/workspace/src/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/data_ref.csv'" >> $HOME/.bashrc

ENV WORKON_HOME=$HOME/.virtualenvs
ENV VIRTUALENVWRAPPER_PYTHON=$HOME/.virtualenvs/venv/bin/python
ENV VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.virtualenvs/venv/bin/virtualenv

# Create a virtual environment for pycram-segmind
RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    mkvirtualenv --system-site-packages pycram-segmind -p python3.10 && \
    workon pycram-segmind && \
    pip install --upgrade pip

RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon pycram-segmind && \
    pip install -r pycram/requirements.txt && \
    pip install -U setuptools

RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon pycram-segmind && \
    pip install -r ripple_down_rules/requirements.txt && \
    pip install lark

RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon pycram-segmind && \
    cd $LIBSDIR/giskardpy && \
    pip install -r requirements.txt && pip install -e . && \
    pip install -U pycryptodomex && \
    cd $LIBSDIR/qpSWIFT/python && \
    sudo python3 setup.py install && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd $HOME/workspace/src && \
    git clone -b ros1-noetic-main https://github.com/SemRoCo/giskard_msgs.git && \
    cd $HOME/workspace && \
    rosdep update --rosdistro=$ROS_DISTRO && \
    sudo apt-get update && \
#    rosdep install --from-paths ./ -i -y --rosdistro ${ROS_DISTRO} && \
    catkin build && \
    source $HOME/workspace/devel/setup.bash

RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon pycram-segmind && \
    cd $LIBSDIR/Multiverse/Multiverse-Launch && \
    pip install -r requirements.txt

RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon pycram-segmind && \
    cd $LIBSDIR/Multiverse/Multiverse-Parser && \
    pip install -r requirements.txt && pip install -e .

RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon pycram-segmind && \
    cd $LIBSDIR/Multiverse/Multiverse-Launch/src/multiverse_connectors/multiverse_simulators_connector/src/mujoco_connector && \
    pip install -r requirements.txt && \
    cd $LIBSDIR/Multiverse/Multiverse-Launch/src/multiverse_connectors/multiverse_ros_connector/ros_ws/multiverse_ws && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd $HOME/workspace && \
    catkin clean -y && \
    catkin build

RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon pycram-segmind && \
    cd $LIBSDIR/Multiverse/Multiverse-Utilities && \
    pip install -r requirements.txt

# Install code-server
RUN sudo apt-get update && sudo apt-get install -y curl wget gnupg git bash sudo build-essential \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://code-server.dev/install.sh | sh
# Install desired extensions
# Set code-server user data dir manually if your container does this
#ENV USER_DATA_DIR=/.jbdevcontainer/data/code-server
ENV USER_DATA_DIR=$HOME/.local/share/code-server
# Create the data dir
RUN mkdir -p $USER_DATA_DIR
# Install extensions into the proper location
RUN code-server \
    --user-data-dir=$USER_DATA_DIR \
    --install-extension ms-python.python \
    && code-server \
    --user-data-dir=$USER_DATA_DIR \
    --install-extension Codeium.codeium \
    && code-server \
    --user-data-dir=$USER_DATA_DIR \
    --install-extension PKief.material-icon-theme

ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR

ENV LIBGL_ALWAYS_SOFTWARE=1

RUN sudo apt update && sudo apt upgrade -y

RUN echo "source ${HOME}/.virtualenvs/pycram-segmind/bin/activate" >> $HOME/.bashrc
RUN echo "source ${HOME}/workspace/devel/setup.bash" >> $HOME/.bashrc

COPY ./start-code-server.sh /usr/local/bin/start-code-server.sh
RUN sudo chmod +x /usr/local/bin/start-code-server.sh

FROM base as final
COPY ./entrypoint.sh /
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
