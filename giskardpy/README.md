# Giskardpy
Giskardpy is an open source library for implementing motion control frameworks.
It uses constraint and optimization based task space control to control the whole body of mobile manipulators.

This is a pure python library with the core functionality.
To use it with ROS you need the following repos, which use giskardpy to create an action server and implement ROS interfaces:
 - **ROS1**: https://github.com/SemRoCo/giskardpy_ros/tree/ros1-noetic-main
 - **ROS2**: https://github.com/SemRoCo/giskardpy_ros/tree/ros2-jazzy-main

## Installation instructions for Ubuntu (tested on 20.04 and 24.04)

### (Optional) create a virtual environment using virtualenvwrapper
```
sudo apt install virtualenvwrapper
echo "export WORKON_HOME=~/venvs" >> ~/.bashrc
echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
mkdir -p $WORKON_HOME

# --system-site-packages is only required if you are using ROS
mkvirtualenv giskardpy --system-site-packages
```
To use it do:
```
workon giskardpy
```

### Build Giskardpy
Switch to your venv, if you use one.
```
workon giskardpy
```
Choose a place where you want to build giskardpy and clone it. This should NOT be in a ROS workspace.
```
mkdir -p ~/libs && cd ~/libs
git clone -b giskard_library https://github.com/SemRoCo/giskardpy.git
cd giskardpy
```
Install Giskardpy, `-e` is optional but prevents you from having to rebuild every time the code changes.
```
pip3 install -e .                           
```

### Tutorials
https://github.com/SemRoCo/giskardpy/wiki

### How to cite
```
@phdthesis{stelter25giskard,
	author = {Simon Stelter},
	title = {A Robot-Agnostic Kinematic Control Framework: Task Composition via Motion Statecharts and Linear Model Predictive Control},
	year = {2025},
	doi = {10.26092/elib/3743},	
}
```
