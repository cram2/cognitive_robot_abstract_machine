# RoboKudo
RoboKudo is an open source framework for robot perception for ROS2

## Installation instructions for Ubuntu (tested on 24.04)
Please follow the instructions in the main README of this repo to install the dependencies with poetry/uv.
During the installation, you might get an error related to graphviz. Check out the README of semantic_digital_twin for more information how to fix this and which package has to be installed.

For RoboKudo, you have to handle the virtual environment creation a bit differently though. See below.


### (Optional) create a virtual environment using virtualenvwrapper
You can use the same virtual environment as the one that CRAM uses and explained in [its main README](https://github.com/Sanic/cognitive_robot_abstract_machine/tree/robokudo?tab=readme-ov-file#installation).
However, please make sure that you create the virtualenv with --system-site-packages as RoboKudo still needs ROS:
```
mkvirtualenv cram-env --system-site-packages
```
or create a new one:
```
mkvirtualenv robokudo --system-site-packages
```

To use it, call:
```
workon robokudo
```

Using virtualenvwrapper is highly encouraged as the whole CRAM architecture uses it anyway.

### Install and use RoboKudo
- Clone the CRAM repository to your filesystem (if you haven't already). In this example, we'll use ~/libs: 
```
mkdir -p ~/libs && cd ~/libs
git clone https://github.com/sanic/cognitive_robot_abstract_machine.git
cd robokudo
git checkout robokudo
```
- Switch to your venv, if you use one.
```
workon cram-env
# or:
workon robokudo
```
- Install robokudo, `-e` is optional but prevents you from having to rebuild every time the code changes.
```
pip3 install -r requirements.txt
pip3 install -e .                           
```

### Tutorials
https://robokudo.ai.uni-bremen.de/

### How to cite
```
@inproceedings{mania2024robokudo,
	title={An Open and Flexible Robot Perception Framework for Mobile Manipulation Tasks},
	author={Mania, Patrick and Stelter, Simon and Kazhoyan, Gayane and Beetz, Michael},
	booktitle={2024 International Conference on Robotics and Automation (ICRA)},
	year={2024},
	url={https://ai.uni-bremen.de/papers/mania2024robokudo.pdf},
	note={},
	organization={IEEE}
}
```
