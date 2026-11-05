# RoboKudo

RoboKudo is an open source framework for robot perception for ROS2.

## Installation instructions for Ubuntu (tested on 24.04)

The RoboKudo consists of two parts: The core `robokudo` Python package and related ROS2 packages.
A key ROS2 package is `robokudo_ros`, which provides entrypoint scripts to start RoboKudo or launch query-related
tooling.

To install RoboKudo,
first [install the main CRAM stack according to it's README](https://github.com/cram2/cognitive_robot_abstract_machine/?tab=readme-ov-file#installation).
Things to note:

* Set up the optional ROS workspace according to
  the [CRAM README](https://github.com/cram2/cognitive_robot_abstract_machine/tree/main#optional-setup-your-ros-workspace)
* Make sure to always have ROS in your environment by executing `source /opt/ros/jazzy/setup,bash` in every new shell or
  adding the line to your `~/.bashrc`
* Use UV instead of poetry. Make sure to `workon cram-env` before calling uv.
* To install `robokudo_ros`, follow the ROS Workspace setup step in the CRAM README mentioned above. The `robokudo_ros` repository is located at
  `https://github.com/cram2/cram_ros2_packages`
* If you want to use the `--symlink-install` option you have to rebuild your workspace as follows:
  ```bash
  cd $OVERLAY_WS
  
  # Remove old builds
  rm -r build/ log/ install/  
  
  # Use "setuptools<80" for compatibility with "--symlink-install"
  pip install "setuptools<80" 
   
  # Use "python3 -m colcon" so that ROS scripts use the cram-env interpreter over the global interpreter
  python3 -m colcon build --symlink-install --merge-install
  ```
  symlink-install installations are recommended if you want to develop extra RoboKudo ROS2 packages and want to change
  code (e.g. Analysis Engines, World Descriptors) without rebuilding.

### Optional requirements

Annotators built on large machine learning libraries need extra packages that a plain `robokudo` install leaves out,
so users who do not need them are spared the download. They are bundled in the `dev` extra:

| Package       | Annotators that need it                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------------------------|
| `torch`       | `CLIPAnnotator` (`robokudo.annotators.clip_annotator`), `YoloAnnotator`, `HumanAndPoseAnnotator`                       |
| `ultralytics` | `YoloAnnotator` (`robokudo.annotators.yolo_annotator`), `HumanAndPoseAnnotator` (`robokudo.annotators.human_and_pose`) |

Install them together with RoboKudo:

```bash
# From the root of the CRAM workspace
uv pip install -e "./robokudo[dev]"

# Or for a standalone RoboKudo installation
pip install "robokudo[dev]"
```

#### FoundationPose pose estimation

The `FoundationPoseAnnotator` (`robokudo.annotators.foundation_pose`) estimates and tracks 6D object poses with
`FoundationPoseEstimator`, which runs on an NVIDIA GPU through the ROS package `robokudo_foundation_pose_tensorrt` of
`cram_ros2_packages`. The ROS workspace setup builds that package without its CUDA extension; build the extension
from the activated CRAM environment:

```bash
sudo apt install -y nvidia-cuda-toolkit g++-12
python3 -m pip install tensorrt-cu12==10.13.3.9
cd $OVERLAY_WS
python3 -m colcon build --merge-install --packages-select robokudo_foundation_pose_tensorrt
```

The meshes of the examples are in the ROS package `robokudo_cad_data`. See the README of
`robokudo_foundation_pose_tensorrt` for the TensorRT engines and their cache.

### PyCharm setup

- Install PyCharm
- Open a terminal, source your ROS workspace and then start PyCharm from that terminal
- In PyCharm:
    - Open the folder where you've cloned the CRAM repository
    - Open the ROS Workspace folder as well and choose the 'Attach' option
    - Set your `cram-env` virtualenv as the Project Interpreter. File → Settings → Python → Interpreter
    - In the left pane, open your ROS workspace, select `build`, `install` and `log`, right-click them and 'Mark
      directory as' → 'Excluded'
    - In the left pane, open your ROS workspace and navigate to the `robokudo_ros` folder. Go to `robokudo_ros/scripts`
      and right-click `main.py` and choose 'Debug' to create your first run config.

#### Troubleshooting

- If your RoboKudo ROS packages cannot be resolved in PyCharm but the actual execution works, check the correct setup of
  the Project Dependencies. To do this, go to File → Settings → Project → Project Dependencies. Make sure that under
  `ros2_ws`, the checkbox next to `cognitive_robot_abstract_machine` is ticked. Also check the other way (`ros2_ws` in
  `cognitive_robot_abstract_machine` is ticked).

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
