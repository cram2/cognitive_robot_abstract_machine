import subprocess
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class Repository:
    """
    Data representation of a Git repository to be included in the workspace.
    """

    url: str
    branch: str
    name: str
    cleanup_dirs: List[str] = field(default_factory=list)


class SystemDependencyManager:
    """
    Handles installation of system-level dependencies using apt.
    """

    def __init__(self):
        self.is_root = os.geteuid() == 0

    def install_packages(self, packages: List[str]):
        """
        Installs a list of packages via apt.
        """
        command = ["apt", "update"]
        if not self.is_root:
            command = ["sudo"] + command

        subprocess.run(command, check=True)

        command = ["apt", "install", "-y"] + packages
        if not self.is_root:
            command = ["sudo"] + command

        subprocess.run(command, check=True)


class GitManager:
    """
    Manages Git operations like cloning and pulling for repositories.
    """

    def __init__(self, workspace_src_path: str):
        self.workspace_src_path = workspace_src_path

    def setup_repository(self, repo: Repository):
        """
        Clones a repository if it doesn't exist, or pulls updates if it does.
        """
        repo_path = os.path.join(self.workspace_src_path, repo.name)

        if os.path.exists(repo_path):
            print(f"{bcolors.OKGREEN}Updating {repo.name}...{bcolors.ENDC}")
            subprocess.run(["git", "-C", repo_path, "pull"], check=True)
        else:
            print(f"{bcolors.OKGREEN}Cloning {repo.name}...{bcolors.ENDC}")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "-b",
                    repo.branch,
                    "--single-branch",
                    repo.url,
                    repo_path,
                ],
                check=True,
            )

        for sub_dir in repo.cleanup_dirs:
            full_sub_path = os.path.join(repo_path, sub_dir)
            if os.path.exists(full_sub_path):
                print(
                    f"{bcolors.WARNING}Removing unwanted subdirectory: {full_sub_path}{bcolors.ENDC}"
                )
                shutil.rmtree(full_sub_path)


class WorkspaceManager:
    """
    Coordinates the overall workspace setup process.
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.src_path = os.path.join(workspace_path, "src")
        self.dep_manager = SystemDependencyManager()
        self.git_manager = GitManager(self.src_path)

    def initialize_directories(self):
        """
        Creates the workspace directories if they do not exist.
        """
        os.makedirs(self.src_path, exist_ok=True)
        # Mark the workspace as safe for Git to avoid "dubious ownership" errors
        # which can cause setuptools_scm to fail during build.
        print(f"{bcolors.OKGREEN}Configuring Git safe directory...{bcolors.ENDC}")
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", "*"], check=True
        )

    def build_workspace(self):
        """
        Builds the ROS 2 workspace using colcon.
        """
        print(f"{bcolors.OKGREEN}Building workspace...{bcolors.ENDC}")
        # Source ROS Jazzy environment
        ros_setup = "/opt/ros/jazzy/setup.bash"
        command = f"source {ros_setup} && cd {self.workspace_path} && colcon build --merge-install"
        subprocess.run(["bash", "-c", command], check=True)

    def update_bashrc(self):
        """
        Adds the workspace sourcing to .bashrc if not already present.
        """
        setup_line = f"source {self.workspace_path}/install/setup.bash"
        bashrc_path = os.path.expanduser("~/.bashrc")

        with open(bashrc_path, "r") as f:
            content = f.read()

        if setup_line not in content:
            print(f"{bcolors.OKGREEN}Adding workspace to .bashrc...{bcolors.ENDC}")
            with open(bashrc_path, "a") as f:
                f.write(f"\n{setup_line}\n")


def main():
    """
    Main execution flow for setting up the ROS workspace.
    """
    workspace_path = os.environ.get("OVERLAY_WS", os.path.expanduser("~/ros2_ws"))
    manager = WorkspaceManager(workspace_path)

    # 1. System Dependencies
    packages = [
        "python3.12-venv",
        "ros-jazzy-xacro",
        "ros-jazzy-navigation2",
        "ros-jazzy-py-trees-ros",
        "python3-vcstool",
        "git",
        "ros-dev-tools",
        "default-jre",
        "graphviz",
        "graphviz-dev",
        "ros-jazzy-rclpy-message-converter",
        "pip",
        "python3-colcon-common-extensions",
        "ros-jazzy-compressed-image-transport",
        "ros-jazzy-image-transport",
        "ros-jazzy-image-transport-plugins",
        "git-lfs",
        "ros-jazzy-ament-cmake-auto",
        "ros-jazzy-ament-lint-auto",
        "ros-jazzy-ament-cmake-ros",
        "ros-jazzy-launch-testing-ament-cmake",
    ]
    manager.dep_manager.install_packages(packages)

    # 2. Setup Directories
    manager.initialize_directories()

    # 3. Repositories
    repos = [
        Repository("https://github.com/code-iai/iai_maps.git", "ros-jazzy", "iai_maps"),
        Repository(
            "https://github.com/code-iai/iai_robots.git", "ros-jazzy", "iai_robots"
        ),
        Repository(
            "https://github.com/code-iai/iai_pr2.git", "ros-jazzy-main", "iai_pr2"
        ),
        Repository(
            "https://github.com/code-iai/hsr_description.git",
            "ros2-jazzy",
            "hsr_description",
        ),
        Repository(
            "https://github.com/code-iai/iai_tracy.git",
            "ros2-jazzy",
            "iai_tracy",
            ["iai_tracy_bringup", "iai_tracy_ur"],
        ),
        Repository(
            "https://github.com/maltehue/ros2_robotiq_gripper.git",
            "iai_dualarm",
            "ros2_robotiq_gripper",
            [
                "robotiq_controllers",
                "robotiq_drivers",
                "robotiq_hardware_tests",
                "robotiq_driver",
            ],
        ),
        Repository(
            "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git",
            "jazzy",
            "Universal_Robots_ROS2_Description",
        ),
        Repository(
            "https://github.com/code-iai/stretch_ros2.git",
            "joint_velocity_interface",
            "stretch_ros2",
            ["stretch_funmap"],
        ),
        Repository(
            "https://github.com/realsenseai/realsense-ros.git",
            "ros2-master",
            "realsense-ros",
            [
                "realsense2_camera",
                "realsense2_camera_msgs",
                "realsense2_rgbd_plugin",
                "realsense2_ros_mqtt_bridge",
            ],
        ),
        Repository(
            "https://github.com/code-iai/iai_tiago_description.git",
            "ros2-main",
            "iai_tiago_description",
        ),
        Repository(
            "https://github.com/pal-robotics/pmb2_robot.git",
            "humble-devel",
            "pmb2_robot",
            ["pmb2_bringup", "pmb2_controller_configuration"],
        ),
        Repository(
            "https://github.com/pal-robotics/pal_gripper.git",
            "humble-devel",
            "pal_gripper",
            [
                "pal_grippper_controller_configuration",
                "pal_gripper_gazeboo",
                "pal_gripper_simulation",
                "pal_parallel_gripper_wrapper",
            ],
        ),
        Repository(
            "https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_msgs.git",
            "ros2_jazzy",
            "robokudo_msgs",
        ),
        Repository(
            "https://github.com/cram2/cram_ros2_packages.git",
            "main",
            "cram_ros2_packages",
        ),
    ]

    for repo in repos:
        manager.git_manager.setup_repository(repo)

    # 4. Build and Source
    manager.build_workspace()
    manager.update_bashrc()


if __name__ == "__main__":
    main()
