import itertools
import os

from setuptools import find_packages, setup

package_name = "robokudo_ros"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
# for dirpath, _, filenames in itertools.chain(os.walk("data"), os.walk("launch")):
#    full_paths = [os.path.join(dirpath, f) for f in filenames]
#    install_path = os.path.join("share", package_name, dirpath)
#    data_files.append((install_path, full_paths))

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Patrick Mania",
    maintainer_email="pmania@uni-bremen.de",
    description="RoboKudo ROS2 package",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "main = robokudo_ros.scripts.main:main",
        ],
    },
)
