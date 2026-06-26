"""
RoboKudo - A behavior tree based perception framework.

This package provides a flexible framework for building robotic perception
systems using behavior trees. It includes:

* Behavior tree based pipeline execution
* Modular annotator system for perception tasks
* Visualization tools for debugging and monitoring
* Support for ROS integration
* Common data structures for perception data
"""

import sys
from pathlib import Path

def _get_version():
    version_file = Path(__file__).resolve().parent.parent.parent.parent / "VERSION"
    with open(version_file) as f:
        return f.read().strip()

__version__: str = _get_version()
