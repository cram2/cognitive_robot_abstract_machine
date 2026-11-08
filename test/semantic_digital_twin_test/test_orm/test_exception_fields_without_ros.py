"""
Tests that semdt's core exceptions describe their fields without ROS types.
"""

import subprocess
import sys
from pathlib import Path

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.utils import classes_of_module

import semantic_digital_twin.exceptions

# %% resolving without ROS


def test_core_exception_fields_resolve_without_ros():
    """
    Every field type of the core exceptions resolves where ``rclpy`` cannot be imported,
    so the ORM generation needs no ROS for them.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "exception_fields_without_ros.py"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == [
        wrapped_class.clazz.__name__
        for wrapped_class in ClassDiagram(
            classes_of_module(semantic_digital_twin.exceptions)
        ).wrapped_classes
    ]
