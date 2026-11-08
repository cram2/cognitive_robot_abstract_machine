"""
Resolves the field types of semdt's core exceptions in a process where ``rclpy`` cannot
be imported, as the ORM generation does, and prints the name of every exception
resolved.

Run as a subprocess by :mod:`test_exception_fields_without_ros`, whose process may have
ROS imported.
"""

# %% hiding ROS

import sys

# A None entry makes every import of the package raise ModuleNotFoundError.
sys.modules["rclpy"] = None

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.utils import classes_of_module

import semantic_digital_twin.exceptions

# %% resolving


def main() -> None:
    """
    Prints, one per line, the name of every core exception whose fields resolved.
    """
    diagram = ClassDiagram(classes_of_module(semantic_digital_twin.exceptions))
    for wrapped_class in diagram.wrapped_classes:
        wrapped_class.fields
        print(wrapped_class.clazz.__name__)


if __name__ == "__main__":
    main()
