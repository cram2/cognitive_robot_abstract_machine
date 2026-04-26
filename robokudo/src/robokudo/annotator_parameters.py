"""
Global parameter definitions for RoboKudo annotators.

This module provides predefined global configuration parameters that are used
across different annotators in RoboKudo. These parameters control common
functionality like depth processing and visualization.
"""


class AnnotatorPredefinedParameters:
    """Global configuration parameters for annotators.

    This class defines predefined parameters that are shared across all annotators.
    These parameters control generic functionality and are not specific to any
    particular vision method. For annotator-specific parameters, use the
    Annotator.Parameters class of the respective annotator.
    """

    global_with_depth: bool = True
    """Whether to use depth image and derived data (e.g. point clouds)"""

    global_with_visualization: bool = True
    """Whether annotators should create visualizations of their results"""
