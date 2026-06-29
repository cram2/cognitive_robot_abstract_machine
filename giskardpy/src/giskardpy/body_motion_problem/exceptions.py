"""
Exceptions for giskardpy BMP physics model implementations.
"""


class MissingMotionStatechartError(Exception):
    """
    Raised when a physics model is asked to run without a configured MotionStatechart.
    """
