"""
Stable field names shared by recording storage and episode inspection.
"""

from enum import StrEnum


# %% bundle metadata
class TrajectoryField(StrEnum):
    """
    Key a recorded scene's trajectory carries its frame stamps under.
    """

    FRAME_TIMES = "at"
    """
    Elapsed capture time for each trajectory frame.
    """


class SceneField(StrEnum):
    """
    Shared metadata keys in the browser's scene bundle format.
    """

    PLAN_TREES = "planTrees"
    """
    Nested execution trees captured by the plan observer.
    """

    DETECTED_EVENTS = "detectedEvents"
    """
    Events observed during the recorded execution.
    """

    TASK = "task"
    """
    Human-readable description of the recorded task.
    """

    ROBOT_NAME = "robotName"
    """
    Display name assigned to the recorded robot.
    """

    ENVIRONMENT_NAME = "environmentName"
    """
    Display name assigned to the recorded environment.
    """
