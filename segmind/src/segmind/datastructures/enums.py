from enum import Enum, auto


class PlayerStatus(Enum):
    """
    A class that represents the state of the episode player.
    """
    CREATED = auto()
    """
    The episode player is created.
    """
    PLAYING = auto()
    """
    The episode player is playing.
    """
    PAUSED = auto()
    """
    The episode player is paused.
    """
    STOPPED = auto()
    """
    The episode player is stopped.
    """


class DistanceFilter(Enum):
    MOVING_AVERAGE = auto()
    LOW_PASS = auto()


class MotionDetectionMethod(Enum):
    CONSISTENT_GRADIENT = auto()
    DISTANCE = auto()
