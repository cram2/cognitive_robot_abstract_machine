from enum import Enum, auto


class PlayerStatus(Enum):
    """
    Represents the state of the episode player.
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