"""Failures when live query knowledge is unavailable."""


class NoQuerySourceRegistered(Exception):
    """
    Raised when a query has neither an explicit source nor an attached world source.
    """

    def __init__(self) -> None:
        """
        Explain that this session has no registered source of queryable state.
        """
        super().__init__("no query source is registered on this bridge")
