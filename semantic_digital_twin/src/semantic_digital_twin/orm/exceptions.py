from dataclasses import dataclass

from krrood.utils import DataclassException


@dataclass
class DatabaseNotAvailableError(DataclassException):
    """
    Exception raised when a database is not available.
    """

    def __post_init__(self):
        self.message = f"Database not available. Check environment variable SEMANTIC_DIGITAL_TWIN_DATABASE_URI."
