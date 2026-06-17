from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class DatabaseNotAvailableError(DataclassException):
    """
    Exception raised when a database is not available.
    """

    def error_message(self) -> str:
        return f"Database not available. Check environment variable SEMANTIC_DIGITAL_TWIN_DATABASE_URI."

    def suggest_correction(self) -> str:
        return ""
