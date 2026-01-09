class IntractableError(Exception):
    """
    Exception raised when an inference is intractable for a model.
    For instance, the mode of a non-deterministic model.
    """
    ...


class UndefinedOperationError(Exception):
    """
    Exception raised when an operation is not defined for a model.
    For instance, invoking the CDF of a model that contains symbolic variables.
    """
    ...