from dataclasses import dataclass

from krrood.class_diagrams.exceptions import MockedClassInstantiationError


@dataclass
class MockedModule:
    """
    Base class for mocking modules.
    """


@dataclass
class MockedClass:
    """
    Base class for mocking classes. Cannot be instantiated (will raise an error).
    """

    def __new__(cls, *args, **kwargs):
        raise MockedClassInstantiationError()