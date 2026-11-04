from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType

import pytest

import probabilistic_model
from krrood.exceptions import DataclassException
from probabilistic_model.exceptions import ProbabilisticModelException

# %% exceptions the package defines


def exceptions_modules() -> list[ModuleType]:
    """
    :return: Every ``exceptions`` module of :mod:`probabilistic_model`.
    """
    package_directory = Path(probabilistic_model.__file__).parent
    return [
        importlib.import_module(
            ".".join(
                (probabilistic_model.__name__,)
                + path.relative_to(package_directory).with_suffix("").parts
            )
        )
        for path in sorted(package_directory.rglob("exceptions.py"))
    ]


def exception_classes_defined_in(module: ModuleType) -> list[type[BaseException]]:
    """
    :param module: The module to read.
    :return: The exception classes the module defines itself.
    """
    return [
        class_
        for _, class_ in inspect.getmembers(module, inspect.isclass)
        if issubclass(class_, BaseException) and class_.__module__ == module.__name__
    ]


# %% common base class


def test_probabilistic_model_exception_is_a_dataclass_exception():
    assert issubclass(ProbabilisticModelException, DataclassException)


@pytest.mark.parametrize(
    "exception_class",
    [
        exception_class
        for module in exceptions_modules()
        for exception_class in exception_classes_defined_in(module)
        if exception_class is not ProbabilisticModelException
    ],
    ids=lambda exception_class: exception_class.__qualname__,
)
def test_every_exception_of_the_package_is_a_probabilistic_model_exception(
    exception_class: type[BaseException],
):
    assert issubclass(exception_class, ProbabilisticModelException)
