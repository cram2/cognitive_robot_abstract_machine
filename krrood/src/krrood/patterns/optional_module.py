from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
from enum import StrEnum

from typing_extensions import Callable, ParamSpec, Self, TypeVar

from krrood.patterns.exceptions import OptionalModuleNotInstalled

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")

# %% extras


class Extra(StrEnum):
    """
    An optional-dependency group of a distribution, as named in its ``pyproject.toml``.

    Each distribution declares its extras in a subclass defined inside its own package.
    """

    @property
    def distribution(self) -> str:
        """
        The name of the distribution that declares this extra, as ``pip install`` takes
        it.
        """
        top_level_package = type(self).__module__.partition(".")[0]
        return importlib.metadata.packages_distributions()[top_level_package][0]


# %% optional modules


class OptionalModule(StrEnum):
    """
    A module that only one extra of a distribution installs.

    The distribution imports it only inside functions decorated with its member's
    :meth:`require`, or under ``TYPE_CHECKING``. Each distribution declares its optional
    modules in a subclass whose members are the module's import name and the
    :class:`Extra` that installs it.
    """

    extra: Extra
    """
    The extra that installs this module.
    """

    def __new__(cls, import_name: str, extra: Extra) -> Self:
        member = str.__new__(cls, import_name)
        member._value_ = import_name
        member.extra = extra
        return member

    def ensure_installed(self) -> None:
        """
        Check that this module can be imported.

        :raises OptionalModuleNotInstalled: If the module is not installed.
        """
        if importlib.util.find_spec(self.value) is None:
            raise OptionalModuleNotInstalled(module=self)

    def require(
        self, function: Callable[Parameters, Result]
    ) -> Callable[Parameters, Result]:
        """
        Decorate a function that imports this module, so that every call first checks
        that it is installed.

        :param function: The function that imports this module.
        :return: The function, raising :class:`OptionalModuleNotInstalled` when called
            while this module is not installed.
        """

        @functools.wraps(function)
        def checked_function(
            *arguments: Parameters.args, **keyword_arguments: Parameters.kwargs
        ) -> Result:
            self.ensure_installed()
            return function(*arguments, **keyword_arguments)

        return checked_function
