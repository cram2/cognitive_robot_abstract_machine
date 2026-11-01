from __future__ import annotations

import importlib.util
from enum import StrEnum

from krrood.exceptions import OptionalModuleNotInstalled

# %% krrood's extras


class KrroodExtra(StrEnum):
    """
    An optional-dependency group of krrood, as named in its ``pyproject.toml``.
    """

    DEVELOPMENT = "dev"
    """
    The development tools, including matplotlib.
    """


# %% optional modules


class OptionalModule(StrEnum):
    """
    A module krrood imports only inside the methods of the feature that needs it.

    The value is the module's import name.
    """

    MATPLOTLIB = "matplotlib"
    """
    Draws graphs and rule trees.
    """

    @property
    def extra(self) -> KrroodExtra:
        """
        The extra of krrood that declares this module.
        """
        match self:
            case OptionalModule.MATPLOTLIB:
                return KrroodExtra.DEVELOPMENT

    def ensure_installed(self) -> None:
        """
        Check that this module can be imported.

        :raises OptionalModuleNotInstalled: If the module is not installed.
        """
        if importlib.util.find_spec(self.value) is None:
            raise OptionalModuleNotInstalled(module=self)
