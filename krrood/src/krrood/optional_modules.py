from __future__ import annotations

from krrood.patterns.optional_module import Extra, OptionalModule

# %% krrood's extras


class KrroodExtra(Extra):
    """
    An optional-dependency group of krrood.
    """

    DEVELOPMENT = "dev"
    """
    The development tools, including matplotlib.
    """


# %% krrood's optional modules


class KrroodOptionalModule(OptionalModule):
    """
    A module krrood imports only inside the functions of the feature that needs it.
    """

    MATPLOTLIB = "matplotlib", KrroodExtra.DEVELOPMENT
    """
    Draws graphs and rule trees.
    """
