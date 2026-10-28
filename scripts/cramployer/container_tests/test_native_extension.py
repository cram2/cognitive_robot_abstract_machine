"""
Check native extension availability with host source mounted into the image.
"""

from __future__ import annotations

# %% Imports
from importlib.machinery import ExtensionFileLoader

import random_events_lib


# %% Native extension availability
def test_native_module_is_a_compiled_extension() -> None:
    """
    Load native code rather than the source directory's namespace package.
    """
    assert isinstance(random_events_lib.__loader__, ExtensionFileLoader)


def test_native_module_constructs_the_real_line() -> None:
    """
    Expose the interval constructor required during variable imports.
    """
    assert isinstance(random_events_lib.reals(), random_events_lib.Interval)
