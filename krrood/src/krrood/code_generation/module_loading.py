"""Dynamically load a Python source file as a fresh, uniquely-named module."""

from __future__ import annotations

import importlib.util
import sys
import types
import uuid


def load_module_from_path(path: str, module_name_prefix: str) -> types.ModuleType:
    """Load *path* as a fresh, uuid-suffixed module registered in ``sys.modules``.

    The uuid-suffixed name and ``sys.modules`` pre-registration ensure that Python's
    ``@dataclass`` annotation-resolution machinery (``dataclasses._is_type``) can look up
    the module's globals when resolving string annotations produced by
    ``from __future__ import annotations``.

    :param path: Absolute path to a ``.py`` file.
    :param module_name_prefix: Prefix for the generated, uuid-suffixed module name
        registered in ``sys.modules``.
    :return: The fully-executed module object.
    """
    module_name = f"{module_name_prefix}{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
