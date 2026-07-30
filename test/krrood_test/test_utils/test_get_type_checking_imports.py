import os
import types
from pathlib import Path
from collections import defaultdict
import sys
import krrood.utils as krrood_utils
from krrood.utils import (
    get_scope_from_imports,
    _warn_about_unresolvable_type_checking_import_once,
)


def test_get_type_checking_imports(tmp_path):
    source = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import os
    from collections import defaultdict
    import math as m
    from pathlib import Path as P
"""
    test_file = tmp_path / "test_imports.py"
    test_file.write_text(source)

    # Currently it returns a list of libcst nodes, but we want it to return a scope
    scope = get_scope_from_imports(str(test_file))

    assert isinstance(scope, dict)
    assert scope["os"] == os
    assert scope["defaultdict"] == defaultdict
    assert "m" in scope
    import math

    assert scope["m"] == math
    assert scope["P"] == Path


def test_get_type_checking_imports_typing_extensions(tmp_path):
    source = """
import typing_extensions
if typing_extensions.TYPE_CHECKING:
    import sys
"""
    test_file = tmp_path / "test_imports_te.py"
    test_file.write_text(source)

    scope = get_scope_from_imports(str(test_file))

    assert isinstance(scope, dict)
    assert scope["sys"] == sys


def test_get_type_checking_imports_already_in_sys_modules(tmp_path):
    # Ensure a module is in sys.modules
    import json

    source = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import json
"""
    test_file = tmp_path / "test_imports_sys.py"
    test_file.write_text(source)

    scope = get_scope_from_imports(str(test_file))
    assert scope["json"] == json


def test_repeated_unresolvable_type_checking_import_warns_only_once(
    tmp_path, monkeypatch
):
    """
    A ``TYPE_CHECKING`` import that keeps failing because its module is still partially
    initialized (an active circular import) must not flood the log: the same ``(module,
    name, file)`` failure repeats identically on every attempt, so only the first one
    should be logged.

    This reproduces what happens when many dataclasses across a codebase each need to
    resolve the same ``TYPE_CHECKING``-only name from a module still mid-import: every
    one of those attempts raises an identical, already self-diagnosing
    ``AttributeError`` and previously logged its own warning, flooding the log with
    thousands of duplicate lines for a single, known-transient cause.
    """
    _warn_about_unresolvable_type_checking_import_once.cache_clear()
    provider_module = "test.krrood_test.dataset.latebound_annotation_type"

    warning_calls = []
    monkeypatch.setattr(
        krrood_utils.logger, "warning", lambda message: warning_calls.append(message)
    )

    source = f"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from {provider_module} import LateBoundAnnotationType
"""
    test_file = tmp_path / "test_repeated_poisoned_import.py"
    test_file.write_text(source)

    saved_module = sys.modules.pop(provider_module, None)
    sys.modules[provider_module] = types.ModuleType(provider_module)
    try:
        for _ in range(5):
            scope = get_scope_from_imports(str(test_file))
            assert "LateBoundAnnotationType" not in scope
    finally:
        if saved_module is not None:
            sys.modules[provider_module] = saved_module
        else:
            sys.modules.pop(provider_module, None)

    assert len(warning_calls) == 1
