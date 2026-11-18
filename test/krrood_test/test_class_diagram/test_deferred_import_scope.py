"""
A ``TYPE_CHECKING`` import never runs at runtime, so resolving a type hint must not load
a module nothing asked for.
"""

import inspect
import sys

from krrood.utils import get_scope_from_imports

from ..dataset.latebound_annotation_owner import OwnerWithLateBoundAnnotation

# %% deferring the imports


def test_deferred_type_checking_import_runs_only_when_its_name_is_looked_up():
    """
    The module a ``TYPE_CHECKING`` import names stays unloaded until a name it binds is
    looked up.
    """
    provider_module = "test.krrood_test.dataset.latebound_annotation_secondary_type"
    owner_source = inspect.getsourcefile(OwnerWithLateBoundAnnotation)
    saved_module = sys.modules.pop(provider_module, None)
    try:
        scope = get_scope_from_imports(
            file_path=owner_source, defers_type_checking_imports=True
        )
        loaded_before_lookup = provider_module in sys.modules

        bound_name = scope["LateBoundAnnotationSecondaryType"]
        bound_from = sys.modules[provider_module]
    finally:
        if saved_module is not None:
            sys.modules[provider_module] = saved_module

    assert not loaded_before_lookup
    assert bound_name is bound_from.LateBoundAnnotationSecondaryType
