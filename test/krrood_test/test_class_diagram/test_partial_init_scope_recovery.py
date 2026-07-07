import inspect
import sys
import types

from typing_extensions import Optional

from krrood.class_diagrams import utils as class_diagram_utils
from krrood.class_diagrams.utils import get_type_hints_of_object

from ..dataset.latebound_annotation_type import LateBoundAnnotationType
from ..dataset.latebound_annotation_owner import OwnerWithLateBoundAnnotation


def test_type_resolution_recovers_when_scope_cache_poisoned_during_partial_init():
    """A ``TYPE_CHECKING`` annotation must still resolve after its target module was partially
    initialised when the owner's import scope was first cached.

    The class-diagram resolver caches each module's import scope. When that scope is first built
    while the annotated type's module is only partially initialised (a circular import during module
    load), the ``TYPE_CHECKING`` import cannot be resolved and the cached scope permanently lacks the
    name. Resolution must recover by recomputing the scope once imports have completed instead of
    raising ``CouldNotResolveType``.
    """
    owner_source = inspect.getsourcefile(OwnerWithLateBoundAnnotation)
    provider_module = "test.krrood_test.dataset.latebound_annotation_type"

    # Build the import-scope cache while the provider module is unavailable, mirroring the provider
    # being only partially initialised during a circular import.
    class_diagram_utils._scope_from_imports_by_mtime.cache_clear()
    saved_module = sys.modules.pop(provider_module, None)
    # A partially initialised module: present in sys.modules but missing the type attribute yet.
    sys.modules[provider_module] = types.ModuleType(provider_module)
    try:
        poisoned_scope = class_diagram_utils._cached_scope_from_imports(owner_source)
        assert "LateBoundAnnotationType" not in poisoned_scope
    finally:
        if saved_module is not None:
            sys.modules[provider_module] = saved_module
        else:
            sys.modules.pop(provider_module, None)

    # The cached scope is still poisoned, but the provider module is importable again: resolution
    # must recompute the scope and recover the type.
    get_type_hints_of_object.cache_clear()
    resolved_type_hints = get_type_hints_of_object(OwnerWithLateBoundAnnotation)

    assert resolved_type_hints["value"] == Optional[LateBoundAnnotationType]
