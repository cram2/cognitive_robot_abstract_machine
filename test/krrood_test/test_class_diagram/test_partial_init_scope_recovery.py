import inspect
import sys
import types

from typing_extensions import Optional

from krrood.class_diagrams import utils as class_diagram_utils
from krrood.class_diagrams.exceptions import CouldNotResolveType
from krrood.class_diagrams.utils import get_type_hints_of_object

from ..dataset.latebound_annotation_type import LateBoundAnnotationType
from ..dataset.latebound_annotation_secondary_type import (
    LateBoundAnnotationSecondaryType,
)
from ..dataset.latebound_annotation_owner import OwnerWithLateBoundAnnotation


def test_type_resolution_recovers_when_scope_cache_poisoned_during_partial_init():
    """
    A ``TYPE_CHECKING`` annotation must still resolve after its target module was
    partially initialised when the owner's import scope was first cached.

    The class-diagram resolver caches each module's import scope. When that scope is
    first built while the annotated type's module is only partially initialised (a
    circular import during module load), the ``TYPE_CHECKING`` import cannot be resolved
    and the cached scope permanently lacks the name. Resolution must recover by
    recomputing the scope once imports have completed instead of raising
    ``CouldNotResolveType``.
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


def test_recovered_scope_is_healed_and_not_rebuilt_on_every_lookup(monkeypatch):
    """
    Once a ``TYPE_CHECKING``-only name recovers from a circular-import cache-poisoning
    event, later lookups of that same name must reuse the recovered scope instead of
    repeating the expensive, uncached AST walk and real-import fallback on every single
    call.

    This reproduces what happens when ``semantic_digital_twin.world`` is imported:
    dozens of dataclass fields across the codebase reference
    ``World``/``GenericSemanticAnnotation`` under ``TYPE_CHECKING``, and the very first
    scope build for their owning module happens while ``world`` is still mid-import (a
    legitimate, unavoidable circular reference). The permanent ``lru_cache`` entry never
    gets corrected once ``world`` finishes loading, so every one of those fields falls
    back to :func:`~krrood.utils.get_scope_from_imports` again, uncached, re-parsing and
    re-importing the whole file from scratch thousands of times over process startup.
    """
    owner_source = inspect.getsourcefile(OwnerWithLateBoundAnnotation)
    provider_module = "test.krrood_test.dataset.latebound_annotation_type"

    class_diagram_utils._scope_from_imports_by_mtime.cache_clear()
    class_diagram_utils._fallback_scope_for_import_generation.cache_clear()
    saved_module = sys.modules.pop(provider_module, None)
    sys.modules[provider_module] = types.ModuleType(provider_module)
    try:
        poisoned_scope = class_diagram_utils._cached_scope_from_imports(owner_source)
        assert "LateBoundAnnotationType" not in poisoned_scope
    finally:
        if saved_module is not None:
            sys.modules[provider_module] = saved_module
        else:
            sys.modules.pop(provider_module, None)

    call_count = 0
    original_get_scope_from_imports = class_diagram_utils.get_scope_from_imports

    def counting_get_scope_from_imports(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_get_scope_from_imports(*args, **kwargs)

    monkeypatch.setattr(
        class_diagram_utils, "get_scope_from_imports", counting_get_scope_from_imports
    )

    for _ in range(5):
        resolved = (
            class_diagram_utils.get_object_by_name_from_another_object_in_same_module(
                "LateBoundAnnotationType", OwnerWithLateBoundAnnotation
            )
        )
        assert resolved is LateBoundAnnotationType

    assert call_count == 1


def test_recovering_one_poisoned_name_heals_other_poisoned_names_in_same_scope():
    """
    Recovering one ``TYPE_CHECKING``-only name must heal every other name poisoned by
    the same circular-import event in that module's cached scope, not just the one that
    was looked up.

    ``world_entity.py`` has both ``World`` and ``GenericSemanticAnnotation`` poisoned
    together by the same partially initialised ``world`` module. Fixing only the queried
    name would leave the other one paying the same uncached-recompute cost on its own
    first lookup.
    """
    owner_source = inspect.getsourcefile(OwnerWithLateBoundAnnotation)
    primary_provider = "test.krrood_test.dataset.latebound_annotation_type"
    secondary_provider = "test.krrood_test.dataset.latebound_annotation_secondary_type"

    class_diagram_utils._scope_from_imports_by_mtime.cache_clear()
    class_diagram_utils._fallback_scope_for_import_generation.cache_clear()
    saved_primary = sys.modules.pop(primary_provider, None)
    saved_secondary = sys.modules.pop(secondary_provider, None)
    sys.modules[primary_provider] = types.ModuleType(primary_provider)
    sys.modules[secondary_provider] = types.ModuleType(secondary_provider)
    try:
        poisoned_scope = class_diagram_utils._cached_scope_from_imports(owner_source)
        assert "LateBoundAnnotationType" not in poisoned_scope
        assert "LateBoundAnnotationSecondaryType" not in poisoned_scope
    finally:
        for provider, saved in (
            (primary_provider, saved_primary),
            (secondary_provider, saved_secondary),
        ):
            if saved is not None:
                sys.modules[provider] = saved
            else:
                sys.modules.pop(provider, None)

    resolved = (
        class_diagram_utils.get_object_by_name_from_another_object_in_same_module(
            "LateBoundAnnotationType", OwnerWithLateBoundAnnotation
        )
    )
    assert resolved is LateBoundAnnotationType

    healed_scope = class_diagram_utils._cached_scope_from_imports(owner_source)
    assert (
        healed_scope["LateBoundAnnotationSecondaryType"]
        is LateBoundAnnotationSecondaryType
    )


def test_fallback_recompute_is_shared_across_lookups_within_the_same_import_generation(
    monkeypatch,
):
    """
    While a dependency stays mid-import (an unchanged "import generation"), repeated
    failed lookups of the same poisoned name must share a single uncached recompute
    instead of each re-parsing and re-importing the file from scratch.

    This is the case the healing fix (see
    ``test_recovered_scope_is_healed_and_not_rebuilt_on_every_lookup``) cannot cover: many classes
    across a codebase needing a name from a module that is *still* mid-import (never resolves
    during the window being measured). Reproduced live: importing ``semantic_digital_twin.world``
    triggers 2154 uncached recomputes of just 8 files (805 alone for a single file), accounting for
    most of an 8.8s import. Only once the import generation actually advances should a fresh
    recompute run again.
    """
    owner_source = inspect.getsourcefile(OwnerWithLateBoundAnnotation)
    provider_module = "test.krrood_test.dataset.latebound_annotation_type"

    class_diagram_utils._scope_from_imports_by_mtime.cache_clear()
    class_diagram_utils._fallback_scope_for_import_generation.cache_clear()
    saved_module = sys.modules.pop(provider_module, None)
    sys.modules[provider_module] = types.ModuleType(provider_module)
    try:
        poisoned_scope = class_diagram_utils._cached_scope_from_imports(owner_source)
        assert "LateBoundAnnotationType" not in poisoned_scope

        call_count = 0
        original_get_scope_from_imports = class_diagram_utils.get_scope_from_imports

        def counting_get_scope_from_imports(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_get_scope_from_imports(*args, **kwargs)

        monkeypatch.setattr(
            class_diagram_utils,
            "get_scope_from_imports",
            counting_get_scope_from_imports,
        )
        monkeypatch.setattr(
            class_diagram_utils,
            "_count_of_modules_that_finished_initializing",
            lambda: 100,
        )

        # The provider module stays poisoned for every one of these lookups: each must fail, but
        # they all share the same import generation, so only the first should actually recompute.
        for _ in range(5):
            try:
                class_diagram_utils.get_object_by_name_from_another_object_in_same_module(
                    "LateBoundAnnotationType", OwnerWithLateBoundAnnotation
                )
                assert (
                    False
                ), "resolution should still fail while the provider stays poisoned"
            except CouldNotResolveType:
                pass

        assert call_count == 1

        # Advancing the generation (something elsewhere finished initializing) makes it worth
        # trying again, even though this particular lookup still fails.
        monkeypatch.setattr(
            class_diagram_utils,
            "_count_of_modules_that_finished_initializing",
            lambda: 101,
        )
        try:
            class_diagram_utils.get_object_by_name_from_another_object_in_same_module(
                "LateBoundAnnotationType", OwnerWithLateBoundAnnotation
            )
            assert (
                False
            ), "resolution should still fail while the provider stays poisoned"
        except CouldNotResolveType:
            pass

        assert call_count == 2
    finally:
        if saved_module is not None:
            sys.modules[provider_module] = saved_module
        else:
            sys.modules.pop(provider_module, None)
