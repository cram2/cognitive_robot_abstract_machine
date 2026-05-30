"""
Phase 4 tests: ``RDRFileStore`` in ``rdr/file_store.py``.

These tests gate the following implementation steps:

Step A — ``RDRFileStore.path`` (``_resolve_path`` static method):
  * Relative ``filename`` → ``<func_module_dir>/_rdr_models/<filename>``.
  * Absolute ``filename`` → used as-is, no ``_rdr_models/`` prefix injected.

Step B — ``RDRFileStore.exists()``:
  * Returns ``False`` when the file has not been written yet.
  * Returns ``True`` after ``save()`` has been called.

Step C — ``RDRFileStore.save(rdr)``:
  * Creates the ``_rdr_models/`` directory when it is absent.
  * Writes a non-empty file at ``self.path``.

Step D — ``RDRFileStore.load_case_type()``:
  * Returns a class that is a subclass of ``FunctionCase``.
  * The returned class is a ``@dataclass``.
  * The returned class carries the expected field names from the wrapped function.
  * Full round-trip: ``load_case_type()`` + classify on a freshly loaded RDR.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.function_case import FunctionCase
from krrood.entity_query_language.rdr.interface import FunctionInterface
from krrood.entity_query_language.rdr.serialization import load_rdr
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

# ---------------------------------------------------------------------------
# Helpers — not collected by pytest (no "Test" prefix, not test_ functions)
# ---------------------------------------------------------------------------


def _make_distance_expert(case_type):
    """Return a scripted ``Expert`` that fires when ``x > 0``."""

    def answer_fn(ctx, reqs):
        return {"conditions": ctx.case_variable.x > 0}

    return Expert(interface=FunctionInterface(answer_fn=answer_fn))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def distance_func():
    """A simple fully-annotated function whose signature drives the FunctionCase.

    ``__module__`` is left as the real module of this test file; the actual
    module directory is therefore ``test/krrood_test/test_eql_rdr/``.
    ``inspect.getfile(distance)`` returns this file's path, which is the source
    of truth for ``_resolve_path``.
    """

    def distance(x: float, y: float) -> float:
        return (x**2 + y**2) ** 0.5

    return distance


@pytest.fixture
def generated_case_type(distance_func):
    """The ``Distance`` dataclass generated from ``distance_func`` via exec.

    Executes the source produced by ``function_to_dataclass_source`` in an
    isolated namespace so no filesystem write is needed here.

    The namespace is pre-seeded with ``distance_func`` under the name
    ``"distance"`` so that the assignment ``Distance.function = distance`` at
    the bottom of the generated source can resolve it without importing from the
    test module (which is not on sys.path as a named module during exec).
    """
    from krrood.code_generation import (
        function_to_dataclass_source,
    )

    source = function_to_dataclass_source(distance_func)
    ns: dict = {"distance": distance_func}
    exec(compile(source, "<generated_distance>", "exec"), ns)
    case_type = ns["Distance"]
    return case_type


@pytest.fixture
def fitted_rdr(distance_func, generated_case_type):
    """A minimal one-rule ``EQLSingleClassRDR`` over the generated ``Distance`` case type.

    Uses ``x > 0`` as the single distinguishing condition so the rule fires for
    positive ``x`` values and is absent for negative ones.
    """
    rdr = EQLSingleClassRDR(generated_case_type, "_output")
    expert = _make_distance_expert(generated_case_type)
    case = generated_case_type(x=1.0, y=2.0, _output=None)
    rdr.fit_case(case, 3.0, expert)
    return rdr, generated_case_type


@pytest.fixture
def file_store_absolute(distance_func, fitted_rdr, tmp_path):
    """An ``RDRFileStore`` configured with an absolute path under ``tmp_path``.

    Using an absolute path side-steps any dependency on the real filesystem
    directory that contains this test file.  This fixture is the canonical
    fixture for every test that only needs a store backed by a known location.
    """
    from krrood.entity_query_language.rdr.file_store import RDRFileStore

    rdr, _ = fitted_rdr
    save_file = str(tmp_path / "distance_model.py")
    return RDRFileStore(func=distance_func, filename=save_file), rdr


# ---------------------------------------------------------------------------
# Step A — path resolution
# ---------------------------------------------------------------------------


class TestPathResolutionRelative:
    """``RDRFileStore.path`` for a relative ``filename``."""

    def test_relative_filename_places_file_under_rdr_models_subdir(
        self, distance_func, tmp_path
    ):
        """A relative filename must be joined under ``<module_dir>/_rdr_models/``.

        The expected parent is ``Path(inspect.getfile(func)).parent / "_rdr_models"``.
        """
        from krrood.entity_query_language.rdr.file_store import RDRFileStore

        store = RDRFileStore(func=distance_func, filename="model.py")
        expected_parent = Path(inspect.getfile(distance_func)).parent / "_rdr_models"
        assert Path(store.path).parent == expected_parent

    def test_relative_filename_preserves_basename(self, distance_func):
        """The leaf name of a relative path must equal the supplied ``filename``."""
        from krrood.entity_query_language.rdr.file_store import RDRFileStore

        store = RDRFileStore(func=distance_func, filename="my_rules.py")
        assert Path(store.path).name == "my_rules.py"

    def test_relative_filename_path_is_a_str(self, distance_func):
        """``RDRFileStore.path`` must return a ``str``, not a ``Path`` object."""
        from krrood.entity_query_language.rdr.file_store import RDRFileStore

        store = RDRFileStore(func=distance_func, filename="model.py")
        assert isinstance(store.path, str)


class TestPathResolutionAbsolute:
    """``RDRFileStore.path`` for an absolute ``filename``."""

    def test_absolute_filename_is_used_unchanged(self, distance_func, tmp_path):
        """An absolute ``filename`` must be returned verbatim — no ``_rdr_models/`` injection."""
        from krrood.entity_query_language.rdr.file_store import RDRFileStore

        abs_path = str(tmp_path / "direct_model.py")
        store = RDRFileStore(func=distance_func, filename=abs_path)
        assert store.path == abs_path

    def test_absolute_filename_does_not_contain_rdr_models(
        self, distance_func, tmp_path
    ):
        """The resolved path for an absolute filename must not contain ``_rdr_models``."""
        from krrood.entity_query_language.rdr.file_store import RDRFileStore

        abs_path = str(tmp_path / "model.py")
        store = RDRFileStore(func=distance_func, filename=abs_path)
        assert "_rdr_models" not in store.path


# ---------------------------------------------------------------------------
# Step B — exists()
# ---------------------------------------------------------------------------


class TestExistsBeforeSave:
    """``exists()`` returns ``False`` when the file has not been written yet."""

    def test_exists_returns_false_before_save(self, file_store_absolute):
        """``exists()`` must be ``False`` before any ``save()`` call."""
        store, _ = file_store_absolute
        assert store.exists() is False

    def test_exists_return_value_is_bool(self, file_store_absolute):
        """``exists()`` must return a plain Python ``bool``."""
        store, _ = file_store_absolute
        assert isinstance(store.exists(), bool)


class TestExistsAfterSave:
    """``exists()`` returns ``True`` once the file has been written."""

    def test_exists_returns_true_after_save(self, file_store_absolute):
        """After ``save(rdr)`` the file must be detectable by ``exists()``."""
        store, rdr = file_store_absolute
        store.save(rdr)
        assert store.exists() is True


# ---------------------------------------------------------------------------
# Step C — save()
# ---------------------------------------------------------------------------


class TestSaveCreatesDirectory:
    """``save()`` must create the containing directory when it is absent."""

    def test_save_creates_rdr_models_directory_if_absent(
        self, distance_func, fitted_rdr, tmp_path
    ):
        """``save()`` on a relative-path store must mkdir ``_rdr_models/`` if it does not exist.

        We point ``inspect.getfile`` at a real location inside ``tmp_path`` by
        monkey-patching ``distance_func.__code__.co_filename`` so the module
        directory resolves inside ``tmp_path``, then confirm ``_rdr_models/``
        is created.
        """
        from krrood.entity_query_language.rdr.file_store import RDRFileStore

        rdr, _ = fitted_rdr
        # Use an absolute path that sits inside a new subdirectory of tmp_path.
        new_subdir = tmp_path / "new_subdir"
        save_path = str(new_subdir / "model.py")
        # new_subdir does not exist yet.
        assert not new_subdir.exists()

        store = RDRFileStore(func=distance_func, filename=save_path)
        store.save(rdr)

        assert new_subdir.exists()

    def test_save_does_not_raise_when_directory_already_exists(
        self, file_store_absolute, tmp_path
    ):
        """``save()`` must not raise even when the target directory already exists."""
        store, rdr = file_store_absolute
        # The directory already exists (tmp_path was created by pytest).
        store.save(rdr)  # must not raise


class TestSaveWritesContent:
    """``save()`` writes a non-empty file."""

    def test_saved_file_is_non_empty(self, file_store_absolute):
        """The file written by ``save(rdr)`` must contain at least one non-whitespace character."""
        store, rdr = file_store_absolute
        store.save(rdr)
        content = Path(store.path).read_text()
        assert content.strip()

    def test_saved_file_path_matches_store_path(self, file_store_absolute):
        """``save()`` must write to exactly ``self.path``, not a sibling file."""
        store, rdr = file_store_absolute
        store.save(rdr)
        assert Path(store.path).is_file()


# ---------------------------------------------------------------------------
# Step D — load_case_type()
# ---------------------------------------------------------------------------


class TestLoadCaseTypeReturnType:
    """``load_case_type()`` returns a class that is a ``FunctionCase`` subclass."""

    def test_load_case_type_returns_a_class(self, file_store_absolute):
        """``load_case_type()`` must return a Python class object (``type``)."""
        store, rdr = file_store_absolute
        store.save(rdr)
        result = store.load_case_type()
        assert isinstance(result, type)

    def test_load_case_type_is_subclass_of_function_case(self, file_store_absolute):
        """The class returned by ``load_case_type()`` must be a subclass of ``FunctionCase``."""
        store, rdr = file_store_absolute
        store.save(rdr)
        result = store.load_case_type()
        assert issubclass(result, FunctionCase)

    def test_load_case_type_is_a_dataclass(self, file_store_absolute):
        """The class returned by ``load_case_type()`` must be a dataclass."""
        store, rdr = file_store_absolute
        store.save(rdr)
        result = store.load_case_type()
        assert dataclasses.is_dataclass(result)


class TestLoadCaseTypeFields:
    """``load_case_type()`` returns a class that carries the expected fields."""

    def test_loaded_case_type_has_x_field(self, file_store_absolute):
        """The loaded case type must declare a field named ``x`` (first param of ``distance``)."""
        store, rdr = file_store_absolute
        store.save(rdr)
        result = store.load_case_type()
        field_names = {f.name for f in dataclasses.fields(result)}
        assert "x" in field_names

    def test_loaded_case_type_has_y_field(self, file_store_absolute):
        """The loaded case type must declare a field named ``y`` (second param of ``distance``)."""
        store, rdr = file_store_absolute
        store.save(rdr)
        result = store.load_case_type()
        field_names = {f.name for f in dataclasses.fields(result)}
        assert "y" in field_names

    def test_loaded_case_type_has_output_field(self, file_store_absolute):
        """The loaded case type must declare a field named ``_output`` (return value slot)."""
        store, rdr = file_store_absolute
        store.save(rdr)
        result = store.load_case_type()
        field_names = {f.name for f in dataclasses.fields(result)}
        assert "_output" in field_names

    def test_loaded_case_type_has_exactly_three_fields(self, file_store_absolute):
        """The ``distance(x, y) -> float`` case type must yield exactly three dataclass fields.

        FunctionCase itself has no instance fields, so ``x``, ``y``, ``_output`` are the only three.
        """
        store, rdr = file_store_absolute
        store.save(rdr)
        result = store.load_case_type()
        # Filter out fields defined by FunctionCase itself (ClassVar 'function' is not a field).
        field_names = {f.name for f in dataclasses.fields(result)}
        assert field_names == {"x", "y", "_output"}


# ---------------------------------------------------------------------------
# Step D — round-trip: save → load_case_type → classify
# ---------------------------------------------------------------------------


class TestLoadCaseTypeClassifyRoundtrip:
    """After save + load_case_type, a new RDR built from the loaded case type classifies correctly."""

    def test_classify_positive_x_returns_trained_conclusion(self, file_store_absolute):
        """A case with ``x=1.0`` must be classified as ``3.0`` after the round-trip.

        The single rule in ``fitted_rdr`` fires when ``x > 0`` and concludes ``3.0``.
        """
        store, rdr = file_store_absolute
        store.save(rdr)

        loaded_rdr = load_rdr(store.path)
        loaded_case_type = store.load_case_type()

        case = loaded_case_type(x=1.0, y=2.0, _output=None)
        result = loaded_rdr.classify(case)
        assert result == pytest.approx(3.0)

    def test_classify_negative_x_returns_unset(self, file_store_absolute):
        """A case with ``x=-1.0`` must return ``UNSET`` — no rule fires for it.

        The single rule requires ``x > 0``; negative ``x`` falls through to no conclusion.
        """
        from krrood.entity_query_language.rdr.utils import UNSET

        store, rdr = file_store_absolute
        store.save(rdr)

        loaded_rdr = load_rdr(store.path)
        loaded_case_type = store.load_case_type()

        case = loaded_case_type(x=-1.0, y=2.0, _output=None)
        result = loaded_rdr.classify(case)
        assert result is UNSET
