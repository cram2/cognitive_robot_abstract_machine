"""
Tests for krrood.generate_role_mixins — the offline role-mixin generation CLI.

Unit tests cover argument parsing and dispatch via monkeypatching; they never
touch the filesystem.  The integration tests call real transformation logic
in dry-run or check mode against the krrood test-dataset package.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from krrood.generate_role_mixins import (
    _are_semantically_equal,
    _format_source,
    _stale_files_for_package,
    ensure_role_mixins_current_for_pytest,
    generate_role_mixins_for_package,
    main,
)
from krrood.patterns.code_generation.generated_code_file_writer import has_class_definitions


# ── helpers ───────────────────────────────────────────────────────────────────

_GENERATE_FN = "krrood.generate_role_mixins.generate_role_mixins_for_package"
_STALE_FN = "krrood.generate_role_mixins._stale_files_for_package"


# ── _format_source ────────────────────────────────────────────────────────────


class TestFormatSource:
    def test_returns_string(self):
        result = _format_source("x = 1\n")
        assert isinstance(result, str)

    def test_removes_unused_import(self):
        src = "import os\nx = 1\n"
        result = _format_source(src)
        assert "import os" not in result

    def test_keeps_used_import(self):
        src = "import os\nx = os.path.join('a', 'b')\n"
        result = _format_source(src)
        assert "import os" in result

    def test_returns_original_on_syntax_error(self):
        bad = "def (broken syntax"
        result = _format_source(bad)
        assert result == bad

    def test_black_formats_long_line(self):
        # Black should wrap a very long string assignment
        src = 'x = "' + "a" * 200 + '"\n'
        result = _format_source(src)
        assert isinstance(result, str)


# ── _are_semantically_equal ───────────────────────────────────────────────────


class TestAreSemanticallEqual:
    def test_identical_sources_are_equal(self):
        src = "def foo():\n    return 1\n"
        assert _are_semantically_equal(src, src)

    def test_whitespace_difference_is_equal(self):
        a = "def foo():\n    return 1\n"
        b = "def foo():\n        return 1\n"  # different indentation
        assert _are_semantically_equal(a, b)

    def test_blank_line_difference_is_equal(self):
        a = "x = 1\ny = 2\n"
        b = "x = 1\n\n\ny = 2\n"
        assert _are_semantically_equal(a, b)

    def test_different_class_body_is_not_equal(self):
        a = "class Foo:\n    x: int\n"
        b = "class Foo:\n    x: int\n    y: str\n"
        assert not _are_semantically_equal(a, b)

    def test_new_import_is_not_equal(self):
        a = "import os\n"
        b = "import os\nimport sys\n"
        assert not _are_semantically_equal(a, b)

    def test_renamed_identifier_is_not_equal(self):
        a = "def foo(): pass\n"
        b = "def bar(): pass\n"
        assert not _are_semantically_equal(a, b)

    def test_syntax_error_falls_back_to_exact_match(self):
        bad = "def (broken syntax"
        assert _are_semantically_equal(bad, bad)
        assert not _are_semantically_equal(bad, "something else")


# ── main: generate mode ───────────────────────────────────────────────────────


class TestMainGenerate:
    def test_single_package_calls_generate_and_returns_0(self, monkeypatch):
        calls: list[tuple[str, bool]] = []
        monkeypatch.setattr(_GENERATE_FN, lambda name, write: calls.append((name, write)))

        assert main(["mypkg"]) == 0
        assert calls == [("mypkg", True)]

    def test_multiple_packages_all_processed(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(_GENERATE_FN, lambda name, write: calls.append(name))

        assert main(["pkg1", "pkg2", "pkg3"]) == 0
        assert calls == ["pkg1", "pkg2", "pkg3"]

    def test_dry_run_passes_write_false(self, monkeypatch):
        calls: list[tuple[str, bool]] = []
        monkeypatch.setattr(_GENERATE_FN, lambda name, write: calls.append((name, write)))

        assert main(["mypkg", "--dry-run"]) == 0
        assert calls == [("mypkg", False)]

    def test_single_failure_returns_1(self, monkeypatch):
        monkeypatch.setattr(
            _GENERATE_FN,
            lambda name, write: (_ for _ in ()).throw(ImportError(f"No module named '{name}'")),
        )
        assert main(["nonexistent_package"]) == 1

    def test_partial_failure_returns_1(self, monkeypatch):
        def maybe_fail(name: str, write: bool) -> None:
            if name == "bad":
                raise RuntimeError("generation failed")

        monkeypatch.setattr(_GENERATE_FN, maybe_fail)
        assert main(["good", "bad"]) == 1

    def test_all_packages_attempted_despite_earlier_failure(self, monkeypatch):
        calls: list[str] = []

        def fail_first(name: str, write: bool) -> None:
            calls.append(name)
            if name == "pkg1":
                raise RuntimeError("boom")

        monkeypatch.setattr(_GENERATE_FN, fail_first)
        result = main(["pkg1", "pkg2"])
        assert result == 1
        assert calls == ["pkg1", "pkg2"]

    def test_no_packages_prints_help_and_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0


# ── main: --check mode ────────────────────────────────────────────────────────


class TestMainCheck:
    def test_check_returns_0_when_no_stale_files(self, monkeypatch):
        monkeypatch.setattr(_STALE_FN, lambda name: [])
        assert main(["--check", "mypkg"]) == 0

    def test_check_returns_1_when_stale_files_exist(self, monkeypatch, tmp_path):
        stale = [tmp_path / "some_role_mixins.py"]
        monkeypatch.setattr(_STALE_FN, lambda name: stale)
        assert main(["--check", "mypkg"]) == 1

    def test_check_multiple_packages_all_current_returns_0(self, monkeypatch):
        monkeypatch.setattr(_STALE_FN, lambda name: [])
        assert main(["--check", "pkg1", "pkg2"]) == 0

    def test_check_one_stale_package_returns_1(self, monkeypatch, tmp_path):
        def stale_fn(name: str) -> list[Path]:
            return [tmp_path / "stale.py"] if name == "pkg2" else []

        monkeypatch.setattr(_STALE_FN, stale_fn)
        assert main(["--check", "pkg1", "pkg2"]) == 1

    def test_check_exception_during_check_returns_1(self, monkeypatch):
        monkeypatch.setattr(_STALE_FN, lambda name: (_ for _ in ()).throw(ImportError()))
        assert main(["--check", "bad_pkg"]) == 1


# ── ensure_role_mixins_current_for_pytest ─────────────────────────────────────


class TestEnsureRoleMixinsCurrentForPytest:
    def test_returns_immediately_when_check_passes(self, monkeypatch):
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr("subprocess.run", mock_run)
        ensure_role_mixins_current_for_pytest(["mypkg"])
        mock_run.assert_called_once()

    def test_regenerates_and_restarts_when_check_fails(self, monkeypatch):
        check_result = MagicMock(returncode=1)
        gen_result = MagicMock(returncode=0)
        run_calls: list = []

        def mock_run(cmd, **kwargs):
            run_calls.append(cmd)
            if "--check" in cmd:
                return check_result
            return gen_result

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setenv("KRROOD_PYTEST_RERUN_COUNT", "2")

        with pytest.raises(SystemExit):
            ensure_role_mixins_current_for_pytest(["mypkg"])

        assert any("--check" in c for c in run_calls)
        assert any("-m" in c and "pytest" in c for c in run_calls)

    def test_decrements_rerun_counter(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: MagicMock(returncode=1 if "--check" in cmd else 0),
        )
        monkeypatch.setenv("KRROOD_PYTEST_RERUN_COUNT", "2")

        import os

        with pytest.raises(SystemExit):
            ensure_role_mixins_current_for_pytest(["mypkg"])
        assert os.environ["KRROOD_PYTEST_RERUN_COUNT"] == "1"

    def test_raises_usage_error_when_counter_exhausted(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: MagicMock(returncode=1 if "--check" in cmd else 0),
        )
        monkeypatch.setenv("KRROOD_PYTEST_RERUN_COUNT", "0")

        import pytest as _pytest

        with _pytest.raises(_pytest.UsageError, match="still outdated"):
            ensure_role_mixins_current_for_pytest(["mypkg"])

    def test_raises_usage_error_when_generation_fails(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: MagicMock(returncode=1),
        )
        monkeypatch.setenv("KRROOD_PYTEST_RERUN_COUNT", "2")

        import pytest as _pytest

        with _pytest.raises(_pytest.UsageError, match="regeneration failed"):
            ensure_role_mixins_current_for_pytest(["mypkg"])


# ── integration: actual generation (dry-run / check) ─────────────────────────


class TestIntegration:
    def test_generate_dry_run_does_not_raise(self):
        """Full pipeline smoke test — no files written."""
        generate_role_mixins_for_package(
            "test.krrood_test.dataset.role_and_ontology", write=False
        )

    def test_check_mode_on_dataset_package_does_not_raise(self):
        """_stale_files_for_package must not raise on a valid package."""
        _stale_files_for_package("test.krrood_test.dataset.role_and_ontology")

    def test_invalid_package_raises_import_error(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            generate_role_mixins_for_package("krrood.__nonexistent_package__")


# ── _stale_files_for_package: empty-mixin staleness logic ────────────────────


def _make_stale_check_monkeypatches(monkeypatch, tmp_path, mixin_src, mod_src):
    """Wire up monkeypatches so _stale_files_for_package exercises just the
    mixin-staleness branch without importing a real package.

    _modules_with_roles and RoleTransformer are local imports inside
    _stale_files_for_package, so they are patched at their source modules.
    """
    from unittest.mock import MagicMock
    import types
    import krrood.generate_role_mixins as grm
    import krrood.patterns.role.helpers as helpers_mod
    import krrood.patterns.role.role_transformer as rt_mod

    module_path = tmp_path / "foo.py"
    module_path.write_text(mod_src)
    mixin_path = tmp_path / "role_mixins" / "foo_role_mixins.py"

    fake_module = types.ModuleType("fake_module")
    fake_module.__file__ = str(module_path)

    transformer = MagicMock()
    transformer.transform.return_value = {fake_module: (mod_src, mixin_src)}
    transformer.get_generated_file_path.return_value = mixin_path

    fake_transformer_cls = MagicMock(return_value=transformer)
    fake_transformer_cls.get_module_file_path = staticmethod(lambda m: module_path)

    monkeypatch.setattr(grm.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(grm, "classes_of_package", lambda pkg: [])
    monkeypatch.setattr(grm, "ClassDiagram", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(helpers_mod, "_modules_with_roles", lambda cd: [fake_module])
    monkeypatch.setattr(rt_mod, "RoleTransformer", fake_transformer_cls)

    return mixin_path


class TestStaleFilesEmptyMixinLogic:
    """Unit tests for the empty-mixin branch of _stale_files_for_package."""

    def test_missing_mixin_with_content_is_stale(self, monkeypatch, tmp_path):
        """A missing mixin file is stale when mixin content has class definitions."""
        mixin_src = "class FooMixin:\n    pass\n"
        mixin_path = _make_stale_check_monkeypatches(monkeypatch, tmp_path, mixin_src, "x = 1\n")
        stale = _stale_files_for_package("fake_pkg")
        assert mixin_path in stale

    def test_missing_mixin_without_content_is_not_stale(self, monkeypatch, tmp_path):
        """A missing mixin file is NOT stale when mixin content has no class definitions."""
        mixin_src = "from __future__ import annotations\n"
        mixin_path = _make_stale_check_monkeypatches(monkeypatch, tmp_path, mixin_src, "x = 1\n")
        stale = _stale_files_for_package("fake_pkg")
        assert mixin_path not in stale

    def test_existing_mixin_without_content_is_stale(self, monkeypatch, tmp_path):
        """An existing mixin file IS stale when new mixin content has no class definitions."""
        mixin_src = "from __future__ import annotations\n"
        mixin_path = _make_stale_check_monkeypatches(monkeypatch, tmp_path, mixin_src, "x = 1\n")
        mixin_path.parent.mkdir(parents=True, exist_ok=True)
        mixin_path.write_text("class OldMixin:\n    pass\n")
        stale = _stale_files_for_package("fake_pkg")
        assert mixin_path in stale


# ── error visibility: _format_source ────────────────────────────────────────────


class TestFormatSourceErrorVisibility:
    def test_logs_warning_when_formatting_fails(self, caplog, monkeypatch):
        """When ruff/black fails, a WARNING with traceback is logged."""
        import krrood.generate_role_mixins as grm

        monkeypatch.setattr(grm, "run_ruff_on_file", lambda p: (_ for _ in ()).throw(RuntimeError("ruff failed")))
        result = _format_source("x = 1\n")
        assert result == "x = 1\n"
        assert any(
            r.levelname == "WARNING"
            and "Formatting generated source failed" in r.message
            for r in caplog.records
        )

    def test_logs_warning_when_black_fails(self, caplog, monkeypatch):
        """When black fails after ruff succeeds, a WARNING is logged."""
        import krrood.generate_role_mixins as grm

        monkeypatch.setattr(grm, "run_ruff_on_file", lambda p: None)
        monkeypatch.setattr(grm, "run_black_on_file", lambda p: (_ for _ in ()).throw(RuntimeError("black failed")))
        result = _format_source("x = 1\n")
        assert result == "x = 1\n"
        assert any(
            r.levelname == "WARNING"
            and "Formatting generated source failed" in r.message
            for r in caplog.records
        )


# ── error visibility: main() traceback logging ──────────────────────────────────


class TestMainErrorVisibility:
    def test_generate_mode_logs_exception_with_traceback(self, caplog, monkeypatch):
        """main() must log full exception (with traceback) when generation fails."""
        monkeypatch.setattr(
            _GENERATE_FN,
            lambda name, write: (_ for _ in ()).throw(ValueError("something went wrong")),
        )
        main(["mypkg"])
        assert any(
            r.levelname == "ERROR"
            and "Failed: mypkg" in r.message
            and r.exc_info is not None
            for r in caplog.records
        )

    def test_check_mode_logs_exception_with_traceback(self, caplog, monkeypatch):
        """--check mode must log full exception when checking fails."""
        monkeypatch.setattr(
            _STALE_FN,
            lambda name: (_ for _ in ()).throw(RuntimeError("import exploded")),
        )
        main(["--check", "badpkg"])
        assert any(
            r.levelname == "ERROR"
            and "Failed to check badpkg" in r.message
            and r.exc_info is not None
            for r in caplog.records
        )


# ── error visibility: _stale_files_for_package per-module resilience ────────────


class TestStaleFilesPerModuleResilience:
    def test_continues_after_module_error(self, monkeypatch, tmp_path, caplog):
        """One failing module must not prevent other modules from being checked."""
        from unittest.mock import MagicMock
        import types
        import krrood.generate_role_mixins as grm
        import krrood.patterns.role.helpers as helpers_mod
        import krrood.patterns.role.role_transformer as rt_mod

        # Module 1: will fail
        bad_module = types.ModuleType("bad_module")
        bad_module.__file__ = str(tmp_path / "bad.py")
        bad_module.__name__ = "bad_module"

        # Module 2: will succeed and report a stale mixin
        good_module_path = tmp_path / "good.py"
        good_module_path.write_text("x = 1\n")
        good_module = types.ModuleType("good_module")
        good_module.__file__ = str(good_module_path)
        good_module.__name__ = "good_module"

        mixin_path = tmp_path / "role_mixins" / "good_role_mixins.py"
        good_mixin_src = "class FooMixin:\n    pass\n"

        good_transformer = MagicMock()
        good_transformer.transform.return_value = {good_module: ("x = 1\n", good_mixin_src)}
        good_transformer.get_generated_file_path.return_value = mixin_path

        def make_transformer(module):
            if module is bad_module:
                raise RuntimeError(f"cannot transform {module.__name__}")
            return good_transformer

        fake_transformer_cls = MagicMock(side_effect=make_transformer)
        fake_transformer_cls.get_module_file_path = staticmethod(lambda m: Path(m.__file__))

        monkeypatch.setattr(grm.importlib, "import_module", lambda name: MagicMock())
        monkeypatch.setattr(grm, "classes_of_package", lambda pkg: [])
        monkeypatch.setattr(grm, "ClassDiagram", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(
            helpers_mod,
            "_modules_with_roles",
            lambda cd: [bad_module, good_module],
        )
        monkeypatch.setattr(rt_mod, "RoleTransformer", fake_transformer_cls)

        stale = _stale_files_for_package("fake_pkg")
        assert mixin_path in stale
        assert any(
            r.levelname == "ERROR"
            and "bad_module" in r.message
            and r.exc_info is not None
            for r in caplog.records
        )


# ── error visibility: ensure_role_mixins stderr ─────────────────────────────────


class TestEnsureRoleMixinsStderr:
    def test_prints_check_stderr_on_failure(self, capsys, monkeypatch):
        """When check fails, captured stderr must be printed to the terminal."""
        check_result = MagicMock(returncode=1, stderr=b"Error: module not found\n")
        gen_result = MagicMock(returncode=0)

        def fake_run(cmd, **kwargs):
            if "--check" in cmd:
                return check_result
            return gen_result

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setenv("KRROOD_PYTEST_RERUN_COUNT", "2")

        import pytest as _pytest
        with _pytest.raises(SystemExit):
            ensure_role_mixins_current_for_pytest(["mypkg"])

        captured = capsys.readouterr()
        assert "Error: module not found" in captured.err
