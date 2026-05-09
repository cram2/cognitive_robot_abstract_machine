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
    _stale_files_for_package,
    ensure_role_mixins_current_for_pytest,
    generate_role_mixins_for_package,
    main,
)


# ── helpers ───────────────────────────────────────────────────────────────────

_GENERATE_FN = "krrood.generate_role_mixins.generate_role_mixins_for_package"
_STALE_FN = "krrood.generate_role_mixins._stale_files_for_package"


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

        execv_calls: list = []
        monkeypatch.setattr("os.execv", lambda exe, args: execv_calls.append((exe, args)))

        ensure_role_mixins_current_for_pytest(["mypkg"])

        assert any("--check" in c for c in run_calls)
        assert len(execv_calls) == 1
        _, args = execv_calls[0]
        assert "-m" in args and "pytest" in args

    def test_decrements_rerun_counter(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: MagicMock(returncode=1 if "--check" in cmd else 0),
        )
        monkeypatch.setenv("KRROOD_PYTEST_RERUN_COUNT", "2")
        monkeypatch.setattr("os.execv", lambda *a: None)

        import os

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
