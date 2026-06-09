"""
Tests for on-demand model save: ``ExpertInterface.save()``, ``_make_save_magic``,
and the auto-injection of ``on_save`` by ``EQLSingleClassRDR.fit()``.
"""

import io
import os
import tempfile
import unittest

from krrood.entity_query_language.rdr.expert import ANSWER_NAME, Expert
from krrood.entity_query_language.rdr.interactive import IPythonInterface, Palette
from krrood.entity_query_language.rdr.interface import FunctionInterface, CaseContext, AnswerRequest
from krrood.entity_query_language.rdr.magics import SAVE_MAGIC, _make_save_magic
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.interface import CASE_VARIABLE_NAME

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()


def first(sp: Species) -> Animal:
    return next(a for a, t in zip(animals, targets) if t is sp)


# ---------------------------------------------------------------------------
# ExpertInterface.save() — base class shared method
# ---------------------------------------------------------------------------


class TestExpertInterfaceSaveMethod(unittest.TestCase):
    """``save()`` on the base class calls the callback when set and no-ops otherwise."""

    def test_save_calls_on_save_callback(self):
        called = []
        interface = FunctionInterface(answer_fn=lambda ctx, reqs: {})
        interface.on_save = lambda: called.append(True)
        interface.save()
        self.assertEqual(called, [True])

    def test_save_noop_when_on_save_is_none(self):
        interface = FunctionInterface(answer_fn=lambda ctx, reqs: {})
        self.assertIsNone(interface.on_save)
        interface.save()  # must not raise

    def test_save_calls_callback_multiple_times(self):
        count = [0]
        interface = FunctionInterface(answer_fn=lambda ctx, reqs: {})
        interface.on_save = lambda: count.__setitem__(0, count[0] + 1)
        interface.save()
        interface.save()
        self.assertEqual(count[0], 2)


# ---------------------------------------------------------------------------
# _make_save_magic — factory function
# ---------------------------------------------------------------------------


class TestMakeSaveMagic(unittest.TestCase):
    """The magic factory returns a callable that delegates to ``interface.save()``."""

    def _palette(self) -> Palette:
        return Palette(use_color=False)

    def test_magic_invokes_save_when_on_save_is_set(self):
        called = []
        interface = FunctionInterface(answer_fn=lambda ctx, reqs: {})
        interface.on_save = lambda: called.append(True)
        magic = _make_save_magic(interface, self._palette())
        magic("")
        self.assertEqual(called, [True])

    def test_magic_prints_hint_when_on_save_is_none(self):
        interface = FunctionInterface(answer_fn=lambda ctx, reqs: {})
        magic = _make_save_magic(interface, self._palette())
        buf = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf):
            magic("")
        self.assertIn("No save path", buf.getvalue())

    def test_magic_is_callable_without_error(self):
        interface = FunctionInterface(answer_fn=lambda ctx, reqs: {})
        magic = _make_save_magic(interface, self._palette())
        magic("ignored line argument")  # must not raise

    def test_magic_sees_late_injected_on_save(self):
        """The magic closes over the interface, so on_save injected after magic creation is visible."""
        called = []
        interface = FunctionInterface(answer_fn=lambda ctx, reqs: {})
        magic = _make_save_magic(interface, self._palette())
        # Inject on_save AFTER building the magic — simulates fit()'s late injection.
        interface.on_save = lambda: called.append(True)
        magic("")
        self.assertEqual(called, [True])


# ---------------------------------------------------------------------------
# EQLSingleClassRDR.fit() — auto-injection
# ---------------------------------------------------------------------------


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestFitAutoInjectsOnSave(unittest.TestCase):
    """``fit()`` injects ``on_save`` when ``save_path`` is set and the callback is absent."""

    def _simple_runner(self):
        def run(namespace, header):
            cv = namespace[CASE_VARIABLE_NAME]
            case = namespace["case_instance"]
            namespace[ANSWER_NAME] = cv.milk == True

        return run

    def test_fit_injects_on_save_when_save_path_set(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
        try:
            rdr = EQLSingleClassRDR(Animal, "species", save_path=path)
            interface = IPythonInterface(shell_runner=self._simple_runner())
            expert = Expert(interface=interface)
            self.assertIsNone(interface.on_save)
            rdr.fit([first(Species.mammal)], [Species.mammal], expert)
            self.assertIsNotNone(interface.on_save)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_fit_does_not_inject_when_save_path_is_none(self):
        rdr = EQLSingleClassRDR(Animal, "species")  # no save_path
        interface = IPythonInterface(shell_runner=self._simple_runner())
        expert = Expert(interface=interface)
        rdr.fit([first(Species.mammal)], [Species.mammal], expert)
        self.assertIsNone(interface.on_save)

    def test_fit_preserves_user_supplied_on_save(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
        try:
            called = []
            custom_callback = lambda: called.append("custom")
            rdr = EQLSingleClassRDR(Animal, "species", save_path=path)
            interface = IPythonInterface(shell_runner=self._simple_runner())
            interface.on_save = custom_callback
            expert = Expert(interface=interface)
            rdr.fit([first(Species.mammal)], [Species.mammal], expert)
            self.assertIs(interface.on_save, custom_callback)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_injected_on_save_writes_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
            os.unlink(path)  # remove so we can test creation
        try:
            rdr = EQLSingleClassRDR(Animal, "species", save_path=path)
            interface = IPythonInterface(shell_runner=self._simple_runner())
            expert = Expert(interface=interface)
            rdr.fit([first(Species.mammal)], [Species.mammal], expert)
            self.assertIsNotNone(interface.on_save)
            interface.on_save()
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
