"""
Unit tests for :class:`cram_viz.monkey_patch.MethodPatch`.
"""

from __future__ import annotations

import pytest

from cram_viz.monkey_patch import MethodPatch


def make_greeter_class() -> type:
    """
    A fresh class whose ``greet`` method a patch replaces.

    Fresh per test: :meth:`MethodPatch.install` mutates the class itself, so sharing
    one class across tests would leak a patch from one test into the next.
    """

    class Greeter:
        def greet(self, name: str) -> str:
            return "hello, %s" % name

        @classmethod
        def named(cls, name: str) -> str:
            return "%s the greeter" % name

    return Greeter


@pytest.fixture()
def greeter_class() -> type:
    return make_greeter_class()


class TestMethodPatch:
    def test_the_replacement_receives_the_call(self, greeter_class):
        calls = []
        MethodPatch(greeter_class, "greet").install(
            lambda original, self, name: calls.append(name)
        )
        greeter_class().greet("Ada")
        assert calls == ["Ada"]

    def test_the_original_stays_reachable(self, greeter_class):
        """
        A replacement must be able to fall back to the real behaviour, not just observe
        the call.
        """
        MethodPatch(greeter_class, "greet").install(
            lambda original, self, name: original(self, name).upper()
        )
        assert greeter_class().greet("Ada") == "HELLO, ADA"

    def test_a_classmethod_keeps_its_calling_convention(self, greeter_class):
        """
        Patching a classmethod must not turn it into a plain instance method: it must
        still be callable on the class itself, with ``cls`` bound automatically.
        """
        MethodPatch(greeter_class, "named").install(
            lambda original, cls, name: original(cls, name) + "!"
        )
        assert greeter_class.named("Ada") == "Ada the greeter!"

    def test_patching_twice_chains_both_replacements(self, greeter_class):
        """
        Installing a second patch must wrap the first, not discard it — hooks are
        installed independently and must compose.
        """
        patch = MethodPatch(greeter_class, "greet")
        patch.install(lambda original, self, name: original(self, name) + "!")
        patch.install(lambda original, self, name: original(self, name) + "?")
        assert greeter_class().greet("Ada") == "hello, Ada!?"
