"""
Replacing one method of an external class while keeping access to its original body.

Used to instrument CRAM classes (coraplex, giskardpy, semantic_digital_twin) that the
observing code does not own, without losing their real behaviour.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from typing_extensions import Any, Callable


@dataclass(frozen=True)
class MethodPatch:
    """
    A method of an external class, identified so it can be replaced and later chained
    into from the replacement.
    """

    owner: type
    """
    The class whose method is being replaced.
    """

    name: str
    """
    Name of the method being replaced.
    """

    def install(self, replacement: Callable[..., Any]) -> Callable[[], None]:
        """
        Replace :attr:`owner`'s :attr:`name` method with a call into ``replacement``.

        Preserves whether the replaced method was a ``classmethod``, so the patch does
        not change its calling convention.

        :param replacement: Called as ``replacement(original, *args, **kwargs)`` on
            every invocation of the patched method; ``original`` is the method being
            replaced, already unwrapped from any ``classmethod`` descriptor.
        :return: Restores the method this call replaced.
        """
        attribute = inspect.getattr_static(self.owner, self.name)
        is_classmethod = isinstance(attribute, classmethod)
        original = attribute.__func__ if is_classmethod else attribute

        def trampoline(*args: Any, **kwargs: Any) -> Any:
            return replacement(original, *args, **kwargs)

        setattr(
            self.owner,
            self.name,
            classmethod(trampoline) if is_classmethod else trampoline,
        )

        def uninstall() -> None:
            setattr(self.owner, self.name, attribute)

        return uninstall
