"""
Explicit data structures for call stack frames captured during EQL object creation.

Replaces raw ``inspect.FrameInfo`` namedtuples with typed, memory-safe dataclasses
that eagerly extract all needed data and drop the live frame reference immediately.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class StackFrame:
    """A single frame in a captured call stack."""

    filename: str
    lineno: int
    function_name: str
    code_snippet: Optional[str]
    """One source line, stripped; ``None`` if unavailable."""
    class_object: Optional[type]
    """The class that owns this method, or ``None`` for free functions."""
    function_object: Optional[Callable]
    """The callable object for this frame, or ``None`` if not resolvable."""
    module_name: Optional[str]
    """Dotted module name (string, not ``ModuleType``) to avoid reference leaks."""

    @property
    def is_method(self) -> bool:
        """True when this frame is inside a class method or classmethod."""
        return self.class_object is not None

    @classmethod
    def from_frame_info(cls, fi: inspect.FrameInfo) -> StackFrame:
        """
        Eagerly extract all data from a live ``FrameInfo`` and drop the frame reference.

        Must be called while the frame is still on the call stack so that
        ``f_locals`` is populated.
        """
        f = fi.frame
        self_obj = f.f_locals.get('self', None)
        cls_obj: Optional[type] = f.f_locals.get('cls', None)
        if cls_obj is None and self_obj is not None:
            cls_obj = type(self_obj)
        fn_obj: Optional[Callable] = f.f_globals.get(fi.function, None)
        if fn_obj is None and cls_obj is not None:
            fn_obj = cls_obj.__dict__.get(fi.function, None)
        module = inspect.getmodule(f)
        snippet = fi.code_context[0].strip() if fi.code_context else None
        return cls(
            filename=fi.filename,
            lineno=fi.lineno,
            function_name=fi.function,
            code_snippet=snippet,
            class_object=cls_obj,
            function_object=fn_obj,
            module_name=module.__name__ if module else None,
        )


@dataclass
class CallStack:
    """An ordered sequence of :class:`StackFrame` objects, innermost frame first."""

    frames: List[StackFrame]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def filter(self, package: Optional[str] = None) -> CallStack:
        """
        Return a new :class:`CallStack` with external-library frames removed.

        :param package: If given, keep only frames whose filename contains this string.
        """
        kept = []
        for f in self.frames:
            if "site-packages" in f.filename or "dist-packages" in f.filename:
                continue
            if package is not None and package not in f.filename:
                continue
            kept.append(f)
        return CallStack(kept)

    def root_frame_in(self, package: str) -> Optional[StackFrame]:
        """
        Return the outermost frame (highest in the call hierarchy) whose
        ``module_name`` contains *package*.  This is the entry point into the
        library from the caller's perspective.

        :param package: Substring to match against ``module_name``.
        :return: The outermost matching :class:`StackFrame`, or ``None``.
        """
        matches = [f for f in self.frames if f.module_name and package in f.module_name]
        return matches[-1] if matches else None

    def classes(self) -> List[type]:
        """Distinct class objects appearing in the stack, in order of first occurrence."""
        seen: List[type] = []
        for f in self.frames:
            if f.class_object is not None and f.class_object not in seen:
                seen.append(f.class_object)
        return seen

    def functions(self) -> List[Callable]:
        """Distinct function objects appearing in the stack, in order of first occurrence."""
        seen: List[Callable] = []
        for f in self.frames:
            if f.function_object is not None and f.function_object not in seen:
                seen.append(f.function_object)
        return seen

    def is_from_method(self) -> bool:
        """True if any frame in this stack is inside a class method."""
        return any(f.is_method for f in self.frames)
