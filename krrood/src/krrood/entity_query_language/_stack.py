"""
Explicit data structures for call stack frames captured during EQL object creation.

Typed, memory-safe dataclasses that eagerly extract all needed data from a live
``inspect.FrameInfo`` and immediately drop the live frame reference, avoiding
memory leaks from retained frame objects.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing_extensions import Callable, List, Optional


@dataclass
class StackFrame:
    """A single frame in a captured call stack."""

    filename: str
    """
    Full path to the source file.
    """
    lineno: int
    """
    Line number within the source file.
    """
    function_name: str
    """
    Name of the function or method.
    """
    code_snippet: Optional[str]
    """
    One source line, stripped; ``None`` if unavailable.
    """
    class_object: Optional[type]
    """
    The class that owns this method, or ``None`` for free functions.
    """
    function_object: Optional[Callable]
    """
    The callable object for this frame, or ``None`` if not resolvable.
    """
    module_name: Optional[str]
    """
    Dotted module name (string, not ``ModuleType``) to avoid reference leaks.
    """

    @property
    def is_method(self) -> bool:
        """True when this frame is inside a class method or classmethod."""
        return self.class_object is not None

    @classmethod
    def from_frame_info(cls, frame_info: inspect.FrameInfo) -> StackFrame:
        """
        Eagerly extract all data from a live ``FrameInfo`` and drop the frame reference.

        Must be called while the frame is still on the call stack so that
        ``f_locals`` is populated.
        """
        raw_frame = frame_info.frame
        instance = raw_frame.f_locals.get("self", None)
        owner_class: Optional[type] = raw_frame.f_locals.get("cls", None)
        if owner_class is None and instance is not None:
            owner_class = type(instance)
        resolved_function: Optional[Callable] = raw_frame.f_globals.get(
            frame_info.function, None
        )
        if resolved_function is None and owner_class is not None:
            resolved_function = owner_class.__dict__.get(frame_info.function, None)
        module = inspect.getmodule(raw_frame)
        snippet = (
            frame_info.code_context[0].strip() if frame_info.code_context else None
        )
        return cls(
            filename=frame_info.filename,
            lineno=frame_info.lineno,
            function_name=frame_info.function,
            code_snippet=snippet,
            class_object=owner_class,
            function_object=resolved_function,
            module_name=module.__name__ if module else None,
        )


@dataclass
class CallStack:
    """An ordered sequence of :class:`StackFrame` objects, innermost frame first."""

    frames: List[StackFrame]
    """
    The captured stack frames.
    """

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def filter(self, package: Optional[str] = None) -> CallStack:
        """
        Build a new :class:`CallStack` with external-library frames removed.

        :param package: When provided, keep only frames whose filename contains this string.
        :return: A new :class:`CallStack` containing only the retained frames.
        """
        kept = []
        for frame in self.frames:
            if "site-packages" in frame.filename or "dist-packages" in frame.filename:
                continue
            if package is not None and package not in frame.filename:
                continue
            kept.append(frame)
        return CallStack(kept)

    def root_frame_in(self, package: str) -> Optional[StackFrame]:
        """
        Find the outermost frame (highest in the call hierarchy) whose
        ``module_name`` contains *package*.  This is the entry point into the
        library from the caller's perspective.

        :param package: Substring to match against ``module_name``.
        :return: The outermost matching :class:`StackFrame`, or ``None`` if no frame matches.
        """
        matches = [
            frame
            for frame in self.frames
            if frame.module_name and package in frame.module_name
        ]
        return matches[-1] if matches else None

    def classes(self) -> List[type]:
        """Distinct class objects appearing in the stack, in order of first occurrence."""
        seen: List[type] = []
        for frame in self.frames:
            if frame.class_object is not None and frame.class_object not in seen:
                seen.append(frame.class_object)
        return seen

    def functions(self) -> List[Callable]:
        """Distinct function objects appearing in the stack, in order of first occurrence."""
        seen: List[Callable] = []
        for frame in self.frames:
            if frame.function_object is not None and frame.function_object not in seen:
                seen.append(frame.function_object)
        return seen

    def is_from_method(self) -> bool:
        """True if any frame in this stack is inside a class method."""
        return any(frame.is_method for frame in self.frames)
