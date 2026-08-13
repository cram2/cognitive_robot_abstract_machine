"""
Explicit data structures for call stack frames captured during EQL object creation.

Typed, memory-safe dataclasses that eagerly extract all needed data from a live
``inspect.FrameInfo`` and immediately drop the live frame reference, avoiding
memory leaks from retained frame objects.
"""

from __future__ import annotations

import inspect
import linecache
import sys
from dataclasses import dataclass
from types import FrameType
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
        snippet = (
            frame_info.code_context[0].strip() if frame_info.code_context else None
        )
        return cls._from_frame(
            frame_info.frame,
            filename=frame_info.filename,
            lineno=frame_info.lineno,
            function_name=frame_info.function,
            code_snippet=snippet,
        )

    @classmethod
    def from_raw_frame(cls, frame: FrameType) -> StackFrame:
        """
        Eagerly extract all data from a live frame object and drop the frame reference.

        Unlike :meth:`from_frame_info` this reads nothing about the surrounding stack: the
        source line is fetched cheaply from :mod:`linecache` (the file is cached after the first
        access). Must be called while *frame* is still live so that ``f_locals`` is populated.

        :param frame: The live frame to extract data from.
        :return: A self-contained :class:`StackFrame`.
        """
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        snippet = linecache.getline(filename, lineno).strip() or None
        return cls._from_frame(
            frame,
            filename=filename,
            lineno=lineno,
            function_name=frame.f_code.co_name,
            code_snippet=snippet,
        )

    @classmethod
    def _from_frame(
        cls,
        frame: FrameType,
        *,
        filename: str,
        lineno: int,
        function_name: str,
        code_snippet: Optional[str],
    ) -> StackFrame:
        """
        Build a :class:`StackFrame` from a live frame and its already-extracted location data.

        Resolves the owning class, the callable object and the dotted module name. The module
        name is read directly from ``frame.f_globals["__name__"]`` rather than via
        :func:`inspect.getmodule`, which is orders of magnitude cheaper and equivalent for normal
        frames.

        :param frame: The live frame, used only to read ``f_locals``/``f_globals``.
        :param filename: Source file path for the frame.
        :param lineno: Line number within the source file.
        :param function_name: Name of the function or method.
        :param code_snippet: One stripped source line, or ``None`` if unavailable.
        :return: The constructed :class:`StackFrame`.
        """
        instance = frame.f_locals.get("self", None)
        owner_class: Optional[type] = frame.f_locals.get("cls", None)
        if owner_class is None and instance is not None:
            owner_class = type(instance)
        resolved_function: Optional[Callable] = frame.f_globals.get(function_name, None)
        if resolved_function is None and owner_class is not None:
            resolved_function = owner_class.__dict__.get(function_name, None)
        return cls(
            filename=filename,
            lineno=lineno,
            function_name=function_name,
            code_snippet=code_snippet,
            class_object=owner_class,
            function_object=resolved_function,
            module_name=frame.f_globals.get("__name__"),
        )


def _is_external_filename(filename: str) -> bool:
    """
    :param filename: A source file path.
    :return: ``True`` when *filename* lives in an installed third-party package
        (``site-packages`` or ``dist-packages``).
    """
    return "site-packages" in filename or "dist-packages" in filename


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

    @classmethod
    def capture(cls, skip: int = 0) -> CallStack:
        """
        Capture the current call stack, building :class:`StackFrame` objects only for frames that
        are not in installed third-party packages.

        This is the fast path used during monitored object creation. Filtering happens during the
        walk, so external frames are skipped without reading their source or resolving their
        modules; only the retained frames are converted. The result is therefore already filtered
        and equivalent to ``CallStack(...).filter()`` over the full stack.

        :param skip: Number of frames to skip starting from the caller of :meth:`capture`. Use
            ``skip=0`` to start at the immediate caller, ``skip=1`` to also drop the caller, and
            so on.
        :return: A new :class:`CallStack` of the retained frames, innermost first.
        """
        try:
            frame: Optional[FrameType] = sys._getframe(1 + skip)
        except ValueError:
            return cls([])
        kept: List[StackFrame] = []
        while frame is not None:
            if not _is_external_filename(frame.f_code.co_filename):
                kept.append(StackFrame.from_raw_frame(frame))
            frame = frame.f_back
        return cls(kept)

    def filter(self, package: Optional[str] = None) -> CallStack:
        """
        Build a new :class:`CallStack` with external-library frames removed.

        :param package: When provided, keep only frames whose filename contains this string.
        :return: A new :class:`CallStack` containing only the retained frames.
        """
        kept = []
        for frame in self.frames:
            if _is_external_filename(frame.filename):
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
