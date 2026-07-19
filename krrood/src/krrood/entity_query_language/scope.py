"""
Definition-scope capture for the Entity Query Language.

When an underspecified query (or any EQL variable/query) is created, we often need to
recover the *caller's* namespace later — for example, to drive an interactive expert
shell whose globals/locals mirror the code that built the query.

This reuses the same ``inspect``-based technique as the monitoring/creation-stack
infrastructure (:mod:`krrood.entity_query_language._monitoring`,
:mod:`krrood.entity_query_language._stack`), but snapshots the namespace of the *first
user frame* — i.e. the frame outside the EQL package and outside installed site-packages
— which is "where the query was defined".
"""

from __future__ import annotations

import builtins
import inspect

from typing_extensions import Any, Dict, Optional, Tuple

import krrood.entity_query_language
from krrood.entity_query_language import factories as eql
from krrood.entity_query_language.utils import T
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    Next,
    Refinement,
)

_SCOPE_ATTRIBUTE = "_definition_scope_"
"""
Attribute name under which a captured scope snapshot is stashed on an object.
"""


def _is_internal_frame(
    frame_info: inspect.FrameInfo, skip_packages: Tuple[str, ...]
) -> bool:
    """
    :return: True if this frame belongs to EQL internals, an installed package, or
        one of the explicitly skipped packages — i.e. not user code.
    """
    filename = frame_info.filename
    if "site-packages" in filename or "dist-packages" in filename:
        return True
    # Synthetic frames with no backing source file — e.g. the dataclass-generated
    # __init__ ("<string>") or frozen importlib frames — are never the user's
    # definition site. Interactive sessions ("<stdin>", "<ipython-input-...>") are
    # genuine user code, so they are deliberately not excluded here.
    if filename.startswith("<") and not (
        filename.startswith("<stdin") or filename.startswith("<ipython")
    ):
        return True
    module = inspect.getmodule(frame_info.frame)
    module_name = module.__name__ if module else ""
    return any(
        module_name == pkg or module_name.startswith(pkg + ".") for pkg in skip_packages
    )


def capture_caller_scope(
    skip_packages: Tuple[str, ...] = (), max_depth: int = 60
) -> Dict[str, Any]:
    """
    Snapshot the merged ``{**f_globals, **f_locals}`` of the innermost user frame.

    Walks outward from the immediate caller, skipping frames that belong to the EQL
    package or installed site-/dist-packages, and returns a shallow copy of the first
    user frame's namespace (locals overriding globals).

    :param skip_packages: Additional dotted module prefixes whose frames are treated as
        internal; the EQL package itself is always skipped.
    :param max_depth: Maximum number of frames to inspect (safety bound).
    :return: A new dict with the caller's namespace, or ``{}`` if none was found.
    """
    internal_packages = (krrood.entity_query_language.__name__,) + tuple(
        skip_packages or ()
    )
    frames = inspect.stack()
    try:
        # frames[0] is capture_caller_scope itself; start from its caller.
        for frame_info in frames[1:max_depth]:
            if _is_internal_frame(frame_info, internal_packages):
                continue
            frame = frame_info.frame
            return {**frame.f_globals, **frame.f_locals}
        return {}
    finally:
        del frames


def eql_factory_namespace() -> Dict[str, Any]:
    """
    :return: A namespace mapping of EQL factory callables for use in an interactive
        shell or evaluation scope.

    Automatically includes every public name defined directly in
    :mod:`krrood.entity_query_language.factories` (``entity``, ``an``, ``variable``,
    ``and_``/``or_``/``not_``, ``contains``, ``exists``, ``for_all``, the rule-tree
    builders ``refinement``/``alternative``/``next_rule``/``add``, and so on) — a new
    public factory becomes available here automatically, without editing this
    function. Names that would shadow a Python builtin (e.g. ``max``, ``sum``) are
    left out of the flat namespace; the ``factories`` module itself is included as
    ``eql`` so those stay reachable as ``eql.max``, ``eql.sum``, etc. The rule-tree
    node classes (``Add``, ``Refinement``, ``Alternative``, ``Next``) are added
    explicitly since they're defined outside ``factories``.
    """
    namespace = {
        name: value
        for name, value in vars(eql).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == eql.__name__
        and not hasattr(builtins, name)
    }
    namespace["eql"] = eql
    namespace.update(
        {
            "Add": Add,
            "Refinement": Refinement,
            "Alternative": Alternative,
            "Next": Next,
        }
    )
    return namespace


def attach_definition_scope(obj: T, scope: Optional[Dict[str, Any]] = None) -> T:
    """
    Capture (or store a provided) caller scope onto ``obj`` for later retrieval.

    :param obj: The object (typically an underspecified query/variable) to annotate.
    :param scope: An explicit scope to attach; if ``None``, the caller scope is
        captured.
    :return:``obj`` (for chaining).
    """
    if scope is None:
        scope = capture_caller_scope()
    try:
        setattr(obj, _SCOPE_ATTRIBUTE, scope)
    except (AttributeError, TypeError):
        # Object does not permit attribute assignment (e.g. has __slots__); skip silently.
        pass
    return obj


def get_definition_scope(
    obj: Optional[Any] = None, *, include_factories: bool = True
) -> Dict[str, Any]:
    """
    Return the definition scope to use for an interactive expert session.

    Resolution order:
      1. A scope previously attached to ``obj`` via :func:`attach_definition_scope`
         (i.e. captured "where the query was defined").
      2. Otherwise, the live caller scope captured now.

    The EQL factory namespace is layered on top (and wins over user names) so the
    shell always has the EQL verbs available.

    :param obj: An object that may carry an attached definition scope.
    :param include_factories: Whether to overlay the EQL factory namespace.
    :return: A merged namespace dict.
    """
    scope: Dict[str, Any] = {}
    # Read from ``__dict__`` rather than ``getattr``: an EQL variable's ``__getattr__``
    # synthesises a symbolic attribute for *any* missing name, so ``getattr(.., None)``
    # would never return ``None`` (it returns a bogus expression) for a variable that
    # carries no attached scope — e.g. one rebuilt by ``load_rdr``.
    attached = (
        getattr(obj, "__dict__", {}).get(_SCOPE_ATTRIBUTE) if obj is not None else None
    )
    if attached:
        scope.update(attached)
    else:
        scope.update(capture_caller_scope())
    if include_factories:
        scope.update(eql_factory_namespace())
    return scope
