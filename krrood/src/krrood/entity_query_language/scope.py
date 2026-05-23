"""
Definition-scope capture for the Entity Query Language.

When an underspecified query (or any EQL variable/query) is created, we often need
to recover the *caller's* namespace later — for example, to drive an interactive
expert shell whose globals/locals mirror the code that built the query.

This reuses the same ``inspect``-based technique as the monitoring/creation-stack
infrastructure (:mod:`krrood.entity_query_language._monitoring`,
:mod:`krrood.entity_query_language._stack`), but snapshots the namespace of the
*first user frame* — i.e. the frame outside the EQL package and outside installed
site-packages — which is "where the query was defined".
"""

from __future__ import annotations

import inspect

from typing_extensions import Any, Dict, Optional

EQL_PACKAGE = "krrood.entity_query_language"
"""Dotted prefix identifying frames internal to the EQL implementation."""

_SCOPE_ATTR = "_definition_scope_"
"""Attribute name under which a captured scope snapshot is stashed on an object."""


def _is_internal_frame(frame_info: inspect.FrameInfo, skip_packages) -> bool:
    """
    :return: True if this frame belongs to EQL internals, an installed package, or
        one of the explicitly skipped packages — i.e. not user code.
    """
    filename = frame_info.filename
    if "site-packages" in filename or "dist-packages" in filename:
        return True
    module = inspect.getmodule(frame_info.frame)
    module_name = module.__name__ if module else ""
    return any(
        module_name == pkg or module_name.startswith(pkg + ".") for pkg in skip_packages
    )


def capture_caller_scope(
    skip_packages=(EQL_PACKAGE,), max_depth: int = 60
) -> Dict[str, Any]:
    """
    Snapshot the merged ``{**f_globals, **f_locals}`` of the innermost user frame.

    Walks outward from the immediate caller, skipping frames that belong to the EQL
    package or installed site-/dist-packages, and returns a shallow copy of the first
    user frame's namespace (locals overriding globals).

    :param skip_packages: Dotted module prefixes whose frames are treated as internal.
    :param max_depth: Maximum number of frames to inspect (safety bound).
    :return: A new dict with the caller's namespace, or ``{}`` if none was found.
    """
    frames = inspect.stack()
    try:
        # frames[0] is capture_caller_scope itself; start from its caller.
        for frame_info in frames[1:max_depth]:
            if _is_internal_frame(frame_info, skip_packages):
                continue
            frame = frame_info.frame
            return {**frame.f_globals, **frame.f_locals}
        return {}
    finally:
        del frames


def eql_factory_namespace() -> Dict[str, Any]:
    """
    :return: A namespace mapping of EQL factory callables for use in an interactive
        shell or evaluation scope. Provides the verbs needed to author conditions and
        rule-tree nodes (``entity``, ``an``, ``variable``, ``and_``/``or_``/``not_``,
        ``contains``, ``exists``, ``for_all``, and the rule-tree builders
        ``refinement``/``alternative``/``next_rule``/``add`` plus their classes).
    """
    from krrood.entity_query_language import factories as f
    from krrood.entity_query_language.rules.conclusion import Add
    from krrood.entity_query_language.rules.conclusion_selector import (
        Alternative,
        Next,
        Refinement,
    )

    return {
        # query construction
        "entity": f.entity,
        "set_of": f.set_of,
        "an": f.an,
        "the": f.the,
        # variable declaration
        "variable": f.variable,
        "variable_from": f.variable_from,
        "match": f.match,
        "match_variable": f.match_variable,
        "underspecified": f.underspecified,
        "inference": f.inference,
        # logical operators
        "and_": f.and_,
        "or_": f.or_,
        "not_": f.not_,
        "for_all": f.for_all,
        "exists": f.exists,
        "contains": f.contains,
        "in_": f.in_,
        # rule-tree builders (functions)
        "add": f.add,
        "refinement": f.refinement,
        "alternative": f.alternative,
        "next_rule": f.next_rule,
        # rule-tree primitives (classes)
        "Add": Add,
        "Refinement": Refinement,
        "Alternative": Alternative,
        "Next": Next,
    }


def attach_definition_scope(obj: Any, scope: Optional[Dict[str, Any]] = None) -> Any:
    """
    Capture (or store a provided) caller scope onto ``obj`` for later retrieval.

    :param obj: The object (typically an underspecified query/variable) to annotate.
    :param scope: An explicit scope to attach; if ``None``, the caller scope is captured.
    :return: ``obj`` (for chaining).
    """
    if scope is None:
        scope = capture_caller_scope()
    try:
        setattr(obj, _SCOPE_ATTR, scope)
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
    attached = getattr(obj, _SCOPE_ATTR, None) if obj is not None else None
    if attached:
        scope.update(attached)
    else:
        scope.update(capture_caller_scope())
    if include_factories:
        scope.update(eql_factory_namespace())
    return scope
