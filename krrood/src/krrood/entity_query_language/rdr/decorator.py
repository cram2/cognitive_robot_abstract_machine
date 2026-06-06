"""
``@rdr`` decorator — wraps a function so every call is classified (or fitted)
by an :class:`EQLSingleClassRDR` whose case type is generated from the
function's own signature.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Any, Callable, Optional, Type

from krrood.code_generation import (
    FunctionMissingAnnotationsError,
    function_to_dataclass_source,
    to_camel_case,
    to_variable_name,
)
from krrood.entity_query_language.rdr.file_store import RDRFileStore
from krrood.entity_query_language.rdr.function_case import FunctionCase
from krrood.entity_query_language.rdr.serialization import (
    _FACTORY_IMPORT,
    load_rdr,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.utils import UNSET

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.expert import Expert

#: Name of the dataclass field that holds the RDR prediction target.
_OUTPUT_FIELD: str = "_output"


def _empty_rdr_preamble(class_name: str) -> str:
    """Return the rule-tree section for a brand-new (empty) RDR file."""
    import os

    from krrood.code_generation import CodeGenerator

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    generator = CodeGenerator(template_dir=template_dir)
    return generator.render(
        "rdr_empty.py.jinja",
        factory_import=_FACTORY_IMPORT,
        var_name=to_variable_name(class_name),
        class_name=class_name,
    )


@dataclass
class RDRWrapper:
    """Wraps a function to intercept calls and classify them via :class:`EQLSingleClassRDR`.

    In **inference mode** (``fit_mode=False``, the default) each call runs the
    original function, builds a :class:`FunctionCase`, and classifies it.  If a
    rule fires the RDR's conclusion replaces the original return value; otherwise
    the original value is returned unchanged.

    In **fit mode** (``fit_mode=True``) the original return value is always
    returned and :meth:`~EQLSingleClassRDR.fit_case` is invoked on every call so
    an expert can grow the rule tree interactively.
    """

    func: Callable
    """The original undecorated callable."""
    store: RDRFileStore
    """Manages the combined class + rule-tree ``.py`` file on disk."""
    expert: "Optional[Expert]"
    """Expert used when ``fit_mode=True``; may be overridden per call."""
    fit_mode: bool
    """When ``True`` every call triggers fitting; when ``False`` (default) only classifies."""

    case_type: Type[FunctionCase] = field(init=False)
    """The :class:`FunctionCase` subclass generated from :attr:`func`'s signature."""
    rdr: EQLSingleClassRDR = field(init=False)
    """The live :class:`EQLSingleClassRDR` used for classification and fitting."""

    def __post_init__(self) -> None:
        self.case_type, self.rdr = self._load_or_generate()
        self.case_type.function = self.func
        self.rdr.save_path = self.store.path
        functools.update_wrapper(self, self.func, updated=[])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept a function call: run original → build case → classify or fit."""
        output = self.func(*args, **kwargs)
        case = self._build_case(args, kwargs, output)
        if self.fit_mode:
            if self.expert is not None:
                self.rdr.fit_case(case, target=UNSET, expert=self.expert)
            return output
        conclusion = self.rdr.classify(case)
        return output if (conclusion is UNSET or conclusion is None) else conclusion

    def fit_case(
        self, case: Any, target: Any = UNSET, expert: "Optional[Expert]" = None
    ) -> Any:
        """Delegate to the internal RDR, using :attr:`expert` when none is supplied."""
        return self.rdr.fit_case(case, target, expert if expert is not None else self.expert)

    def fit(
        self,
        cases: list,
        targets: "Optional[list]" = None,
        expert: "Optional[Expert]" = None,
    ) -> EQLSingleClassRDR:
        """Delegate to the internal RDR's batch :meth:`~EQLSingleClassRDR.fit`."""
        return self.rdr.fit(
            cases, targets, expert if expert is not None else self.expert
        )

    def _load_or_generate(self) -> "tuple[Type[FunctionCase], EQLSingleClassRDR]":
        if self.store.exists():
            case_type = self.store.load_case_type()
            rdr = load_rdr(self.store.path)
        else:
            class_name = to_camel_case(self.func.__name__)
            class_source = function_to_dataclass_source(self.func)
            preamble = _empty_rdr_preamble(class_name)
            source = class_source + "\n\n\n" + preamble
            Path(self.store.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.store.path).write_text(source)
            case_type = self.store.load_case_type()
            rdr = EQLSingleClassRDR(case_type, _OUTPUT_FIELD)
        return case_type, rdr

    def _build_case(self, args: tuple, kwargs: dict, output: Any) -> FunctionCase:
        """Bind call arguments to the :class:`FunctionCase` fields."""
        sig = inspect.signature(self.func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
        return self.case_type(**params, _output=output)


def rdr(
    filename: Optional[str] = None,
    *,
    expert: "Optional[Expert]" = None,
    fit: bool = False,
) -> Callable[[Callable], RDRWrapper]:
    """Decorator that augments a function with an EQL Ripple Down Rules classifier.

    Usage::

        @rdr("my_model.py")
        def decide(x: float, state: str) -> Action:
            return Action.DEFAULT   # fallback

        @rdr("my_model.py", fit=True, expert=Expert(interface=IPythonInterface()))
        def decide(x: float, state: str) -> Action:
            return Action.DEFAULT

    :param filename: **Required.** User-supplied ``.py`` filename for the model
        file.  If relative it is placed in a ``_rdr_models/`` subdirectory beside
        the decorated function's module file.  If absolute it is used as-is.
    :param expert: :class:`~krrood.entity_query_language.rdr.expert.Expert` to
        use during fitting.
    :param fit: When ``True`` every call to the decorated function triggers
        :meth:`~EQLSingleClassRDR.fit_case`; when ``False`` (default) calls only
        classify.
    :raises FunctionMissingAnnotationsError: If the decorated function lacks type
        annotations on any parameter (excluding ``self``/``cls``) or the return type.
    :returns: A decorator that replaces the function with an :class:`RDRWrapper`.
    """
    if filename is None:
        raise TypeError("rdr() requires a filename as its first argument.")

    def _decorate(func: Callable) -> RDRWrapper:
        store = RDRFileStore(func=func, filename=filename)
        return RDRWrapper(func=func, store=store, expert=expert, fit_mode=fit)

    return _decorate
