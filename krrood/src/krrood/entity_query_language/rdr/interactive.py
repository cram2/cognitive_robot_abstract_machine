"""
Interactive :class:`ExpertInterface` backed by an embedded IPython shell.

The expert is shown the case rendered as a table with the instructions printed *below* it
(nearest the prompt), then authors a **live EQL condition expression** over ``case_variable``
and assigns it to ``conditions`` (and a ``conclusion`` when no ground-truth target is known).

Pressing Ctrl-D *submits*: the assignment is validated and, if it is invalid or missing, the
error is printed inline and the **same shell stays open** rather than bailing out. Calling
``exit()`` (or ``quit()``) cancels the session unconditionally, raising
:class:`~krrood.entity_query_language.rdr.interface.ExpertAbort`.

The actual shell launch is injectable (``shell_runner``) so tests can play the expert's
part without a real terminal.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Callable, Dict, List, Optional

from colorama import Fore, Style

from krrood.entity_query_language.rdr.case_table import (
    DEFAULT_COLUMNS,
    render_case_table,
)
from krrood.entity_query_language.rdr.interface import (
    CASE_INSTANCE_NAME,
    CASE_VARIABLE_NAME,
    EXIT_NAME,
    _ABORT_FLAG,
    AnswerRequest,
    CaseContext,
    ExpertInterface,
)

#: A shell runner takes ``(namespace, header)`` and must leave the expert's assignments
#: (and any ``exit()`` flag) visible in ``namespace`` when it returns.
ShellRunner = Callable[[Dict[str, Any], str], None]


@dataclass
class IPythonInterface(ExpertInterface):
    """Elicits a rule's answers through an embedded IPython shell."""

    shell_runner: Optional[ShellRunner] = None
    """Injectable launcher; defaults to a real embedded IPython shell. Tests pass a stub."""

    case_table_columns: int = DEFAULT_COLUMNS
    """Number of ``(attribute, value)`` pairs the case table lays out per row."""

    def _render_header(
        self,
        context: CaseContext,
        requests: List[AnswerRequest],
        errors: Dict[str, str],
    ) -> str:
        parts: List[str] = [
            render_case_table(context.case_instance, self.case_table_columns)
        ]
        parts.append(
            f"{Fore.CYAN}{Style.BRIGHT}EQL RDR — author a rule for this case."
            f"{Style.RESET_ALL}"
        )
        parts.append(
            f"{Fore.MAGENTA}current conclusion: {Style.RESET_ALL}"
            f"{Fore.WHITE}{context.current_conclusion!r}{Style.RESET_ALL}"
        )
        if context.has_target:
            parts.append(
                f"{Fore.MAGENTA}target conclusion:  {Style.RESET_ALL}"
                f"{Fore.GREEN}{context.target_conclusion!r}{Style.RESET_ALL}"
            )
        parts.append(
            f"{Fore.MAGENTA}Inspect the concrete case with "
            f"`{CASE_INSTANCE_NAME}` (e.g. {CASE_INSTANCE_NAME}.some_attr).{Style.RESET_ALL}"
        )
        parts.append(
            f"{Fore.YELLOW}Build your answer(s) over `{CASE_VARIABLE_NAME}` "
            f"(the EQL variable), e.g.:{Style.RESET_ALL}"
        )
        for request in requests:
            parts.append(f"  {Fore.GREEN}{request.example}{Style.RESET_ALL}")
        parts.append(
            f"{Fore.YELLOW}Press Ctrl-D to submit. "
            f"Call {Fore.CYAN}{EXIT_NAME}(){Fore.YELLOW} to cancel.{Style.RESET_ALL}"
        )
        error_block = self._format_errors(errors)
        if error_block:
            parts.append(error_block)
        return "\n".join(parts)

    @staticmethod
    def _format_errors(errors: Dict[str, str]) -> str:
        """:return: A red, one-line-per-error block, or ``""`` when there are no errors."""
        return "\n".join(
            f"{Fore.RED}[error] {name}: {message}{Style.RESET_ALL}"
            for name, message in errors.items()
        )

    def _run(
        self,
        namespace: Dict[str, Any],
        header: str,
        validate: Callable[[], Dict[str, str]],
    ) -> None:
        if self.shell_runner is not None:
            self.shell_runner(namespace, header)
        else:
            self._default_run_shell(namespace, header, validate)

    def _default_run_shell(
        self,
        namespace: Dict[str, Any],
        header: str,
        validate: Callable[[], Dict[str, str]],
    ) -> None:
        from IPython.terminal.embed import InteractiveShellEmbed

        class _ValidatingEmbeddedShell(InteractiveShellEmbed):
            """Vetoes a Ctrl-D exit while the answer is invalid; ``exit()`` forces the leave."""

            def ask_exit(self) -> None:
                if getattr(self, "_force_exit", False):
                    super().ask_exit()
                    return
                errors = validate()
                if errors:
                    print(IPythonInterface._format_errors(errors))
                    return
                super().ask_exit()

        shell = _ValidatingEmbeddedShell(banner1=header, user_ns=namespace)
        shell.auto_match = True
        shell.confirm_exit = False
        shell._force_exit = False

        def _cancel() -> None:
            shell._force_exit = True
            namespace[_ABORT_FLAG] = True
            shell.ask_exit()

        namespace[EXIT_NAME] = _cancel
        namespace["quit"] = _cancel
        # The shell shares ``namespace``, so the expert's assignments are already visible
        # to the caller when it returns.
        shell()
