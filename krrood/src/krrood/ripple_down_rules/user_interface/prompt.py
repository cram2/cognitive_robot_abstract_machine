import ast
import logging
import sys
from _ast import AST

from krrood.ripple_down_rules import logger

try:
    from PyQt6.QtWidgets import QApplication
    from .gui import RDRCaseViewer, style
except ImportError:
    QApplication = None
    RDRCaseViewer = None

from colorama import Fore, Style
from pygments import highlight
from pygments.formatters.terminal import TerminalFormatter
from pygments.lexers.python import PythonLexer
from typing_extensions import Optional, Tuple

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
    parse_string_to_expression,
)
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import PromptFor, ExitStatus
from krrood.ripple_down_rules.user_interface.ipython_custom_shell import IPythonShell
from krrood.ripple_down_rules.utils import make_list
from threading import Lock


class UserPrompt:
    """
    A class to handle user prompts for the RDR.
    """

    shell_lock: Lock = (
        Lock()
    )  # To ensure that only one thread can access the shell at a time

    def __init__(self, prompt_user: bool = True):
        """
        Initialize the UserPrompt class.
        """
        self.viewer = (
            RDRCaseViewer.instances[0]
            if RDRCaseViewer and any(RDRCaseViewer.instances)
            else None
        )
        self.print_func = self.viewer.print if self.viewer else print

    def prompt_user_for_expression(
        self,
        case_query: CaseQuery,
        prompt_for: PromptFor,
        prompt_str: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[CallableExpression]]:
        """
        Prompt the user for an executable python expression to the given case query.

        :param case_query: The case query to prompt the user for.
        :param prompt_for: The type of information ask user about.
        :param prompt_str: The prompt string to display to the user.
        :return: A callable expression that takes a case and executes user expression on it.
        """
        with self.shell_lock:
            prev_user_input: Optional[str] = None
            user_input_to_modify: Optional[str] = None
            callable_expression: Optional[CallableExpression] = None
            while True:
                user_input, expression_tree = self.prompt_user_about_case(
                    case_query, prompt_for, prompt_str, code_to_modify=prev_user_input
                )
                if user_input is None:
                    if prompt_for == PromptFor.Conclusion:
                        self.print_func(
                            f"\n{Fore.YELLOW}No conclusion provided. Exiting.{Style.RESET_ALL}"
                        )
                        return None, None
                    else:
                        self.print_func(
                            f"\n{Fore.RED}Conditions must be provided. Please try again.{Style.RESET_ALL}"
                        )
                        continue
                elif user_input in ["exit", "quit"]:
                    self.print_func(f"\n{Fore.YELLOW}Exiting.{Style.RESET_ALL}")
                    return user_input, None

                prev_user_input = "\n".join(user_input.split("\n")[2:-1])
                conclusion_type = (
                    bool
                    if prompt_for == PromptFor.Conditions
                    else case_query.attribute_type
                )
                callable_expression = CallableExpression(
                    user_input,
                    conclusion_type,
                    expression_tree=expression_tree,
                    scope=case_query.scope,
                    mutually_exclusive=case_query.mutually_exclusive,
                )
                try:
                    result = callable_expression(case_query.case)
                    if len(make_list(result)) == 0 and (
                        user_input_to_modify is not None
                        and (prev_user_input != user_input_to_modify)
                    ):
                        user_input_to_modify = prev_user_input
                        self.print_func(
                            f"{Fore.YELLOW}The given expression gave an empty result for case {case_query.name}."
                            f" Please accept or modify!{Style.RESET_ALL}"
                        )
                        continue
                    break
                except Exception as e:
                    logging.error(e)
                    self.print_func(f"{Fore.RED}{e}{Style.RESET_ALL}")
            return user_input, callable_expression

    def prompt_user_about_case(
        self,
        case_query: CaseQuery,
        prompt_for: PromptFor,
        prompt_str: Optional[str] = None,
        code_to_modify: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[AST]]:
        """
        Prompt the user for input.

        :param case_query: The case query to prompt the user for.
        :param prompt_for: The type of information the user should provide for the given case.
        :param prompt_str: The prompt string to display to the user.
        :param code_to_modify: The code to modify. If given will be used as a start for user to modify.
        :return: The user input, and the executable expression that was parsed from the user input.
        """
        logger.debug("Entered shell")
        initial_prompt_str = f"{prompt_str}\n" if prompt_str is not None else ""
        if prompt_for == PromptFor.Conclusion:
            prompt_for_str = f"Give possible value(s) for:"
        else:
            prompt_for_str = f"Give conditions for:"
        case_query.scope.update({"case": case_query.case})
        shell = None

        if self.viewer is None:
            prompt_for_str = prompt_for_str.replace(":", f" {case_query.name}:")
            prompt_str = (
                f"{Fore.WHITE}{initial_prompt_str}{Fore.MAGENTA}{prompt_for_str}"
            )
            prompt_str = self.construct_prompt_str_for_shell(
                case_query, prompt_for, prompt_str
            )
            shell = IPythonShell(
                header=prompt_str,
                prompt_for=prompt_for,
                case_query=case_query,
                code_to_modify=code_to_modify,
            )
        else:
            prompt_str = case_query.current_value_str
            self.viewer.update_for_case_query(
                case_query,
                prompt_for=prompt_for,
                code_to_modify=code_to_modify,
                title=prompt_for_str,
                prompt_str=prompt_str,
            )
        user_input, expression_tree = self.prompt_user_input_and_parse_to_expression(
            shell=shell
        )
        logger.debug("Exited shell")
        return user_input, expression_tree

    def build_prompt_str_for_ai(
        self,
        case_query: CaseQuery,
        prompt_for: PromptFor,
        initial_prompt_str: Optional[str] = None,
    ) -> str:
        initial_prompt_str = (
            f"{initial_prompt_str}\n" if initial_prompt_str is not None else ""
        )
        if prompt_for == PromptFor.Conclusion:
            prompt_for_str = f"Give possible value(s) for:"
        else:
            prompt_for_str = f"Give conditions for:"
        prompt_for_str = prompt_for_str.replace(":", f" {case_query.name}:")
        prompt_str = f"{Fore.WHITE}{initial_prompt_str}{Fore.MAGENTA}{prompt_for_str}"
        prompt_str += "\n" + case_query.current_value_str
        return prompt_str

    def construct_prompt_str_for_shell(
        self,
        case_query: CaseQuery,
        prompt_for: PromptFor,
        prompt_str: Optional[str] = None,
    ) -> str:
        """
        Construct the prompt string for the shell.

        :param case_query: The case query to prompt the user for.
        :param prompt_for: The type of information the user should provide for the given case.
        :param prompt_str: The prompt string to display to the user.
        """
        prompt_str += "\n" + case_query.current_value_str
        if prompt_for == PromptFor.Conditions:
            prompt_str += (
                f"\ne.g. `{Fore.GREEN}return {Fore.BLUE}len{Fore.RESET}(case.attribute) > {Fore.BLUE}0` "
                f"{Fore.MAGENTA}\nOR `{Fore.GREEN}return {Fore.YELLOW}True`{Fore.MAGENTA} (If you want the"
                f" rule to be always evaluated) \n"
                f"You can also do {Fore.YELLOW}%edit{Fore.MAGENTA} for more complex conditions.\n"
            )

        prompt_str = f"{Fore.MAGENTA}{prompt_str}{Fore.YELLOW}\n(Write %help for guide){Fore.RESET}\n"
        return prompt_str

    def prompt_user_input_and_parse_to_expression(
        self, shell: Optional[IPythonShell] = None, user_input: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[ast.AST]]:
        """
        Prompt the user for input.

        :param shell: The Ipython shell to use for prompting the user.
        :param user_input: The user input to use. If given, the user input will be used instead of prompting the user.
        :return: The user input and the AST tree.
        """
        while True:
            if user_input is None:
                user_input = self.start_shell_and_get_user_input(shell=shell)
                if user_input is None or user_input in ["exit", "quit"]:
                    return user_input, None
                if logger.level <= logging.DEBUG:
                    self.print_func(
                        f"\n{Fore.GREEN}Captured User input: {Style.RESET_ALL}"
                    )
                    highlighted_code = highlight(
                        user_input, PythonLexer(), TerminalFormatter()
                    )
                    self.print_func(highlighted_code)
            try:
                return user_input, parse_string_to_expression(user_input)
            except Exception as e:
                msg = f"Error parsing expression: {e}"
                logging.error(msg)
                self.print_func(f"\n{Fore.RED}{msg}{Style.RESET_ALL}")
                user_input = None

    def start_shell_and_get_user_input(
        self, shell: Optional[IPythonShell] = None
    ) -> Optional[str]:
        """
        Start the shell and get user input.

        :param shell: The Ipython shell to use for prompting the user.
        :return: The user input.
        """
        if self.viewer is None:
            shell = IPythonShell() if shell is None else shell
            if not hasattr(shell.shell, "auto_match"):
                shell.shell.auto_match = True  # or True, depending on your preference
            shell.run()
            user_input = shell.user_input
        else:
            app = QApplication.instance()
            if app is None:
                raise RuntimeError(
                    "QApplication instance is None. Please run the application first."
                )
            self.viewer.show()
            app.exec()
            if self.viewer.exit_status == ExitStatus.CLOSE:
                sys.exit()
            user_input = self.viewer.user_input
        return user_input
