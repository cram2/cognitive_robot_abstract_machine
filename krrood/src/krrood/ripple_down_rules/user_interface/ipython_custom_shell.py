import logging
import os
from typing_extensions import Optional, List

from IPython.core.magic import magics_class, Magics, line_magic
from IPython.terminal.embed import InteractiveShellEmbed
from colorama import Fore, Style
from traitlets.config import Config

from krrood.ripple_down_rules.user_interface.template_file_creator import TemplateFileCreator
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import PromptFor
from krrood.ripple_down_rules.utils import contains_return_statement, extract_dependencies, encapsulate_code_lines_into_a_function


@magics_class
class MyMagics(Magics):

    def __init__(self, shell,
                 code_to_modify: Optional[str] = None,
                 prompt_for: Optional[PromptFor] = None,
                 case_query: Optional[CaseQuery] = None):
        super().__init__(shell)
        self.case_query: Optional[CaseQuery] = case_query
        self.rule_editor = TemplateFileCreator(case_query, prompt_for=prompt_for, code_to_modify=code_to_modify)
        self.all_code_lines: Optional[List[str]] = None
        self.edited: bool = False
        self.loaded: bool = False

    @line_magic
    def edit(self, line):
        if self.edited:
            self.rule_editor.open_file_in_editor()
        else:
            self.rule_editor.edit()
            self.edited = True

    @line_magic
    def load(self, line):
        self.all_code_lines, updates = self.rule_editor.load(self.rule_editor.temp_file_path,
                                                             self.rule_editor.func_name,
                                                             self.rule_editor.print_func)
        self.shell.user_ns.update(updates)
        self.case_query.scope.update(updates)
        self.loaded = True

    @line_magic
    def current_value(self, line):
        """
        Display the current value of the attribute of the case.
        """
        if self.case_query is None:
            print(f"{Fore.RED}No case query provided.{Style.RESET_ALL}")
            return
        print(self.case_query.current_value_str)

    @line_magic
    def show_rule_tree(self, line):
        """
        Display the rule tree for the current case query.
        """
        if self.case_query is None:
            print(f"{Fore.RED}No case query provided.{Style.RESET_ALL}")
            return
        if self.case_query.rdr is None:
            print(f"{Fore.RED}No RDR available for the current case query.{Style.RESET_ALL}")
            return
        self.case_query.render_rule_tree(view=True)

    @line_magic
    def help(self, line):
        """
        Display help information for the Ipython shell.
        """
        help_text = f"""
Directly write python code in the shell, and then `{Fore.GREEN}return {Fore.RESET}output`. Or use 
the magic commands to write the code in a temporary file and edit it in PyCharm:
{Fore.MAGENTA}%edit{Style.RESET_ALL}
Opens a temporary file in PyCharm for editing a function (conclusion or conditions for case)
 that will be executed on the case object.
{Fore.MAGENTA}%load{Style.RESET_ALL}
Loads the function defined in the temporary file into the user namespace, that can then be used inside the
 Ipython shell. You can then do `{Fore.GREEN}return {Fore.RESET}function_name(case)`.
{Fore.MAGENTA}%current_value{Style.RESET_ALL}
Shows the current value of the case attribute on which the rule are being fit.
{Fore.MAGENTA}%show_rule_tree{Style.RESET_ALL}
Displays the rule tree for the current case query.
        """
        print(help_text)


class CustomInteractiveShell(InteractiveShellEmbed):
    def __init__(self, code_to_modify: Optional[str] = None,
                 prompt_for: Optional[PromptFor] = None,
                 case_query: Optional[CaseQuery] = None,
                 **kwargs):
        kwargs.update({'user_ns': case_query.scope})
        super().__init__(**kwargs)
        self.my_magics = MyMagics(self, code_to_modify=code_to_modify,
                                  prompt_for=prompt_for, case_query=case_query)
        self.register_magics(self.my_magics)
        self.all_lines = []

    def run_cell(self, raw_cell: str, **kwargs):
        """
        Override the run_cell method to capture return statements.
        """
        if contains_return_statement(raw_cell) and 'def ' not in raw_cell:
            if self.my_magics.rule_editor.func_name in raw_cell:
                self.all_lines = self.my_magics.all_code_lines
            self.all_lines.append(raw_cell)
            self.history_manager.store_inputs(line_num=self.execution_count, source=raw_cell)
            self.ask_exit()
            return None
        result = super().run_cell(raw_cell, **kwargs)
        if result.error_in_exec is None and result.error_before_exec is None:
            self.all_lines.append(raw_cell)
        return result


class IPythonShell:
    """
    Create an embedded Ipython shell that can be used to prompt the user for input.
    """

    def __init__(self, header: Optional[str] = None,
                 prompt_for: Optional[PromptFor] = None, case_query: Optional[CaseQuery] = None,
                 code_to_modify: Optional[str] = None):
        """
        Initialize the Ipython shell with the given scope and header.

        :param header: The header to display when the shell is started.
        :param prompt_for: The type of information to ask the user about.
        :param case_query: The case query which contains the case and the attribute to ask about.
        :param code_to_modify: The code to modify. If given, will be used as a start for user to modify.
        """
        self.header: str = header or ">>> Embedded Ipython Shell"
        self.case_query: Optional[CaseQuery] = case_query
        self.prompt_for: Optional[PromptFor] = prompt_for
        self.code_to_modify: Optional[str] = code_to_modify
        self.user_input: Optional[str] = None
        self.shell: CustomInteractiveShell = self._init_shell()
        self.all_code_lines: List[str] = []

    def _init_shell(self):
        """
        Initialize the Ipython shell with a custom configuration.
        """
        cfg = Config()
        shell = CustomInteractiveShell(config=cfg, banner1=self.header,
                                       code_to_modify=self.code_to_modify,
                                       prompt_for=self.prompt_for,
                                       case_query=self.case_query,
                                       )
        return shell

    def run(self):
        """
        Run the embedded shell.
        """
        while True:
            try:
                self.shell()
                self.update_user_input_from_code_lines()
                break
            except Exception as e:
                logging.error(e)
                print(f"{Fore.RED}ERROR:: {e}{Style.RESET_ALL}")

    def update_user_input_from_code_lines(self):
        """
        Update the user input from the code lines captured in the shell.
        """
        if self.shell.all_lines[-1] in ['quit', 'exit']:
            self.user_input = 'exit'
        elif self.shell.all_lines[-1].replace('return', '').strip() == '':
            self.user_input = None
        else:
            self.all_code_lines = extract_dependencies(self.shell.all_lines)
            if len(self.all_code_lines) == 1 and self.all_code_lines[0].strip() == '':
                self.user_input = None
            else:
                self.user_input = encapsulate_code_lines_into_a_function(self.all_code_lines,
                                                                         function_name=self.shell.my_magics.rule_editor.func_name,
                                                                         function_signature=self.shell.my_magics.rule_editor.function_signature,
                                                                         func_doc=self.shell.my_magics.rule_editor.func_doc,
                                                                         case_query=self.case_query)
