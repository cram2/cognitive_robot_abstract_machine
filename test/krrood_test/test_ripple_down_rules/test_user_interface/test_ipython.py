from IPython.core.interactiveshell import ExecutionInfo
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config

from krrood.ripple_down_rules.utils import (
    capture_variable_assignment,
    contains_return_statement,
    extract_dependencies,
)


class IpythonShell(InteractiveShellEmbed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_lines = []

    def run_cell(self, raw_cell: str, **kwargs):
        """
        Override the run_cell method to capture return statements.
        """
        if contains_return_statement(raw_cell):
            self.all_lines.append(raw_cell)
            print("Exiting shell on `return` statement.")
            self.ask_exit()
            return None
        result = super().run_cell(raw_cell, **kwargs)
        if not result.error_in_exec:
            self.all_lines.append(raw_cell)
        return result


class IpythonShellManager:
    def __init__(self, scope=None, header=None):
        self.scope = scope or {}
        self.header = header or ">>> Embedded Ipython Shell"
        self.raw_condition = None
        self.shell = self._init_shell()
        self.all_code_lines = []

    def _init_shell(self):
        """
        Initialize the Ipython shell with a custom configuration.
        """
        cfg = Config()
        shell = IpythonShell(config=cfg, user_ns=self.scope, banner1=self.header)
        return shell

    def run(self):
        """
        Run the embedded shell.
        """
        self.shell()
        self.all_code_lines = extract_dependencies(self.shell.all_lines)
        user_input = f"def get_value(case):\n    "
        user_input += "\n    ".join(self.all_code_lines)
        print(user_input)
        eval(compile(user_input, "<string>", "exec"), self.scope)
        print(self.scope["get_value"](4))


def run_ipython_shell():
    x = 10
    case = 5
    scope = locals()
    mgr = IpythonShellManager(scope)
    mgr.run()
    print(mgr.all_code_lines)
    # Apply changes to outer scope
    x = scope["x"]
    print(f"Back in code: x={x}")


# run_ipython_shell()
