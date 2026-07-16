from unittest import TestCase
from unittest.mock import Mock, patch

from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.exceptions import NonInteractiveTerminalError
from krrood.ripple_down_rules.user_interface.ipython_custom_shell import IPythonShell


def make_shell() -> IPythonShell:
    case_query = CaseQuery(object(), "attribute", (int,), True)
    return IPythonShell(case_query=case_query)


class TestIPythonShellRunFailsFastOnNonInteractiveTerminal(TestCase):

    def test_raises_without_invoking_the_embedded_shell(self):
        shell = make_shell()
        shell.shell = Mock(side_effect=IndexError("list index out of range"))
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(NonInteractiveTerminalError):
                shell.run()
        shell.shell.assert_not_called()


class TestIPythonShellRunDoesNotRetryForever(TestCase):

    def test_propagates_the_first_failure_instead_of_looping(self):
        shell = make_shell()
        shell.shell = Mock(side_effect=IndexError("list index out of range"))
        with patch("sys.stdin.isatty", return_value=True):
            with self.assertRaises(IndexError):
                shell.run()
        shell.shell.assert_called_once()
