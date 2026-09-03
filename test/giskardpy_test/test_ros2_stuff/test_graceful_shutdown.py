import signal

import pytest

from giskardpy.middleware.ros2.graceful_shutdown import GracefulShutdownSignals


class TestSignalsBecomeKeyboardInterrupt:
    """
    Inside the ``with`` block, SIGINT and SIGTERM both raise KeyboardInterrupt, so a
    ROS2 launch file's SIGINT-then-SIGTERM escalation is handled the same way either
    time.
    """

    def test_sigint_raises_keyboard_interrupt(self):
        with pytest.raises(KeyboardInterrupt):
            with GracefulShutdownSignals():
                signal.raise_signal(signal.SIGINT)

    def test_sigterm_raises_keyboard_interrupt(self):
        with pytest.raises(KeyboardInterrupt):
            with GracefulShutdownSignals():
                signal.raise_signal(signal.SIGTERM)


class TestOriginalHandlersAreRestored:
    """
    Leaving the ``with`` block puts back whatever handled SIGINT/SIGTERM before, so
    nothing outside this feature is left with handlers it never installed.
    """

    def test_handlers_are_restored_after_a_normal_exit(self):
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        with GracefulShutdownSignals():
            pass

        assert signal.getsignal(signal.SIGINT) == original_sigint_handler
        assert signal.getsignal(signal.SIGTERM) == original_sigterm_handler

    def test_handlers_are_restored_after_the_block_raised(self):
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        with pytest.raises(KeyboardInterrupt):
            with GracefulShutdownSignals():
                signal.raise_signal(signal.SIGINT)

        assert signal.getsignal(signal.SIGINT) == original_sigint_handler
        assert signal.getsignal(signal.SIGTERM) == original_sigterm_handler


class TestLeavingWithoutASignalIsQuiet:
    """
    Restoring the previous handlers only installs them; it never delivers the signals
    they handle, so a block that no signal arrived in exits silently.
    """

    def test_leaving_the_block_raises_nothing_when_no_signal_arrived(self):
        with GracefulShutdownSignals():
            pass
