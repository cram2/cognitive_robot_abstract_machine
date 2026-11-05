"""
Thread-safety behaviour of :mod:`krrood.symbolic_math.symbolic_math`.

Building and compiling expressions from multiple threads, each owning its own
:class:`~krrood.symbolic_math.symbolic_math.CompiledFunction`, is safe even when the
threads share the same :class:`~krrood.symbolic_math.symbolic_math.FloatVariable`
instances or the same source expression. Sharing one *compiled* function across threads
is also safe: :meth:`CompiledFunction.__call__` binds arguments into, and evaluates
from, a single reusable output buffer to avoid allocating on every call, and that
bind-then-evaluate sequence is serialised by a per-instance lock.
"""

from __future__ import annotations

import threading

import numpy as np

import krrood.symbolic_math.symbolic_math as sm

# %% safe patterns: independent state per thread, shared read-only graph nodes


def test_concurrent_float_variable_creation_keeps_registry_consistent():
    """
    Creating many :class:`~krrood.symbolic_math.symbolic_math.FloatVariable` instances
    from multiple threads at once must not corrupt the shared, class-level
    ``FloatVariable._registry``: each thread's own expression must report back exactly
    the variables that thread created, with no variables lost or borrowed from another
    thread.
    """
    variables_per_thread = 200
    thread_count = 8
    errors: list[BaseException] = []
    mismatched_threads: list[int] = []
    lock = threading.Lock()

    def create_and_check(thread_index: int) -> None:
        try:
            variables = [
                sm.FloatVariable(name=f"registry_t{thread_index}_v{i}")
                for i in range(variables_per_thread)
            ]
            expression = sm.Scalar(0)
            for variable in variables:
                expression = expression + variable
            found_variables = expression.free_variables()
            if set(map(id, found_variables)) != set(map(id, variables)):
                with lock:
                    mismatched_threads.append(thread_index)
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [
        threading.Thread(target=create_and_check, args=(i,))
        for i in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert mismatched_threads == []


def test_concurrent_jacobians_over_shared_variables_stay_correct():
    """
    Threads that build different expressions from the same shared
    :class:`~krrood.symbolic_math.symbolic_math.FloatVariable` instances, then each
    compile and evaluate their own
    :class:`~krrood.symbolic_math.symbolic_math.CompiledFunction`, must get correct,
    uncorrupted results: nothing here is written to shared mutable state, only read.
    """
    x = sm.FloatVariable(name="jacobian_shared_x")
    y = sm.FloatVariable(name="jacobian_shared_y")
    thread_count = 24
    x_value, y_value = 2.0, 3.0
    errors: list[BaseException] = []
    mismatches: list[tuple[int, float, float]] = []
    lock = threading.Lock()

    def build_compile_and_check(thread_index: int) -> None:
        power = thread_index + 1
        try:
            expression = (x**power) * y + x * y
            jacobian = expression.jacobian(variables=[x, y])
            compiled = jacobian.compile(parameters=sm.VariableParameters.from_lists([x, y]))
            result = compiled(np.array([x_value, y_value]))
            expected_dx = power * x_value ** (power - 1) * y_value + y_value
            expected_dy = x_value**power + x_value
            if not np.isclose(result[0, 0], expected_dx) or not np.isclose(
                result[0, 1], expected_dy
            ):
                with lock:
                    mismatches.append((thread_index, result[0, 0], result[0, 1]))
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [
        threading.Thread(target=build_compile_and_check, args=(i,))
        for i in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert mismatches == []


# %% the fixed bug: one CompiledFunction instance shared across threads


def test_concurrent_calls_to_a_shared_compiled_function_are_correct():
    """
    A single :class:`~krrood.symbolic_math.symbolic_math.CompiledFunction` instance,
    called concurrently from many threads with distinct inputs, must return the result
    for the calling thread's own input.

    :meth:`~krrood.symbolic_math.symbolic_math.CompiledFunction.__call__` serialises
    binding arguments into, and evaluating from, the shared output buffer, so one
    thread can no longer observe another thread's in-flight input or result.
    """
    x = sm.FloatVariable(name="shared_compiled_function_x")
    compiled = (x * x).compile(parameters=sm.VariableParameters.from_lists([x]))
    thread_count = 16
    calls_per_thread = 50
    errors: list[BaseException] = []
    mismatches: list[tuple[int, float, float]] = []
    lock = threading.Lock()

    def hammer(thread_index: int) -> None:
        value = float(thread_index + 1)
        expected = value * value
        try:
            for _ in range(calls_per_thread):
                result = compiled(np.array([value]))
                if not np.isclose(result[0], expected):
                    with lock:
                        mismatches.append((thread_index, value, float(result[0])))
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert mismatches == []
