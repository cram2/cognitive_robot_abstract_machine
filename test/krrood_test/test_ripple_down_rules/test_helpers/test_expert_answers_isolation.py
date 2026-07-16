import os
import threading

from .helpers import isolated_expert_answers_path

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "test_expert_answers", "scrdr_expert_answers_fit"
)


def read_fixture_content() -> str:
    """
    :return: The current content of the committed ``scrdr_expert_answers_fit.py``
        fixture used by these tests.
    """
    with open(FIXTURE_PATH + ".py", "r") as f:
        return f.read()


def test_isolated_expert_answers_path_yields_a_separate_copy():
    """
    The isolated path must point at a different file than the committed fixture, with
    identical content, so callers never operate on the shared file directly.
    """
    original_content = read_fixture_content()

    with isolated_expert_answers_path(FIXTURE_PATH) as isolated_path:
        assert isolated_path != FIXTURE_PATH
        assert os.path.exists(isolated_path + ".py")
        with open(isolated_path + ".py", "r") as f:
            assert f.read() == original_content


def test_isolated_expert_answers_path_survives_deletion_of_the_copy():
    """
    Regression test: deleting the isolated copy - exactly what ``Human.__init__`` does
    for a freshly loaded, non-appending expert - must never delete or otherwise affect
    the committed fixture it was copied from.
    """
    original_content = read_fixture_content()

    with isolated_expert_answers_path(FIXTURE_PATH) as isolated_path:
        os.remove(isolated_path + ".py")
        assert not os.path.exists(isolated_path + ".py")
        assert os.path.exists(FIXTURE_PATH + ".py")
        assert read_fixture_content() == original_content

    assert read_fixture_content() == original_content


def test_isolated_expert_answers_path_is_safe_under_concurrent_use():
    """
    Regression test for the parallel-unsafe expert-answer fixture race: many concurrent
    "workers" (threads) each isolate and then delete their own copy of the same
    committed fixture, exactly as ``Human.__init__`` does when constructed with
    ``use_loaded_answers=True`` and ``append=False``.

    The committed fixture must never be observed missing, and must never be mutated, no
    matter how the workers interleave.
    """
    original_content = read_fixture_content()
    worker_count = 8
    iterations_per_worker = 20
    errors = []

    def worker():
        try:
            for _ in range(iterations_per_worker):
                with isolated_expert_answers_path(FIXTURE_PATH) as isolated_path:
                    assert os.path.exists(isolated_path + ".py")
                    os.remove(isolated_path + ".py")
                    assert os.path.exists(
                        FIXTURE_PATH + ".py"
                    ), "the committed fixture must survive a concurrent worker's delete"
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads), "a worker thread hung"
    assert not errors, f"errors during concurrent access: {errors}"
    assert read_fixture_content() == original_content
