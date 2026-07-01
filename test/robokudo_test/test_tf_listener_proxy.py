from robokudo.io import tf_listener_proxy
from robokudo.io.tf_listener_proxy import TFListenerProxy


class FakeBuffer:
    pass


class FakeTransformListener:
    calls = []

    def __init__(self, buffer, node):
        self.buffer = buffer
        self.node = node
        self.calls.append((buffer, node))


def setup_function():
    TFListenerProxy.clear()
    FakeTransformListener.calls.clear()


def teardown_function():
    TFListenerProxy.clear()
    FakeTransformListener.calls.clear()


def test_buffer_for_node_reuses_listener_for_same_node(monkeypatch):
    monkeypatch.setattr(tf_listener_proxy, "Buffer", FakeBuffer)
    monkeypatch.setattr(tf_listener_proxy, "TransformListener", FakeTransformListener)

    node = object()

    first_buffer = TFListenerProxy().buffer_for_node(node)
    second_buffer = TFListenerProxy().buffer_for_node(node)

    assert first_buffer is second_buffer
    assert len(FakeTransformListener.calls) == 1
    assert FakeTransformListener.calls[0] == (first_buffer, node)


def test_buffer_for_node_creates_separate_listeners_per_node(monkeypatch):
    monkeypatch.setattr(tf_listener_proxy, "Buffer", FakeBuffer)
    monkeypatch.setattr(tf_listener_proxy, "TransformListener", FakeTransformListener)

    first_node = object()
    second_node = object()

    first_buffer = TFListenerProxy().buffer_for_node(first_node)
    second_buffer = TFListenerProxy().buffer_for_node(second_node)

    assert first_buffer is not second_buffer
    assert FakeTransformListener.calls == [
        (first_buffer, first_node),
        (second_buffer, second_node),
    ]


def test_clear_node_removes_only_that_node_listener(monkeypatch):
    monkeypatch.setattr(tf_listener_proxy, "Buffer", FakeBuffer)
    monkeypatch.setattr(tf_listener_proxy, "TransformListener", FakeTransformListener)

    first_node = object()
    second_node = object()
    proxy = TFListenerProxy()

    old_first_buffer = proxy.buffer_for_node(first_node)
    second_buffer = proxy.buffer_for_node(second_node)

    proxy.clear_node(first_node)

    new_first_buffer = proxy.buffer_for_node(first_node)

    assert new_first_buffer is not old_first_buffer
    assert proxy.buffer_for_node(second_node) is second_buffer


def test_clear_removes_cached_listeners_from_existing_proxy(monkeypatch):
    monkeypatch.setattr(tf_listener_proxy, "Buffer", FakeBuffer)
    monkeypatch.setattr(tf_listener_proxy, "TransformListener", FakeTransformListener)

    node = object()
    proxy = TFListenerProxy()
    old_buffer = proxy.buffer_for_node(node)

    TFListenerProxy.clear()

    assert proxy.buffer_for_node(node) is not old_buffer
