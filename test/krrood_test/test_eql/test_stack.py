"""
Tests for StackFrame, CallStack, and the stack-query methods on InferenceExplanation.

Covers:
  - StackFrame.from_frame_info: class method vs free function
  - CallStack.filter: site-packages removal and package narrowing
  - CallStack.root_frame_in: outermost matching frame
  - CallStack.classes / functions / is_from_method
  - InferenceExplanation.frame_count
  - InferenceExplanation.is_triggered_from_method
  - InferenceExplanation.triggering_classes
  - InferenceExplanation.triggering_functions
  - InferenceExplanation.root_frame_in
"""

import inspect
from dataclasses import dataclass

import pytest

from krrood.entity_query_language._stack import CallStack, StackFrame
from krrood.entity_query_language.explanation.explanation import explain_inference
from krrood.entity_query_language.factories import entity, inference
from krrood.symbol_graph.symbol_graph import Symbol


@dataclass(unsafe_hash=True)
class Person(Symbol):
    name: str


# ---------------------------------------------------------------------------
# StackFrame
# ---------------------------------------------------------------------------


def test_stack_frame_is_method_true():
    frame = StackFrame("f.py", 1, "m", None, Person, None, None)
    assert frame.is_method is True


def test_stack_frame_is_method_false():
    frame = StackFrame("f.py", 1, "f", None, None, None, None)
    assert frame.is_method is False


def test_stack_frame_from_frame_info_free_function():
    fi = inspect.stack()[0]
    frame = StackFrame.from_frame_info(fi)

    assert frame.function_name == "test_stack_frame_from_frame_info_free_function"
    assert frame.class_object is None
    assert frame.is_method is False
    assert frame.lineno > 0
    assert "test_stack.py" in frame.filename
    assert frame.module_name is not None


def test_stack_frame_from_frame_info_class_method():
    captured: list[StackFrame] = []

    class Inner:
        def capture(self):
            fi = inspect.stack()[0]
            captured.append(StackFrame.from_frame_info(fi))

    Inner().capture()

    frame = captured[0]
    assert frame.class_object is Inner
    assert frame.is_method is True
    assert frame.function_name == "capture"


def test_stack_frame_from_frame_info_classmethod():
    captured: list[StackFrame] = []

    class Inner:
        @classmethod
        def capture(cls):
            fi = inspect.stack()[0]
            captured.append(StackFrame.from_frame_info(fi))

    Inner.capture()

    frame = captured[0]
    assert frame.class_object is Inner
    assert frame.is_method is True


def test_stack_frame_code_snippet_stripped():
    fi = inspect.stack()[0]
    frame = StackFrame.from_frame_info(fi)
    assert frame.code_snippet is not None
    assert frame.code_snippet == frame.code_snippet.strip()


# ---------------------------------------------------------------------------
# CallStack
# ---------------------------------------------------------------------------


def _make_frame(**kwargs) -> StackFrame:
    defaults = dict(
        filename="f.py",
        lineno=1,
        function_name="f",
        code_snippet=None,
        class_object=None,
        function_object=None,
        module_name=None,
    )
    defaults.update(kwargs)
    return StackFrame(**defaults)


def test_call_stack_len():
    stack = CallStack([_make_frame() for _ in range(5)])
    assert len(stack) == 5


def test_call_stack_iter():
    frames = [_make_frame(lineno=i) for i in range(3)]
    assert list(CallStack(frames)) == frames


def test_call_stack_filter_removes_site_packages():
    frames = [
        _make_frame(
            filename="/usr/lib/python3/site-packages/foo.py", module_name="foo"
        ),
        _make_frame(filename="/my/project/bar.py", module_name="project.bar"),
    ]
    filtered = CallStack(frames).filter()
    assert len(filtered) == 1
    assert filtered.frames[0].filename == "/my/project/bar.py"


def test_call_stack_filter_removes_dist_packages():
    frames = [
        _make_frame(filename="/usr/lib/dist-packages/pkg.py"),
        _make_frame(filename="/app/mycode.py"),
    ]
    filtered = CallStack(frames).filter()
    assert len(filtered) == 1


def test_call_stack_filter_by_package():
    frames = [
        _make_frame(filename="/project/krrood/core.py"),
        _make_frame(filename="/project/tests/test_foo.py"),
    ]
    filtered = CallStack(frames).filter(package="krrood")
    assert len(filtered) == 1
    assert "krrood" in filtered.frames[0].filename


def test_call_stack_filter_returns_new_instance():
    stack = CallStack([_make_frame()])
    filtered = stack.filter()
    assert filtered is not stack


def test_call_stack_root_frame_in_returns_outermost():
    frames = [
        _make_frame(function_name="innermost", module_name="mylib.core"),
        _make_frame(function_name="middle", module_name="mylib.high"),
        _make_frame(function_name="outermost", module_name="tests"),
    ]
    root = CallStack(frames).root_frame_in("mylib")
    assert root is not None
    assert root.function_name == "middle"  # last in list that matches mylib


def test_call_stack_root_frame_in_no_match():
    frames = [_make_frame(module_name="tests")]
    assert CallStack(frames).root_frame_in("mylib") is None


def test_call_stack_root_frame_in_none_module_name_skipped():
    frames = [
        _make_frame(module_name=None),
        _make_frame(module_name="mylib.x"),
    ]
    root = CallStack(frames).root_frame_in("mylib")
    assert root is not None
    assert root.module_name == "mylib.x"


def test_call_stack_classes_distinct_ordered():
    class A:
        pass

    class B:
        pass

    frames = [
        _make_frame(class_object=A),
        _make_frame(class_object=B),
        _make_frame(class_object=A),  # duplicate — should not appear twice
        _make_frame(class_object=None),
    ]
    classes = CallStack(frames).classes()
    assert classes == [A, B]


def test_call_stack_functions_distinct_ordered():
    def f():
        pass

    def g():
        pass

    frames = [
        _make_frame(function_object=f),
        _make_frame(function_object=g),
        _make_frame(function_object=f),  # duplicate
    ]
    fns = CallStack(frames).functions()
    assert fns == [f, g]


def test_call_stack_is_from_method_true():
    frames = [_make_frame(), _make_frame(class_object=Person)]
    assert CallStack(frames).is_from_method() is True


def test_call_stack_is_from_method_false():
    frames = [_make_frame(), _make_frame()]
    assert CallStack(frames).is_from_method() is False


# ---------------------------------------------------------------------------
# InferenceExplanation stack query methods
# ---------------------------------------------------------------------------


def _infer_person(name: str):
    person_inf = inference(Person)
    return entity(person_inf(name=name))


def test_frame_count_positive():
    query = _infer_person("Alice")
    alice = list(query.evaluate())[0]
    explanation = explain_inference(alice)
    assert explanation is not None
    assert explanation.frame_count > 0
    assert explanation.frame_count == len(explanation.stack)


def test_is_triggered_from_method_true_when_called_from_class():
    class QueryBuilder:
        def build(self):
            person_inf = inference(Person)
            return entity(person_inf(name="Bob"))

    query = QueryBuilder().build()
    bob = list(query.evaluate())[0]
    explanation = explain_inference(bob)
    assert explanation is not None
    assert explanation.is_triggered_from_method() is True


def test_triggering_classes_contains_calling_class():
    class QueryBuilder:
        def build(self):
            person_inf = inference(Person)
            return entity(person_inf(name="Carol"))

    query = QueryBuilder().build()
    carol = list(query.evaluate())[0]
    explanation = explain_inference(carol)
    assert explanation is not None
    classes = explanation.triggering_classes()
    assert QueryBuilder in classes


def test_triggering_classes_returns_list_of_types():
    class QueryBuilder:
        def build(self):
            person_inf = inference(Person)
            return entity(person_inf(name="Dan"))

    query = QueryBuilder().build()
    dan = list(query.evaluate())[0]
    explanation = explain_inference(dan)
    assert explanation is not None
    classes = explanation.triggering_classes()
    assert isinstance(classes, list)
    assert all(isinstance(c, type) for c in classes)


def test_triggering_functions_returns_list_of_callables():
    query = _infer_person("Eve")
    eve = list(query.evaluate())[0]
    explanation = explain_inference(eve)
    assert explanation is not None
    functions = explanation.triggering_functions()
    assert isinstance(functions, list)
    # Module-level functions that are captured should be callable
    assert all(callable(f) for f in functions)


def test_triggering_functions_contains_module_level_helper():
    # _infer_person is a module-level function, so it must appear
    query = _infer_person("Frank")
    frank = list(query.evaluate())[0]
    explanation = explain_inference(frank)
    assert explanation is not None
    functions = explanation.triggering_functions()
    assert _infer_person in functions


def test_root_frame_in_finds_frame_in_test_module():
    query = _infer_person("Grace")
    grace = list(query.evaluate())[0]
    explanation = explain_inference(grace)
    assert explanation is not None
    # "test_stack" appears in the module name of frames from this file
    root = explanation.root_frame_in("test_stack")
    assert root is not None
    assert "test_stack" in root.filename


def test_root_frame_in_returns_none_for_unknown_package():
    query = _infer_person("Hank")
    hank = list(query.evaluate())[0]
    explanation = explain_inference(hank)
    assert explanation is not None
    assert explanation.root_frame_in("nonexistent_package_xyz") is None


def test_root_frame_in_outermost_wins():
    """
    The outermost frame in the package is returned, not the innermost.
    """

    def inner():
        person_inf = inference(Person)
        return entity(person_inf(name="Iris"))

    def outer():
        return inner()

    query = outer()
    iris = list(query.evaluate())[0]
    explanation = explain_inference(iris)
    assert explanation is not None
    root = explanation.root_frame_in("test_stack")
    # The outermost test_stack frame is the test function itself, not inner()
    assert root is not None
    assert root.function_name == "test_root_frame_in_outermost_wins"
