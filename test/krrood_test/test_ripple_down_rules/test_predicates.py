import pytest

from krrood.ripple_down_rules import *
from .datasets import Drawer, Handle


@pytest.mark.skip(
    "Skipping test_fit_depends_on_predicate as it is not implemented yet."
)
def test_fit_depends_on_predicate() -> None:
    dependsOn.rdr_decorator.fit = True
    dependsOn.rdr_decorator.update_existing_rules = False
    assert any(dependsOn(Drawer, Handle))
