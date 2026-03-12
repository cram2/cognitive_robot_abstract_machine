"""Tests for CRAM action Pydantic models."""

from llmr.workflows.models.cram_gen_models import (
    Cutting,
    Mixing,
    PickingUp,
    Placing,
    Pouring,
    Stirring,
)


class TestCuttingModel:
    def test_required_fields(self) -> None:
        c = Cutting(obj_to_be_cut="apple", utensil="knife", action_verb="cut")
        assert c.obj_to_be_cut == "apple"
        assert c.utensil == "knife"

    def test_optional_fields_default_none(self) -> None:
        c = Cutting(obj_to_be_cut="apple", utensil="knife", action_verb="cut")
        assert c.amount is None
        assert c.unit is None
        assert c.cram_plan is None

    def test_optional_amount_provided(self) -> None:
        c = Cutting(obj_to_be_cut="apple", utensil="knife", action_verb="cut", amount=2.0)
        assert c.amount == 2.0


class TestPickingUpModel:
    def test_construction(self) -> None:
        p = PickingUp(obj_to_be_grabbed="pan", action_verb="pick")
        assert p.obj_to_be_grabbed == "pan"

    def test_optional_location(self) -> None:
        p = PickingUp(obj_to_be_grabbed="pan", action_verb="pick", location="drawer")
        assert p.location == "drawer"


class TestPouringModel:
    def test_construction(self) -> None:
        p = Pouring(stuff="water", source="jug", goal="glass", action_verb="pour")
        assert p.goal == "glass"
        assert p.source == "jug"
        assert p.stuff == "water"

    def test_optional_amount(self) -> None:
        p = Pouring(stuff="water", source="jug", goal="glass", action_verb="pour", amount=200.0)
        assert p.amount == 200.0


class TestMixingModel:
    def test_content_list(self) -> None:
        m = Mixing(content=["flour", "eggs", "milk"], action_verb="mix")
        assert len(m.content) == 3
        assert "flour" in m.content


class TestStirringModel:
    def test_construction(self) -> None:
        s = Stirring(action_verb="stir", content=["soup"])
        assert s.action_verb == "stir"
        assert s.content == ["soup"]


class TestPlacingModel:
    def test_minimal_construction(self) -> None:
        p = Placing(obj_to_be_put="bowl", action_verb="place")
        assert p.obj_to_be_put == "bowl"
        assert p.location is None
