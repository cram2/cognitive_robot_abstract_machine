"""Tests for EntityDescriptionSchema.

Pure Pydantic validation — no fixtures needed.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from llmr.schemas.entities import EntityDescriptionSchema


class TestEntityDescriptionSchema:
    """EntityDescriptionSchema Pydantic model."""

    def test_all_fields_accepted(self) -> None:
        """All fields (name, semantic_type, spatial_context, attributes) accepted."""
        schema = EntityDescriptionSchema(
            name="milk bottle",
            semantic_type="FoodItem",
            spatial_context="on the kitchen counter",
            attributes={"color": "white", "size": "medium"},
        )
        assert schema.name == "milk bottle"
        assert schema.semantic_type == "FoodItem"
        assert schema.spatial_context == "on the kitchen counter"
        assert schema.attributes == {"color": "white", "size": "medium"}

    def test_only_name_required(self) -> None:
        """Only name is required; others default to None."""
        schema = EntityDescriptionSchema(name="table")
        assert schema.name == "table"
        assert schema.semantic_type is None
        assert schema.spatial_context is None
        assert schema.attributes is None

    def test_missing_name_raises_validation_error(self) -> None:
        """Missing name field raises ValidationError."""
        with pytest.raises(ValidationError):
            EntityDescriptionSchema(semantic_type="Surface")

    def test_attributes_defaults_to_none(self) -> None:
        """Attributes field defaults to None when not provided."""
        schema = EntityDescriptionSchema(name="cup")
        assert schema.attributes is None

    def test_round_trip_json(self) -> None:
        """Schema can be serialized to JSON and reconstructed."""
        original = EntityDescriptionSchema(
            name="red cup", semantic_type="Container", attributes={"color": "red"}
        )
        json_str = original.model_dump_json()
        reconstructed = EntityDescriptionSchema.model_validate_json(json_str)
        assert reconstructed.name == original.name
        assert reconstructed.semantic_type == original.semantic_type
        assert reconstructed.attributes == original.attributes

    def test_extra_fields_ignored(self) -> None:
        """Extra fields are ignored (not rejected) by Pydantic."""
        schema = EntityDescriptionSchema(name="object", unknown_field="ignored")
        assert schema.name == "object"
        assert not hasattr(schema, "unknown_field")
