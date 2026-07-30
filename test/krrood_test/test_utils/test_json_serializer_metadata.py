from dataclasses import dataclass, field
from krrood.adapters.json_serializer import to_json, from_json
from krrood.patterns.field_metadata import JSONMetadata


@dataclass
class MetadataTestDataclass:
    public_init: str = "public_init"
    _private_init: str = field(
        default="private_init", metadata=JSONMetadata(serialize=True).as_dict()
    )
    public_non_init: str = field(
        default="public_non_init",
        init=False,
        metadata=JSONMetadata(serialize=True).as_dict(),
    )
    _private_non_init: str = field(
        default="private_non_init",
        init=False,
        metadata=JSONMetadata(serialize=True).as_dict(),
    )

    hidden_public_init: str = field(
        default="hidden", metadata=JSONMetadata(serialize=False).as_dict()
    )
    _hidden_private_init: str = "hidden_private"  # Should be hidden by default


def test_json_metadata_serialization():
    obj = MetadataTestDataclass()
    obj._private_init = "custom_private_init"
    obj.public_non_init = "custom_public_non_init"
    obj._private_non_init = "custom_private_non_init"

    data = to_json(obj)

    assert "public_init" in data
    assert "_private_init" in data
    assert "public_non_init" in data
    assert "_private_non_init" in data

    assert "hidden_public_init" not in data
    assert "_hidden_private_init" not in data

    assert data["public_init"] == "public_init"
    assert data["_private_init"] == "custom_private_init"
    assert data["public_non_init"] == "custom_public_non_init"
    assert data["_private_non_init"] == "custom_private_non_init"

    # Deserialization
    obj2 = from_json(data)
    assert isinstance(obj2, MetadataTestDataclass)
    assert obj2.public_init == "public_init"
    assert obj2._private_init == "custom_private_init"
    assert obj2.public_non_init == "custom_public_non_init"
    assert obj2._private_non_init == "custom_private_non_init"
