.. _json_serialization:

Working with JSON: SubclassJSONSerializer
=========================================

`krrood` ships a way to serialize polymorphic JSON data.
It is centered around :class:`krrood.adapters.json_serializer.SubclassJSONSerializer`
and two convenience functions :func:`krrood.adapters.json_serializer.to_json`
and :func:`krrood.adapters.json_serializer.from_json`.

When to use it
--------------
Use ``SubclassJSONSerializer`` whenever you need to persist or exchange
polymorphic dataclass instances (for example a base ``Shape`` with concrete
``Circle`` and ``Rectangle`` subclasses) and you want round‑trip safety without
hand‑crafted ``if/else`` type switches.


.. warning::
    Be aware that due to the limitations of JSON this only works for ONE-TO-ONE/MANY relationships.

How it works (in short)
-----------------------
Each serialized object stores its fully qualified class name under the
``"__json_type__"`` key. During deserialization this type is imported and the
correct subclass is instantiated.

What you implement in your subclass
-----------------------------------
To participate in automatic (de)serialization:

1. Inherit from :class:`krrood.adapters.json_serializer.SubclassJSONSerializer`.
2. Implement ``to_json(self) -> dict`` and include your fields. Always call
   ``super().to_json()`` and merge the result so that ``"__json_type__"`` is
   present.
3. Implement ``@classmethod _from_json(cls, data: dict, **kwargs)`` and return
   an instance of ``cls`` using values from ``data``. This method is invoked by
   the framework after it has resolved the correct subclass.

Minimal example
---------------
.. code-block:: python

   from dataclasses import dataclass
   from krrood.adapters.json_serializer import (
       SubclassJSONSerializer, to_json, from_json
   )

   @dataclass
   class Circle(SubclassJSONSerializer):
       radius: float

       def to_json(self) -> dict:
           base = super().to_json()
           return {**base, "radius": self.radius}

       @classmethod
       def _from_json(cls, data: dict, **kwargs):
           return cls(radius=data["radius"])

   circle = Circle(radius=2.5)
   s = to_json(circle)              # JSON string
   same = from_json(s)              # -> Circle(radius=2.5)

Nested objects and containers
-----------------------------
``from_json`` and the decoder also handle lists and dictionaries recursively.
You can freely nest serializable objects inside containers:

.. code-block:: python

   payload = {
       "title": "shapes",
       "items": [Circle(1.0), Circle(2.0)],
   }

   s = to_json(payload)
   restored = from_json(s)
   # restored["items"] contains Circle instances

Working with dataclasses
------------------------
The library plays well with Python dataclasses. Keep your constructors simple
and let ``_from_json`` mirror your dataclass fields. Prefer short, descriptive
field names and avoid unnecessary nesting.

Type registry for third‑party types
-----------------------------------
If you need to embed types that you do not control, register dedicated
serializers with :class:`krrood.adapters.json_serializer.TypeRegistry`. UUIDs
are built in and already registered.

.. code-block:: python

   from dataclasses import dataclass
   from decimal import Decimal
   from krrood.utils import get_full_class_name
   from krrood.adapters.json_serializer import (
       JSON_TYPE_NAME, TypeRegistry, to_json, from_json
   )

   # 1) Provide pair of functions
   def serialize_decimal(obj: Decimal) -> dict:
       return {JSON_TYPE_NAME: get_full_class_name(type(obj)), "value": str(obj)}

   def deserialize_decimal(data: dict) -> Decimal:
       return Decimal(data["value"])

   # 2) Register once at startup
   TypeRegistry().register(Decimal, serialize_decimal, deserialize_decimal)

   # 3) Now Decimals inside payloads round‑trip automatically
   s = to_json({"price": Decimal("9.99")})
   restored = from_json(s)  # {"price": Decimal("9.99")}



