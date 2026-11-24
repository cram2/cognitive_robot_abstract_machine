from enum import StrEnum


class JSONField(StrEnum):
    """
    The keys that give the JSON of a serialized object its structure, whatever class it
    was serialized from.

    What a class writes about itself beyond these is its own, and belongs to the
    serializer that writes it.
    """

    TYPE = "__json_type__"
    """
    The fully qualified name of the class the object was serialized from.
    """

    IS_CLASS = "__is_class__"
    """
    Whether what was serialized is a class rather than an instance of one, which
    :attr:`TYPE` cannot tell apart because the type of a class is usually just ``type``.
    """

    COLLECTION_TYPE = "__collection_type__"
    """
    The fully qualified name of the class of a ``list``-like field, so that reading it
    restores that class instead of defaulting to ``list``.
    """

    ITEMS = "__items__"
    """
    The serialized items of a ``list``-like field.
    """

    KEYS = "keys"
    """
    The serialized keys of a ``dict`` field.
    """

    VALUES = "values"
    """
    The serialized values of a ``dict`` field, in the order of :attr:`KEYS`.
    """
