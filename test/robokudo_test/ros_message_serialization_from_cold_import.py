"""
Serializes a ROS2 message with robokudo's annotation codecs in a fresh process that
imports nothing else that registers ROS serializers, and prints the type name the
serialized message carries.

Run as a subprocess by :mod:`test_cas_annotation_codecs`, whose process has every
serializer imported already.
"""

# %% serializing

from geometry_msgs.msg import Point

from krrood.adapters.json_field import JSONField

from robokudo.io.cas_annotation_codecs import krrood_to_json


def main() -> None:
    """
    Prints the type name of a ROS2 message serialized by the annotation codecs.
    """
    print(krrood_to_json(Point(x=1.0))[JSONField.TYPE])


if __name__ == "__main__":
    main()
