"""
Keeping copies of a model in different processes in step: the streams of messages they
publish, and the record of how far each copy caught up with every publisher.
"""

from __future__ import annotations

from krrood_synchronization._version import __version__
