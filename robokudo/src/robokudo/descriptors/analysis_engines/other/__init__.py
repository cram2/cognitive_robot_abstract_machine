"""Analysis engines initialization module.

This module serves as the initialization point for all analysis engines in the
RoboKudo framework. It provides access to various analysis engine implementations
for different perception tasks.

Available analysis engines include:

* Basic perception pipelines (demo, realsense)
* Query-based processing engines
* Storage and playback engines
* Robot-specific engines (TIAGo, HSR)
* Specialized processing engines (region filtering, object detection)

.. note::
    Each analysis engine is implemented as a separate module and can be
    imported individually based on the application requirements.
"""
