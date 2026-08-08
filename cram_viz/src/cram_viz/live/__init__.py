"""
Live mode: stream a running coraplex demo into the web viewer.

See ``cram_viz``'s ``README.md`` ("Live mode") for usage.

.. note:: This ``__init__`` intentionally imports nothing: the runner pulls in
   coraplex and giskardpy, which only exist in a demo environment, while
   :mod:`cram_viz.live.bridge` must stay importable everywhere (tests, the
   plain viewer).
"""
