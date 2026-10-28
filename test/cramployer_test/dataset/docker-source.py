#!/usr/bin/env python3
"""
Report the source checkout passed to Docker without creating containers.
"""

from __future__ import annotations

# %% Imports
import os

# %% Checkout selection
print(os.environ["CRAM_SOURCE"])
