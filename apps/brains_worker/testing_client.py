from __future__ import annotations

from ._pyc_loader import load_pyc

if not load_pyc(__name__, __file__):
    raise ImportError(f"Missing compiled module for {__name__}")
