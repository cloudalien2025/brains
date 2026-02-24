from __future__ import annotations

import sys
from importlib.machinery import SourcelessFileLoader
from importlib.util import spec_from_loader
from pathlib import Path


def load_pyc(module_name: str, file_path: str) -> bool:
    cache_tag = sys.implementation.cache_tag
    if not cache_tag:
        return False
    pyc_path = Path(file_path).with_name("__pycache__") / f"{Path(file_path).stem}.{cache_tag}.pyc"
    if not pyc_path.exists():
        return False
    loader = SourcelessFileLoader(module_name, str(pyc_path))
    spec = spec_from_loader(module_name, loader)
    module = sys.modules[module_name]
    module.__file__ = str(pyc_path)
    module.__spec__ = spec
    loader.exec_module(module)
    return True
