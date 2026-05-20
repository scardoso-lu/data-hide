"""Compatibility shim for `python main.py` and legacy `import main` callers."""

from __future__ import annotations

import sys

from app import main as _app_main


if __name__ == "__main__":
    _app_main.main()
else:
    sys.modules[__name__] = _app_main
