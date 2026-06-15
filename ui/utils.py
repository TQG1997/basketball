"""Re-export DataFactory from shared/ — single source of truth.

Both src/ (training) and ui/ (inference) import DataFactory from this module.
The canonical implementation lives in shared/__init__.py.
"""

import sys
import os

# Make the project root importable so that 'shared' can be found
_ui_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_ui_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared import DataFactory  # noqa: E402, F401
