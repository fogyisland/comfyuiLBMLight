"""Pytest configuration.

Adds the project root to sys.path so `import lbm_core` and
`import nodes.*` work without needing the project to be installed.

Also prevents pytest from treating the project root as a package
(which would cause the top-level __init__.py — which imports
ComfyUI-dependent modules — to execute).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
