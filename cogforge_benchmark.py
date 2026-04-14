"""
Teammate-friendly entrypoint for CogForge benchmark.

This file re-exports everything from the canonical script:
`cogforge_benchmark (1).py`
so collaborators can import/run without relying on a filename that contains spaces/parentheses.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SRC = Path(__file__).with_name("cogforge_benchmark (1).py")
_SPEC = spec_from_file_location("cogforge_benchmark_source", _SRC)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load benchmark source file: {_SRC}")

_MOD = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

# Re-export public names so teammates can use:
#   from cogforge_benchmark import cogforge_benchmark
# and all task functions/datasets.
for _name in dir(_MOD):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MOD, _name)

