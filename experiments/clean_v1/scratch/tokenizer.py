"""Clean-v1 entry point for the audited ScratchTokenizer implementation."""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The clean training script fits this tokenizer on train-manifest text only.
# Load with a distinct name to avoid colliding with this clean wrapper package.
_SOURCE = PROJECT_ROOT / "scratch" / "tokenizer.py"
_SPEC = importlib.util.spec_from_file_location("_clean_v1_p1_scratch_tokenizer", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load original ScratchTokenizer source: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ScratchTokenizer = _MODULE.ScratchTokenizer

__all__ = ["ScratchTokenizer"]
