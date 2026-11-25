"""Clean-v1 entry point for the audited, original custom Scratch Transformer."""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load P1's inspected implementation under a distinct module name. The clean
# wrapper itself is also called ``scratch.model`` when run as a script, so a
# normal package import here would create a circular/self import.
_SOURCE = PROJECT_ROOT / "scratch" / "model.py"
_SPEC = importlib.util.spec_from_file_location("_clean_v1_p1_scratch_model", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load original Scratch Transformer source: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FeedForward = _MODULE.FeedForward
MultiHeadAttention = _MODULE.MultiHeadAttention
PositionalEmbedding = _MODULE.PositionalEmbedding
ScaledDotProductAttention = _MODULE.ScaledDotProductAttention
ScratchTransformer = _MODULE.ScratchTransformer
TokenEmbedding = _MODULE.TokenEmbedding
TransformerEncoderBlock = _MODULE.TransformerEncoderBlock

__all__ = [
    "FeedForward", "MultiHeadAttention", "PositionalEmbedding",
    "ScaledDotProductAttention", "ScratchTransformer", "TokenEmbedding",
    "TransformerEncoderBlock",
]
