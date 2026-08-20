"""MiniMax H3 inference components independent from any UI graph runtime.

The package deliberately avoids importing CUDA or the 42 GiB model stack at
module import time.  Construct :class:`NativeH3Engine` only in the GPU worker.
"""

from .engine import NativeGenerationResult, NativeH3Engine, NativeHotH3Engine
from .hot_session import (
    HotSessionCancelled,
    HotSessionRequest,
    HotSessionResult,
    NativeT2AVHotSession,
)

__all__ = [
    "HotSessionRequest",
    "HotSessionResult",
    "HotSessionCancelled",
    "NativeGenerationResult",
    "NativeH3Engine",
    "NativeHotH3Engine",
    "NativeT2AVHotSession",
]
