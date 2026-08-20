"""ComfyUI-independent MiniMax H3 conditioning and VAE adapters.

The package deliberately has no import-time PyTorch dependency.  Header
auditing and image geometry therefore remain usable by the service's CPU-only
readiness tooling; tensor libraries are imported only by an actual encode or
decode call.
"""

from .audit import CheckpointAudit, TensorHeader, audit_checkpoint
from .audio_vae import H3AudioVAEAdapter
from .contracts import (
    FrameConditioning,
    KeyframeCondition,
    PreparedKeyframe,
    PreparedReferenceAudio,
    PreparedReferenceVideo,
    ReferenceConditioning,
    ReferenceImageConditioning,
    TextConditioning,
)
from .preprocess import cover_crop_plan, prepare_keyframes, prepare_reference_audios, prepare_reference_images, prepare_reference_videos
from .text import H3Qwen3VLConditioner
from .qwen_quantized import PackedQwen3VLT2AVConditioner, TextEncodingResult
from .video_vae import H3VideoVAEAdapter

__all__ = [
    "CheckpointAudit",
    "FrameConditioning",
    "H3AudioVAEAdapter",
    "H3Qwen3VLConditioner",
    "PackedQwen3VLT2AVConditioner",
    "TextEncodingResult",
    "H3VideoVAEAdapter",
    "KeyframeCondition",
    "PreparedKeyframe",
    "PreparedReferenceAudio",
    "PreparedReferenceVideo",
    "ReferenceConditioning",
    "ReferenceImageConditioning",
    "TensorHeader",
    "TextConditioning",
    "audit_checkpoint",
    "cover_crop_plan",
    "prepare_keyframes",
    "prepare_reference_audios",
    "prepare_reference_images",
    "prepare_reference_videos",
]
