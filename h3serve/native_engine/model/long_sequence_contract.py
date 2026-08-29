"""Physical capability contract for one streamed Attention cell.

The request planner selects a memory graph for the whole H3 request, while
the V24 scheduler may route individual step/layer cells to different physical
Attention backends.  Release optimizations must therefore be intersected with
the capabilities of the backend resolved for the current cell.  Keeping that
intersection here prevents a request-level HND optimization from leaking into
an NHD Dense cell without duplicating model or scheduler policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


KVLayout = Literal["HND", "NHD"]


@dataclass(frozen=True, slots=True)
class PhysicalLongSequenceContract:
    """Effective layout-sensitive features for one physical backend."""

    kv_layout: KVLayout
    fused_qknorm_hnd_layout: bool
    direct_hnd_fp8_value: bool


def resolve_physical_long_sequence_contract(
    physical: object,
    *,
    compact_kv: bool,
    direct_nhd_kv_requested: bool,
    fused_qknorm_hnd_requested: bool,
    direct_hnd_fp8_value_requested: bool,
) -> PhysicalLongSequenceContract:
    """Intersect request-level optimizations with one backend's ABI.

    Unsupported optional optimizations fall back to the backend's established
    preparation path.  Invalid backend layout declarations fail closed.
    """

    direct_nhd_kv = bool(
        direct_nhd_kv_requested
        and getattr(physical, "supports_direct_nhd_kv", False)
    )
    declared_layout = "NHD" if direct_nhd_kv else getattr(
        physical, "long_sequence_kv_layout", "HND"
    )
    if declared_layout not in ("HND", "NHD"):
        raise ValueError("unsupported long-sequence K/V layout")

    hnd_noncompact = bool(declared_layout == "HND" and not compact_kv)
    return PhysicalLongSequenceContract(
        kv_layout=declared_layout,
        fused_qknorm_hnd_layout=bool(
            hnd_noncompact and fused_qknorm_hnd_requested
        ),
        direct_hnd_fp8_value=bool(
            hnd_noncompact and direct_hnd_fp8_value_requested
        ),
    )


__all__ = [
    "KVLayout",
    "PhysicalLongSequenceContract",
    "resolve_physical_long_sequence_contract",
]
