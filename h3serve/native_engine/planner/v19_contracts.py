"""Release and quality contracts for the V19 final-candidate research path."""

from __future__ import annotations

from dataclasses import dataclass
import math


V19_POLICY_ID = "h3_v19_human_aligned_budgeted_adaptive_inference"
V19_CONTRACT_SCHEMA = "h3_v19_contract_v1"


class V19ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class V19HumanRiskVector:
    """Non-compensating upper bounds for Human-visible failure mechanisms."""

    prompt_adherence: float = 0.0
    contact_causality: float = 0.0
    trajectory_continuity: float = 0.0
    temporal_clarity: float = 0.0
    identity_binding: float = 0.0
    audio_integrity: float = 0.0
    anomaly: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.as_tuple()
        ):
            raise V19ContractError("V19 Human risks must be finite and non-negative")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.prompt_adherence,
            self.contact_causality,
            self.trajectory_continuity,
            self.temporal_clarity,
            self.identity_binding,
            self.audio_integrity,
            self.anomaly,
        )

    def __add__(self, other: "V19HumanRiskVector") -> "V19HumanRiskVector":
        return V19HumanRiskVector(
            *(left + right for left, right in zip(self.as_tuple(), other.as_tuple()))
        )

    def within(self, limits: "V19HumanRiskVector") -> bool:
        """Every dimension must pass; good dimensions never erase a failure."""

        return all(value <= limit for value, limit in zip(self.as_tuple(), limits.as_tuple()))


@dataclass(frozen=True, slots=True)
class V19TrajectoryDebt:
    """Small state carried across denoising actions; Dense does not erase it."""

    consecutive_forecasts: int = 0
    forecast_debt: float = 0.0
    sparse_mass_deficit: float = 0.0
    audio_debt: float = 0.0
    last_refresh_step: int | None = None

    def __post_init__(self) -> None:
        if self.consecutive_forecasts < 0:
            raise V19ContractError("consecutive forecast count cannot be negative")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.forecast_debt,
                self.sparse_mass_deficit,
                self.audio_debt,
            )
        ):
            raise V19ContractError("trajectory debt must be finite and non-negative")
        if self.last_refresh_step is not None and self.last_refresh_step < 0:
            raise V19ContractError("refresh step cannot be negative")

    def within(self, limits: "V19TrajectoryDebt") -> bool:
        """Debt dimensions are hard limits and never offset one another."""

        return (
            self.consecutive_forecasts <= limits.consecutive_forecasts
            and self.forecast_debt <= limits.forecast_debt
            and self.sparse_mass_deficit <= limits.sparse_mass_deficit
            and self.audio_debt <= limits.audio_debt
        )

    def as_pareto_tuple(self) -> tuple[float, ...]:
        """Return monotone debt dimensions; absolute refresh position is metadata."""

        return (
            float(self.consecutive_forecasts),
            self.forecast_debt,
            self.sparse_mass_deficit,
            self.audio_debt,
        )


@dataclass(frozen=True, slots=True)
class V19ParetoObjectiveVector:
    """Complete non-compensating objective for one executed trajectory.

    Peak VRAM and both terminal/maximum trajectory debt matter when several
    approximate techniques are coupled.  Omitting them would allow a plan to
    remain on the reported frontier merely because its Human-risk UCB and
    latency match another plan, even when it consumes more memory or carries
    strictly more unresolved approximation debt.
    """

    cost_p90_ms: float
    peak_vram_gib: float
    human_risk: V19HumanRiskVector
    terminal_debt: V19TrajectoryDebt
    maximum_debt: V19TrajectoryDebt

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.cost_p90_ms)
            or self.cost_p90_ms < 0.0
            or not math.isfinite(self.peak_vram_gib)
            or self.peak_vram_gib < 0.0
        ):
            raise V19ContractError(
                "V19 Pareto cost and peak VRAM must be finite and non-negative"
            )

    def approximation_risk_tuple(self) -> tuple[float, ...]:
        """Human risk and trajectory debt; no dimension compensates another."""

        return (
            *self.human_risk.as_tuple(),
            *self.terminal_debt.as_pareto_tuple(),
            *self.maximum_debt.as_pareto_tuple(),
        )

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.cost_p90_ms,
            self.peak_vram_gib,
            *self.approximation_risk_tuple(),
        )

    def dominates(self, other: "V19ParetoObjectiveVector") -> bool:
        left = self.as_tuple()
        right = other.as_tuple()
        return all(a <= b for a, b in zip(left, right)) and any(
            a < b for a, b in zip(left, right)
        )


@dataclass(frozen=True, slots=True)
class V19InputCapabilityContract:
    """Dense-supported inputs remain usable even outside acceleration evidence."""

    service_families: tuple[str, ...] = ("first_last", "reference")
    model_variants: tuple[str, ...] = ("base", "lora")
    maximum_reference_images: int = 9
    maximum_reference_audio: int = 3
    maximum_reference_videos: int = 3
    accepts_unbounded_prompt_length: bool = True
    ood_policy: str = "accept_and_fallback"

    def __post_init__(self) -> None:
        if self.ood_policy != "accept_and_fallback":
            raise V19ContractError("V19 OOD requests must be accepted and fail closed")
        if min(
            self.maximum_reference_images,
            self.maximum_reference_audio,
            self.maximum_reference_videos,
        ) < 0:
            raise V19ContractError("reference capability cannot be negative")

    def accepts(
        self,
        *,
        service_family: str,
        model_variant: str,
        reference_images: int = 0,
        reference_audio: int = 0,
        reference_videos: int = 0,
    ) -> bool:
        return (
            service_family in self.service_families
            and model_variant in self.model_variants
            and 0 <= reference_images <= self.maximum_reference_images
            and 0 <= reference_audio <= self.maximum_reference_audio
            and 0 <= reference_videos <= self.maximum_reference_videos
        )


V19_INPUT_CAPABILITY = V19InputCapabilityContract()


__all__ = [
    "V19_CONTRACT_SCHEMA",
    "V19_INPUT_CAPABILITY",
    "V19_POLICY_ID",
    "V19ContractError",
    "V19HumanRiskVector",
    "V19InputCapabilityContract",
    "V19ParetoObjectiveVector",
    "V19TrajectoryDebt",
]
