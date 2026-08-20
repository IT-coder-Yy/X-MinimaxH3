"""LoRA-specific projection of the frozen H3 acceleration scheduler.

The distilled Turbo LoRA owns a short, user-selected sigma trajectory.  It
must therefore never inherit Base-model forecast evaluations.  This adapter
keeps the reviewed Round229 per-step/per-layer Attention allocator, but makes
the no-forecast contract structural instead of relying on a call-site flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from .joint_acceleration import (
    H3JointAccelerationScheduler,
    JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
    JointAccelerationError,
    JointAccelerationPlan,
    JointWorkloadContext,
)


FROZEN_INT8_JOINT_POLICY = (
    JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP
)
LORA_NO_FORECAST_SCHEDULER_ID = "h3_lora_v1_no_forecast_round229"


@dataclass(frozen=True, slots=True)
class H3LoraAccelerationScheduler:
    """Allocate only Attention sparsity on a complete LoRA trajectory."""

    policy_id: str = FROZEN_INT8_JOINT_POLICY
    scheduler_id: str = LORA_NO_FORECAST_SCHEDULER_ID

    def plan(
        self,
        total_steps: int,
        acceleration: float,
        *,
        workload: JointWorkloadContext,
    ) -> JointAccelerationPlan:
        if not 4 <= int(total_steps) <= 10:
            raise JointAccelerationError(
                "LoRA total steps must be an integer inside [4, 10]"
            )
        if workload.model_variant != "lora":
            raise JointAccelerationError(
                "LoRA scheduler requires model_variant=lora"
            )
        plan = H3JointAccelerationScheduler(policy_id=self.policy_id).plan(
            int(total_steps),
            acceleration,
            allow_forecast=False,
            workload=workload,
        )
        expected_steps = tuple(range(int(total_steps)))
        if (
            plan.actual_step_indices != expected_steps
            or plan.forecast_step_indices
            or plan.forecast_allowed
        ):
            raise JointAccelerationError(
                "LoRA scheduler attempted to alter the distilled trajectory"
            )
        schedule = plan.physical_action_schedule()
        if set(schedule) != {
            (step, layer)
            for step in expected_steps
            for layer in range(50)
        }:
            raise JointAccelerationError(
                "LoRA scheduler did not allocate all step/layer cells"
            )
        # Full certificate replay intentionally belongs to release tests.  It
        # resolves the global allocation problem again and would otherwise
        # impose a measurable planning tax on every first-seen online shape.
        # The cheap invariants above remain fail-closed on every request.
        return plan


__all__ = [
    "FROZEN_INT8_JOINT_POLICY",
    "H3LoraAccelerationScheduler",
    "LORA_NO_FORECAST_SCHEDULER_ID",
]
