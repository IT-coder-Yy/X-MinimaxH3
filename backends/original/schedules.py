"""User-facing production quality/speed schedules for a 20-step RES run."""

from __future__ import annotations


PRESETS: dict[str, tuple[int, ...]] = {
    # Validated default balance point.
    "balanced": (0, 1, 2, 3, 4, 8, 12, 16, 19),
    # Conservative 12/8 refresh schedule.
    "quality": (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
    # Reference path: no forecast approximation.
    "full": tuple(range(20)),
}


def parse_actual_steps(*, preset: str, custom: str | None) -> tuple[int, ...]:
    """Resolve a preset or explicit list without silently changing its budget."""
    if custom:
        try:
            steps = tuple(sorted({int(value.strip()) for value in custom.split(",") if value.strip()}))
        except ValueError as error:
            raise ValueError("actual steps must be comma-separated integers") from error
    else:
        try:
            steps = PRESETS[preset]
        except KeyError as error:
            raise ValueError(f"unknown quality preset: {preset}") from error
    if not steps:
        raise ValueError("at least one actual step is required")
    if steps[0] != 0 or steps[-1] != 19:
        raise ValueError("20-step release schedules must include steps 0 and 19")
    if any(step < 0 or step >= 20 for step in steps):
        raise ValueError("actual step indices must be in [0, 19]")
    return steps


def schedule_label(steps: tuple[int, ...]) -> str:
    return f"{len(steps)}a{20 - len(steps)}f"
