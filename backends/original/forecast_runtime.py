"""Spectrum transaction hooks for the configurable directional forecast release path."""

from __future__ import annotations

import torch

from forecast_controller import DirectionalForecastController


def patch_runtime_module(module, *, sample_channels=32, actual_steps=None) -> None:
    cls = module.SpectrumH3Runtime
    if getattr(cls, "_h3_directional_forecast_runtime", False):
        return
    original_init = cls.__init__
    original_start_run = cls.start_run
    original_begin_step = cls.begin_step

    def observe_actual_metadata(self, run_id, step_id, call_id, feature):
        """Archive only branch metadata; directional forecast never calls Spectrum's predictor.

        The accepted directional forecast controller owns the two tail tensors that its local
        directional formula actually consumes.  Keeping Spectrum's additional
        eight full final-block tensors is dead D2H work and dead system memory.
        A one-element-per-branch placeholder preserves Spectrum's transactional
        topology/label validation and accounting without changing model math.
        """
        step = self._require_step(run_id, step_id)
        call = step.calls[int(call_id)]
        if step.mode != "actual":
            raise RuntimeError("actual H3 feature observed during a forecast-only step")
        if tuple(feature.shape) != call.expected_shape:
            raise RuntimeError(
                f"actual H3 feature shape {tuple(feature.shape)} does not match {call.expected_shape}"
            )
        metadata = torch.zeros(
            (feature.shape[0], 1, 1), dtype=feature.dtype, device="cpu"
        )
        call.observed_actual = True
        step.actual_records.append(module._ActualRecord(metadata, call.labels))

    def init(self, config):
        original_init(self, config)
        self.configured_actual_steps = frozenset(actual_steps or ())
        self.forecast_controller = DirectionalForecastController(
            sample_channels=sample_channels,
            actual_steps=actual_steps,
        )
        self.profile_exporters = dict(getattr(self, "profile_exporters", {}))
        self.profile_exporters["directional_forecast"] = self.forecast_controller.export

    def start_run(self, sigmas, sampler_name, **kwargs):
        run_id = original_start_run(self, sigmas, sampler_name, **kwargs)
        # Forecast schedules are calibrated for 20 steps.  Smoke/warm-up runs
        # and arbitrary step totals fall back to all-actual execution instead
        # of silently stretching a 20-step approximation onto another solver.
        self.forecast_controller.actual_steps = (
            self.configured_actual_steps
            if self.stats.total_steps == 20
            else frozenset(range(self.stats.total_steps))
        )
        self.forecast_controller.reset(run_id, self.stats.total_steps)
        return run_id

    def begin_step(self, timestep):
        decision = original_begin_step(self, timestep)
        if self._step is not None and not self._disabled:
            self._step.mode = "actual"
            self._step.reason = "local directional forecast"
            self._step.adaptive_recompute = False
            decision["actual"] = True
            decision["reason"] = self._step.reason
        return decision

    def commit_forecast(self, run_id, step_id, call_id):
        step = self._require_step(run_id, step_id)
        call = step.calls[int(call_id)]
        if self._history_labels is None or call.labels is None:
            raise RuntimeError("directional forecast release cannot commit a forecast without branch labels")
        try:
            positions = [self._history_labels.index(label) for label in call.labels]
        except ValueError as error:
            raise RuntimeError("directional forecast release conditional branch identity changed") from error
        step.mode = "forecast"
        step.reason = "local directional forecast early exit"
        call.used_forecast = True
        step.used_history_rows.update(int(position) for position in positions)

    cls.__init__ = init
    cls.start_run = start_run
    cls.begin_step = begin_step
    cls.observe_actual = observe_actual_metadata
    cls.commit_forecast = commit_forecast
    cls._h3_directional_forecast_runtime = True
    schedule = sorted(DirectionalForecastController(
        sample_channels=sample_channels,
        actual_steps=actual_steps,
    ).actual_steps)
    print(
        "directional forecast runtime: configurable schedule, "
        f"depth=3 local_directional=True actual_steps={schedule}",
        flush=True,
    )
