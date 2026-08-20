from __future__ import annotations

import time

import torch


def patch_forecaster(module) -> None:
    """Use reusable pinned host buffers for Spectrum history transfers."""
    cls = module.HistoryWeightForecaster
    original_update = cls.update

    def pinned_update(self, coordinate, feature, *, take_ownership=False):
        if self.history_storage != "system_ram" or feature.device.type != "cuda":
            return original_update(self, coordinate, feature, take_ownership=take_ownership)
        if not torch.is_tensor(feature) or not feature.dtype.is_floating_point:
            raise ValueError("Spectrum history features must be floating-point tensors")
        shape = tuple(int(value) for value in feature.shape)
        if len(shape) < 2:
            raise ValueError("Spectrum history requires a branch dimension")
        if self._feature_shape is None:
            self._feature_shape, self._feature_dtype = shape, feature.dtype
        elif shape != self._feature_shape or feature.dtype != self._feature_dtype:
            raise ValueError("Spectrum history shape or dtype changed during a run")
        storage_device = torch.device("cpu")
        if self._history_device is None:
            self._history_device = storage_device
        elif storage_device != self._history_device:
            raise ValueError("Spectrum history device changed during a run")

        detached = feature.detach().contiguous().reshape(-1)
        if len(self._history) >= self.max_history:
            archived = self._history.pop(0).feature_flat
            if archived.numel() != detached.numel() or archived.dtype != self._feature_dtype:
                archived = None
        else:
            archived = None
        if archived is None:
            archived = torch.empty(
                detached.numel(), dtype=self._feature_dtype, device="cpu", pin_memory=True
            )
        archived.copy_(detached, non_blocking=True)
        self._history.append(module._HistoryEntry(float(coordinate), archived))
        self._generation += 1
        self._design = self._cholesky = None

    cls.update = pinned_update


def patch_runtime_archive(module) -> None:
    """Archive actual features at Spectrum's transaction boundary using pinned RAM."""
    cls = module.SpectrumH3Runtime
    original_observe_actual = cls.observe_actual

    def pinned_observe_actual(self, run_id, step_id, call_id, feature):
        if self.config.history_storage != "system_ram" or feature.device.type != "cuda":
            return original_observe_actual(self, run_id, step_id, call_id, feature)
        step = self._require_step(run_id, step_id)
        call = step.calls[int(call_id)]
        if step.mode != "actual" or tuple(feature.shape) != call.expected_shape:
            raise RuntimeError("invalid actual feature at Spectrum archive boundary")
        started = time.perf_counter()
        try:
            detached = feature.detach().contiguous()
            archived = torch.empty(
                detached.shape, dtype=detached.dtype, device="cpu", pin_memory=True
            )
            archived.copy_(detached, non_blocking=True)
        finally:
            self.stats.history_archive_seconds += time.perf_counter() - started
        call.observed_actual = True
        step.actual_records.append(module._ActualRecord(archived, call.labels))

    cls.observe_actual = pinned_observe_actual

