"""Execution-equivalent pipeline improvements around directional forecast's unchanged predictor."""

from __future__ import annotations

import torch


class _AcceptedForecast(Exception):
    def __init__(self, output):
        super().__init__("directional forecast local DCTA early exit")
        self.output = output


def patch_minimax_spectrum_module(module) -> None:
    if getattr(module, "_h3_directional_forecast_wrapper", False):
        return

    def execute_actual(
        executor, inner, runtime, run_id, step_id, call_id, layout, x, timestep,
        context, transformer_options, minimax_payload, kwargs,
    ):
        if len(inner.blocks) < 4:
            raise RuntimeError("directional forecast requires at least four transformer blocks")
        controller = runtime.forecast_controller
        local_options = dict(transformer_options)
        patches_replace = dict(local_options.get("patches_replace") or {})
        replacements = dict(patches_replace.get("dit") or {})
        patches_replace["dit"] = replacements
        local_options["patches_replace"] = patches_replace
        first_key = ("double_block", 0)
        anchor_key = ("double_block", 2)
        last_key = ("double_block", len(inner.blocks) - 1)
        existing_first = replacements.get(first_key)
        existing_anchor = replacements.get(anchor_key)
        existing_last = replacements.get(last_key)
        holder = {}
        observed = False

        def first(args, replacement_context):
            (aa, _), (_, vb) = module.target_segments(layout)
            input_full = args["img"][aa:vb]
            holder["input_sample"] = controller.sample(input_full)
            if not controller.should_forecast(step_id) and controller.needs_cache_release():
                torch.cuda.empty_cache()
            return existing_first(args, replacement_context) if existing_first else replacement_context["original_block"](args)

        def anchor(args, replacement_context):
            output = existing_anchor(args, replacement_context) if existing_anchor else replacement_context["original_block"](args)
            (aa, _), (_, vb) = module.target_segments(layout)
            hidden = output["img"][aa:vb]
            holder["anchor_sample"] = controller.sample(hidden)
            if not controller.should_forecast(step_id):
                anchor_host = torch.empty(hidden.shape, dtype=hidden.dtype, device="cpu", pin_memory=True)
                anchor_host.copy_(hidden.detach(), non_blocking=True)
                holder["anchor_host"] = anchor_host
                return output
            predicted = controller.predict(
                step_id, hidden, holder["input_sample"], holder["anchor_sample"], layout
            )
            sanitized, _event = module._sanitize_prediction(predicted, context.dtype)
            if sanitized is None:
                return output
            state = module._prepare_output_state(
                inner, x[0], x[1], timestep, context, local_options,
                minimax_payload or {}, layout,
            )
            forecast_output = module._execute_forecast(inner, sanitized, state, x[0], x[1])
            runtime.commit_forecast(run_id, step_id, call_id)
            raise _AcceptedForecast(forecast_output)

        def last(args, replacement_context):
            nonlocal observed
            output = existing_last(args, replacement_context) if existing_last else replacement_context["original_block"](args)
            (aa, _), (_, vb) = module.target_segments(layout)
            final = output["img"][aa:vb]
            tail_host = torch.empty(final.shape, dtype=final.dtype, device="cpu", pin_memory=True)
            anchor_host = holder["anchor_host"]
            for start in range(0, final.shape[0], 4096):
                stop = min(start + 4096, final.shape[0])
                anchor_chunk = anchor_host[start:stop].to(device=final.device, non_blocking=True)
                tail_chunk = final[start:stop] - anchor_chunk
                tail_host[start:stop].copy_(tail_chunk, non_blocking=True)
                del anchor_chunk, tail_chunk
            # All future H2D consumers use this same CUDA stream; stream ordering
            # preserves correctness without a host-side synchronize per refresh.
            controller.observe_actual(
                step_id, holder["input_sample"], holder["anchor_sample"], tail_host
            )
            runtime.observe_actual(run_id, step_id, call_id, final.unsqueeze(0))
            observed = True
            return output

        replacements[first_key] = first
        replacements[anchor_key] = anchor
        replacements[last_key] = last
        try:
            result = executor(
                x, timestep, context, local_options,
                minimax_payload=minimax_payload, **kwargs,
            )
        except _AcceptedForecast as accepted:
            return accepted.output
        if not observed:
            raise RuntimeError("directional forecast expected a final actual block observation")
        return result

    module._execute_actual = execute_actual
    module._h3_directional_forecast_wrapper = True
    print("directional forecast Release hook: unchanged depth3 local DCTA + exact pipeline", flush=True)
