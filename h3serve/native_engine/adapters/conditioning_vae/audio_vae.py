"""MiniMax H3 stereo audio-VAE adapter."""

from __future__ import annotations

from typing import Any, Sequence

AUDIO_LATENT_CHANNELS = 32


def _unwrap_sample(value: Any) -> Any:
    if hasattr(value, "sample") and not callable(value.sample):
        return value.sample
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


class H3AudioVAEAdapter:
    """Bridge H3's [B,32,2,T] latents to a mono-batch DAC decoder."""

    sample_rate = 32000

    def __init__(
        self,
        model: Any,
        *,
        latents_mean: Sequence[float],
        latents_std: Sequence[float],
    ) -> None:
        if not callable(getattr(model, "decode", None)):
            raise TypeError("audio VAE must expose decode()")
        if len(latents_mean) != AUDIO_LATENT_CHANNELS or len(latents_std) != AUDIO_LATENT_CHANNELS:
            raise ValueError("audio VAE requires 32 latent means/stds")
        model_rate = getattr(model, "sample_rate", self.sample_rate)
        if int(model_rate) != self.sample_rate:
            raise ValueError("MiniMax H3 audio VAE must decode at 32000 Hz")
        self.model = model
        self.latents_mean = tuple(float(value) for value in latents_mean)
        self.latents_std = tuple(float(value) for value in latents_std)

    def _stats(self, tensor: Any):
        import torch

        mean = torch.as_tensor(
            self.latents_mean, device=tensor.device, dtype=tensor.dtype
        ).view(1, AUDIO_LATENT_CHANNELS, 1)
        std = torch.as_tensor(
            self.latents_std, device=tensor.device, dtype=tensor.dtype
        ).view(1, AUDIO_LATENT_CHANNELS, 1)
        return mean, std

    def decode(self, normalized_latents: Any) -> tuple[Any, int]:
        """Decode, apply the verified loudness rule, and return stereo audio."""

        import torch

        if normalized_latents.ndim != 4:
            raise ValueError("audio latents must have shape [B,32,2,T]")
        batch, channels, stereo, time = map(int, normalized_latents.shape)
        if channels != AUDIO_LATENT_CHANNELS or stereo != 2:
            raise ValueError("audio latents must have shape [B,32,2,T]")
        flattened = normalized_latents.permute(0, 2, 1, 3).reshape(
            batch * stereo, channels, time
        )
        mean, std = self._stats(flattened)
        decoded = _unwrap_sample(self.model.decode(flattened * std + mean))
        if decoded.ndim == 3 and int(decoded.shape[1]) == 1:
            decoded = decoded[:, 0]
        if decoded.ndim != 2 or int(decoded.shape[0]) != batch * stereo:
            raise ValueError(f"unexpected decoded audio shape {tuple(decoded.shape)}")
        waveform = decoded.reshape(batch, stereo, int(decoded.shape[-1])).float()
        divisor = waveform.std(dim=(1, 2), keepdim=True).mul_(5.0)
        waveform = waveform / divisor.clamp_min_(1.0)
        return waveform.clamp_(-1.0, 1.0).contiguous(), self.sample_rate

    def encode(self, waveform: Any, *, sample_posterior: bool = False) -> Any:
        """Encode stereo reference audio to normalized ``[B,32,2,T]`` rows.

        Both released H3 Audio-VAE implementations are accepted: the compact
        native model directly returns normalized mono-batch latents, while the
        reference wrapper may return a posterior object.
        """

        import torch

        encode = getattr(self.model, "encode", None)
        if not callable(encode):
            raise TypeError("audio VAE does not expose encode()")
        if waveform.ndim != 3 or int(waveform.shape[1]) != 2:
            raise ValueError("waveform must have shape [B,2,samples]")
        batch, stereo, samples = map(int, waveform.shape)
        mono_batch = waveform.reshape(batch * stereo, 1, samples)
        try:
            posterior = encode(mono_batch, return_cpu=False)
        except TypeError:
            posterior = encode(mono_batch)
        posterior = posterior.latent_dist if hasattr(posterior, "latent_dist") else posterior
        if torch.is_tensor(posterior):
            latents = posterior
            if latents.ndim != 3 or int(latents.shape[0]) != batch * stereo:
                raise ValueError(f"unexpected encoded audio shape {tuple(latents.shape)}")
            return latents.reshape(batch, stereo, AUDIO_LATENT_CHANNELS, -1).permute(
                0, 2, 1, 3
            ).contiguous()
        if sample_posterior:
            sample = getattr(posterior, "sample", None)
            if not callable(sample):
                raise TypeError("audio encoder returned no sampleable posterior")
            latents = sample()
        else:
            mode = getattr(posterior, "mode", None)
            if not callable(mode):
                raise TypeError("audio encoder returned no posterior mode")
            latents = mode()
        mean, std = self._stats(latents)
        latents = (latents - mean) / std
        return latents.reshape(batch, stereo, AUDIO_LATENT_CHANNELS, -1).permute(
            0, 2, 1, 3
        ).contiguous()


__all__ = ["AUDIO_LATENT_CHANNELS", "H3AudioVAEAdapter"]
