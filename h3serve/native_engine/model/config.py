"""Static MiniMax H3 dimensions for the single-GPU inference core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class H3CoreConfig:
    """Only model-graph dimensions; serving and memory policy live elsewhere."""

    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_layers: int = 2
    num_heads: int = 56
    head_dim: int = 128
    ffn_size: int = 14336
    video_channels: int = 24
    audio_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    time_input_dim: int = 256
    time_hidden_dim: int = 5376
    time_embedding_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    curve_grid_size: int = 1025
    pruned_curve_dim: int = 8
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        if len(self.patch_size) != 3 or any(v <= 0 for v in self.patch_size):
            raise ValueError("patch_size must contain three positive integers")
        if self.rope_inv_freq_len * 6 > self.head_dim:
            raise ValueError("RoPE dimensions cannot exceed the attention head")
        if self.pruned_curve_dim <= 0 or self.curve_grid_size < 2:
            raise ValueError("invalid pruned AdaLN curve dimensions")

    @property
    def attention_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.video_channels * pt * ph * pw

    @property
    def block_adaln_width(self) -> int:
        return 3 * 6 * self.hidden_size

    @property
    def final_adaln_width(self) -> int:
        return 2 * self.hidden_size

