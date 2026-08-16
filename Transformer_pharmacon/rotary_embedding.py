import torch
import torch.nn as nn


def rotate_half(x):
    """
    Rotate pairs of features.

    Example:
    [x0, x1, x2, x3]
        ->
    [-x1, x0, -x3, x2]
    """

    x = x.view(*x.shape[:-1], -1, 2)

    x1 = x[..., 0]
    x2 = x[..., 1]

    x = torch.stack((-x2, x1), dim=-1)

    return x.flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply Rotary Positional Embedding.

    q, k:
        (B, H, L, D)

    cos, sin:
        (1, 1, L, D)
    """

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoFormer style)
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=512,
        base=10000,
    ):
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_seq_len_cached = max_position_embeddings

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

        self._build_cache(
            max_position_embeddings,
            device="cpu",
            dtype=torch.float32,
        )

    def _build_cache(
        self,
        seq_len,
        device,
        dtype,
    ):

        self.max_seq_len_cached = seq_len

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(
            positions,
            self.inv_freq,
        )

        emb = torch.repeat_interleave(
            freqs,
            repeats=2,
            dim=-1,
        )

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        self.register_buffer(
            "cos_cached",
            cos.to(dtype),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin.to(dtype),
            persistent=False,
        )

    def forward(
        self,
        q,
        k,
    ):

        seq_len = q.size(-2)

        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != q.device
        ):

            self._build_cache(
                seq_len,
                device=q.device,
                dtype=q.dtype,
            )

        cos = self.cos_cached[:, :, :seq_len].to(
            device=q.device,
            dtype=q.dtype,
        )

        sin = self.sin_cached[:, :, :seq_len].to(
            device=q.device,
            dtype=q.dtype,
        )

        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
        )

        return q, k