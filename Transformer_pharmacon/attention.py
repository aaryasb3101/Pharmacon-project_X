import math
import torch
import torch.nn as nn

from rotary_embedding import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Positional Embeddings (RoPE)

    Input:
        x : (B, L, d_model)

    Output:
        out : (B, L, d_model)
        attention_weights : (B, num_heads, L, L)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Rotary Positional Embedding
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=512,
        )

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        (B, L, d_model)
            ->
        (B, H, L, head_dim)
        """

        B, L, _ = x.shape

        return (
            x.view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    def combine_heads(self, x):
        """
        (B, H, L, head_dim)
            ->
        (B, L, d_model)
        """

        B, _, L, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, L, self.d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):

        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply RoPE
        Q, K = self.rope(Q, K)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores *= self.head_dim ** -0.5

        # Padding mask
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Context
        context = torch.matmul(attention_weights, V)

        # Merge heads
        context = self.combine_heads(context)

        # Final projection
        out = self.W_o(context)
        out = self.dropout(out)

        return out, attention_weights