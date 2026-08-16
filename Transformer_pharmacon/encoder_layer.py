import torch.nn as nn

from attention import MultiHeadSelfAttention
from feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """
    One Transformer Encoder Layer (Pre-LayerNorm)

    Input:
        x : (B, L, d_model)

    Output:
        x : (B, L, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attention_mask=None,
    ):

        # -----------------------------
        # Pre-LayerNorm + Self-Attention
        # -----------------------------

        residual = x

        attention_output, attention_weights = self.self_attention(
            self.norm1(x),
            attention_mask,
        )

        x = residual + self.dropout1(attention_output)

        residual = x

        ff_output = self.feed_forward(
            self.norm2(x)
        )

        x = residual + self.dropout2(ff_output)

        return x, attention_weights