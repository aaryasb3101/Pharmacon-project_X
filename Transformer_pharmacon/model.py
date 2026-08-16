import torch.nn as nn

from embedding import SmilesEmbedding
from encoder import TransformerEncoder


class SmilesTransformer(nn.Module):
    """
    Custom SMILES Transformer.

    Input:
        input_ids      : (B, L)
        attention_mask : (B, L)

    Returns:
        cls_embedding    : (B, d_model)
        token_embeddings : (B, L, d_model)
        attention_maps   : List of attention maps
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Embedding
        
        self.embedding = SmilesEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_id=pad_id,
            max_len=max_len,
            dropout=dropout,
        )

        # Transformer Encoder

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Final LayerNorm

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        # Embedding
        x = self.embedding(input_ids)

        # Transformer Encoder
        token_embeddings, attention_maps = self.encoder(
            x,
            attention_mask,
        )

        # Final normalization
        token_embeddings = self.output_norm(
            token_embeddings
        )

        # CLS Pooling
        cls_embedding = token_embeddings[:, 0, :]

        return (
            cls_embedding,
            token_embeddings,
            attention_maps,
        )