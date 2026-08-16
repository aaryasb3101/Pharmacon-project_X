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

    x = x.view(*x.shape[:-1], -1, 2) # groups into pairs, so for 96 dimension 48 pairs of 2

    x1 = x[..., 0] #define variables x1 and x2 that loads each pair
    x2 = x[..., 1]

    x = torch.stack((-x2, x1), dim=-1) #rotates the pair (x0,x1)->(-x1,x0)

    return x.flatten(-2) #merge the rotated pairs back to 96 features


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply Rotary Positional Embedding.

    q, k:
        (B, H, L, D)

    cos, sin:
        (1, 1, L, D)
    """
    #Take Q/K, rotate its feature pairs, and combine that rotation with position-dependent sine/cosine values.
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
                    0,  #torch.arrange (0, dim, 2) gives 0,2,4...94, so we get 48 values cause we're working with pairs
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

        self._build_cache(          #builds initial cache and precomputes the sine and cosine values
            max_position_embeddings,
            device="cpu",
            dtype=torch.float32,
        )

    def _build_cache(  #this function generates the positional information for a particular sequence length
        self,
        seq_len,
        device,
        dtype,
    ):

        self.max_seq_len_cached = seq_len #stores current cache size

        positions = torch.arange(   #generates token positions, if seq_len=5 then positions=[0,1,2,3,4]
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(   #combines token positions+rotary frequencies so position x frequency
            positions,
            self.inv_freq,
        )

        emb = torch.repeat_interleave(  #since we originally calculated 48 frequency values, this duplicates each one so that we haev 96 values for 96 features
            freqs,
            repeats=2,
            dim=-1,
        )
        #generate position dependent sine and cosine values with resulting shape (1,1,L,96) so we can broadcastover batches and heads and dont have to create separate copies for batch/head
        cos = emb.cos()[None, None, :, :] 
        sin = emb.sin()[None, None, :, :]

        self.register_buffer(     #store cos values
            "cos_cached",
            cos.to(dtype),
            persistent=False,
        )

        self.register_buffer(     #store sin values
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
            seq_len > self.max_seq_len_cached #if SMILES string is greater than max_seq_length, rebuild cache
            or self.cos_cached.device != q.device #if cache and Q are not at the same place CPU/GPU, rebuild or move appropriately
        ):

            self._build_cache(
                seq_len,
                device=q.device,
                dtype=q.dtype,
            )

        cos = self.cos_cached[:, :, :seq_len].to(  #take only the required positions, if your SMILES is 40 tokens but cache has 512 positions we slice positions 0->39
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