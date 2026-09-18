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

        if d_model % num_heads != 0: #we gotta check if our dimension are divisible by no. of heads casue every head must have same dimension
            raise ValueError("d_model must be divisible by num_heads")

        #save configuration
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Query, Key, Value projections (we create 3 learnable linear transformations)
        #Attention compares Q with K to determine relevance, then uses those weights to combine V.
        self.W_q = nn.Linear(d_model, d_model) #query: what info am i looking for
        self.W_k = nn.Linear(d_model, d_model) #key: what info do i have
        self.W_v = nn.Linear(d_model, d_model) #value: what info should i pass along

        # Output projection(after all 8 heads have produced outputs, they a combined back into (B,L,768))
        self.W_o = nn.Linear(d_model, d_model) #linear transformation again

        # Rotary Positional Embedding
        self.rope = RotaryEmbedding(
            dim=self.head_dim, #RoPE operates on the per-head representation
            max_position_embeddings=512, #RoPE implementation supports positions upto 512
        )

        self.dropout = nn.Dropout(dropout) #creates dropout

    def split_heads(self, x): #splits the 768-D features across 8 heads
        """
        (B, L, d_model)
            ->
        (B, H, L, head_dim)
        """

        B, L, _ = x.shape

        return (
            x.view(B, L, self.num_heads, self.head_dim) #built a tensor (B,L,no, of heads,dimension of head)
            .transpose(1, 2) #swap dimension 1 and 2 so (B,L,num_heads,num_dimensions)->(B,num_heads,L,num_dimension)
        )                    #why swap? Because attention calculations are easier when the head dimension is separated: Batch->Heads->Token->Features per head

    def combine_heads(self, x): #reverse the process
        """
        (B, H, L, head_dim)
            ->
        (B, L, d_model)
        """

        B, _, L, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous() #makes sure the tensor's memory layout is suitable for the next operation
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

        # Split into 8 heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply RoPE (only applied to Q and K, because positional information is incorporated into the mechanism that determines which tokens attend to which positions.)
        Q, K = self.rope(Q, K)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) #transpose the last 2 dimensions of K so it becomes (B,8,96,L)
                                                      #matrix multiplication of  Q(B,8,L,96) and K(B,8,96,L) to get attention score matrix(B,8,L,L)
        scores *= self.head_dim ** -0.5 #scaling, scores/(96)^1/2

        # Padding mask
        if attention_mask is not None: #if mask is provided apply it
            mask = attention_mask[:, None, None, :] #reshape the mask so it can broadcast across batch,heads,query positions, key positions
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min) #replace the masked position to extreme negative numbers so after softmax is applied they become 0
                                                                              #So the model effectively gives zero attention to padding tokens.

        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1) #apply softmax and produce normalized probabilities/weights from attention scores
        attention_weights = self.dropout(attention_weights) #dropout to the attention_weights

        # Context
        context = torch.matmul(attention_weights, V) #matrix multiplication gives (B,8,L,96) so each head gives (B,L,96)
                                                     #so essentially Take the values from the tokens I'm attending to, weighted by how much attention I give them

        # Merge heads
        context = self.combine_heads(context) #merge the heads

        # Final projection
        out = self.W_o(context) #concantenated heads go through learned output projection
        out = self.dropout(out) #final dropout

        return out, attention_weights