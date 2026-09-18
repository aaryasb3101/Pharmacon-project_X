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

        self.self_attention = MultiHeadSelfAttention(  #creates an attention component
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(   #creates a feed forward component
            d_model=d_model, #feed forward is applied independently to every token position
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model) #creates layer normalization over 768-D representation (happens before attention)
        self.norm2 = nn.LayerNorm(d_model) #happens before feed-forward

        self.dropout1 = nn.Dropout(dropout) #dropout after attention
        self.dropout2 = nn.Dropout(dropout) #dropout after feed-forward

    def forward(
        self,
        x,
        attention_mask=None,
    ):

        residual = x #saving the original input

        attention_output, attention_weights = self.self_attention(
            self.norm1(x), #normalize the input and send it to self.self_attention
            attention_mask,#which gives us attention output(transformed vector after attention) and attention weights(matrix for each head is LXL dimensional where L is number of tokens and we get different matrix for each SMILE )
        )

        x = residual + self.dropout1(attention_output) #residual/skip connection

        residual = x #save the new residual 

        ff_output = self.feed_forward(
            self.norm2(x) #normalize and give input to self.feed_forward which expands from 768>3072>768
        )

        x = residual + self.dropout2(ff_output) #new residual=original FFN input+FFN output

        return x, attention_weights