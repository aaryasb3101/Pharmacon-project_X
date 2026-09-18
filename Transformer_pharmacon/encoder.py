import torch.nn as nn

from encoder_layer import EncoderLayer #importing the encoder layers


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple Encoder Layers.

    Input:
        x : (B, L, d_model)

    Output:
        token_embeddings : (B, L, d_model)
        attention_maps : List of attention matrices from each layer
    """

    def __init__(
        self,
        num_layers: int, #no. of layers=2
        d_model: int,    #768
        num_heads: int,  #no. of attention heads=8 (768/8=96) so each attention head operates on head dimension of 96
        d_ff: int = 3072,#hidden dimmension so we expand 768>3072>768
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(  #we create a pytorch modulelist because we need pytorch to track the parametres for these
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers) #creates copies of encoder layers, since our model has 2 layers it builds 2 copies
            ]
        )

    def forward(   #forward function 
        self,
        x,
        attention_mask=None,
    ):

        attention_maps = [] #create an empty python list to store attention map from each encoder layer

        for layer in self.layers:
            #our layer returns 2 things
            x, attention_map = layer( 
                x,              #updated token representation
                attention_mask, #attention produced by that layer.
            )
            #attention maps is a matrix showing attention weights
            #that is how much each token attends to every other token
            attention_maps.append(attention_map)

        return x, attention_maps