import math
import torch.nn as nn #importing pytorch's neural network model

from rotary_embedding import RotaryEmbedding


class SmilesEmbedding(nn.Module): #the class inherits from nn.module(base class for pytorch neural network components)
    """
    Embedding layer for SMILES Transformer.

    Input:
        input_ids : (B, L) B:batch size, L:sequence length

    Output:
        embeddings : (B, L, d_model) d_model: model dimension (768)
    """

    def __init__(        #constructor
        self,
        vocab_size: int,  #vocab size=29
        d_model: int,     #d_model=768
        pad_id: int,      #pad_id=0
        max_len: int = 256, 
        dropout: float = 0.1,
    ):
        super().__init__() #initializes the base pytorch module so its parametre and submodules are properly registered

        self.d_model = d_model #store the value in d_model

        # Token embedding
        self.token_embedding = nn.Embedding(  #creates a learnable lookup table with 768x29 learnable parametres
            num_embeddings=vocab_size,        #so each token like 0[PAD] has a vector [0.12,-0.03....768 values] and same for all tokens in vocabulary
            embedding_dim=d_model,            #these vectors learn during training
            padding_idx=pad_id,               
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) #creates a dropout layer

    def forward(self, input_ids): #defines what happens when you pass input into the model
                                  #embedding(input_ids) automatically calls forward(input_ids)
        # (B, L) -> (B, L, d_model)
        x = self.token_embedding(input_ids) #token ids get converted to vectors so each integer token id gets replaced by its 768-D vector

        # Scale embeddings
        x = x * math.sqrt(self.d_model) #scaling, to ensure that the magnitude of embeddings are at a suitable scale

        # Dropout
        x = self.dropout(x)

        return x