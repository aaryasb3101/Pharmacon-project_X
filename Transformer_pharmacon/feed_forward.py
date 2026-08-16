import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Input:
        x : (B, L, d_model)

    Output:
        x : (B, L, d_model)
    """
    #FFN processes each token representation separately, uses the same weights for every position
    def __init__(
        self,
        d_model: int,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(  #first fully connected layer, so (B,L,768)->(B,L,3072)
            d_model,
            d_ff,
        )#nn.linear performs a learned linear transformation y=xW^T+b

        self.activation = nn.GELU() #Gaussian error linear unit as the activation function
                                    #since the first linear layer gives linear transformation, GELU introduces non-linearity
        self.dropout1 = nn.Dropout(dropout) #first dropout after the activation function

        self.fc2 = nn.Linear( #creates the second linear layer (B,L,3072)->(B,L,768)
            d_ff,
            d_model,
        )

        self.dropout2 = nn.Dropout(dropout) #second dropout

    def forward(self, x): #defines what happens when the input goes through the FFN

        x = self.fc1(x)

        x = self.activation(x)

        x = self.dropout1(x)

        x = self.fc2(x)

        x = self.dropout2(x)

        return x