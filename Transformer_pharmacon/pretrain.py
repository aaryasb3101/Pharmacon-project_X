import torch
import torch.nn as nn
import torch.nn.functional as F


class ParentMetaboliteContrastiveLoss(nn.Module):
    """
    Contrastive loss for paired parent-drug and metabolite embeddings.

    Positive pair:
        Parent[i] <-> Metabolite[i]

    Negative pairs:
        Parent[i] <-> Metabolite[j], where i != j
    """

    def __init__(self, temperature=0.07): #temparature controls how sharply the similarity scores are distributed
        super().__init__()

        self.temperature = temperature #lower temparature makes similarity distribution sharper

    def forward(self, parent_embeddings, metabolite_embeddings):

        # Normalize embeddings (performs L2 normalization)
        parent_embeddings = F.normalize( 
            parent_embeddings,
            p=2,
            dim=-1, #normalize across teh embedding dimension
        )

        metabolite_embeddings = F.normalize(
            metabolite_embeddings,
            p=2,
            dim=-1,
        )

        # Similarity matrix
        #
        # (B, 768) x (B, 768)
        #        ↓
        #      (B, B)
        #
        # Diagonal = correct parent-metabolite pairs
        similarity = torch.matmul(
            parent_embeddings,
            metabolite_embeddings.T,
        )

        similarity = similarity / self.temperature

        # Correct pair for row i is column i
        labels = torch.arange(
            parent_embeddings.size(0),
            device=parent_embeddings.device,
        )

        # Parent -> Metabolite
        loss_parent_to_metabolite = F.cross_entropy(  #for every parent we ask which metabolite in this batch is the correct one?
            similarity,                               #the model is rewarded when the diagonal similarity is high relative to other values
            labels,
        )

        # Metabolite -> Parent
        loss_metabolite_to_parent = F.cross_entropy(  #now we ask, given this metabolite which is the correct parent
            similarity.T,
            labels,
        )

        # Symmetric contrastive loss (since we do parent->metabolite and metabolite->parent, we have symmetric contrastive loss)
        loss = (              
            loss_parent_to_metabolite
            + loss_metabolite_to_parent
        ) / 2

        return loss