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

    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature

    def forward(self, parent_embeddings, metabolite_embeddings):

        # Normalize embeddings
        parent_embeddings = F.normalize(
            parent_embeddings,
            p=2,
            dim=-1,
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
        loss_parent_to_metabolite = F.cross_entropy(
            similarity,
            labels,
        )

        # Metabolite -> Parent
        loss_metabolite_to_parent = F.cross_entropy(
            similarity.T,
            labels,
        )

        # Symmetric contrastive loss
        loss = (
            loss_parent_to_metabolite
            + loss_metabolite_to_parent
        ) / 2

        return loss