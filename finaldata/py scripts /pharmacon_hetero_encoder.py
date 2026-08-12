"""
Pharmacon biological branch -- heterogeneous graph encoder.

Uses PyTorch Geometric's HeteroData / HeteroConv (both PyG-native, no extra
deps beyond torch + torch_geometric).

Node types:
    - drug
    - protein
    - pathway

Edge types (each is a distinct (src_type, relation_name, dst_type) triple):
    - ('drug', 'ddi', 'drug')                      -- ~963 relation types, RGCNConv + basis decomposition
    - ('drug', 'targets', 'protein')                -- drug-protein interactions
    - ('protein', 'interacts', 'protein')            -- protein-protein interactions
    - ('protein', 'in', 'pathway')                   -- pathway membership
    - ('drug', 'metabolite_targets', 'protein')      -- 222 metabolite-protein interactions

Only the drug-drug relation needs RGCNConv's multi-relation machinery, since
that's the only edge type with a large (~963) relation vocabulary -- everything
else is treated as a single relation (existence of an interaction), using a
plain SAGEConv-style layer instead of forcing basis decomposition where it
isn't needed.

This file defines ONLY the encoder (per current scope) -- gating,
cross-attention, and the MLP classifier are separate, later work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, SAGEConv, HeteroConv


class PharmaconHeteroEncoder(nn.Module):
    """
    Two-layer heterogeneous encoder producing final embeddings for every
    node (drug, protein, pathway) in the graph, using relation-aware
    message-passing.

    Args:
        in_channels_dict: e.g. {'drug': 768, 'protein': 256, 'pathway': 64}
        hidden_channels: dimensionality used for the intermediate layer
        out_channels: dimensionality of the FINAL node embeddings
                      (this is what feeds into the gating/cross-attention
                      step later -- not built here)
        num_ddi_relations: number of distinct drug-drug interaction types
                            (~963 in your case)
        num_bases: basis-decomposition parameter for the drug-drug RGCNConv
                   layer -- keeps parameter count tractable at ~963 relations.
                   Without this, RGCNConv would need a full d x d weight
                   matrix PER relation type, which is not feasible at this
                   relation count. Lower num_bases = more parameter sharing
                   across relations = fewer parameters, but less relation-
                   specific expressiveness. Decagon-style thinking: this is
                   the direct analogue of their "effective sharing of model
                   parameters across edge types" (Section 4.1 of the paper).
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int,
        out_channels: int,
        num_ddi_relations: int,
        num_bases: int = 30,
    ):
        super().__init__()

        self.in_channels_dict = in_channels_dict
        self.num_ddi_relations = num_ddi_relations

        # ---- Layer 1 ----
        # HeteroConv wraps a DIFFERENT conv module per edge type, then combines
        # the outputs for each node type (default: sum) after one message-passing
        # round. This is the PyG-native way to do exactly what Decagon's Eq. (1)
        # describes conceptually -- a different transform per relation type,
        # aggregated per node.
        self.conv1 = HeteroConv({
            ('drug', 'ddi', 'drug'): RGCNConv(
                in_channels_dict['drug'],
                hidden_channels,
                num_relations=num_ddi_relations,
                num_bases=num_bases,
            ),
            ('drug', 'targets', 'protein'): SAGEConv(
                (in_channels_dict['drug'], in_channels_dict['protein']),
                hidden_channels,
            ),
            ('protein', 'interacts', 'protein'): SAGEConv(
                in_channels_dict['protein'],
                hidden_channels,
            ),
            ('protein', 'in', 'pathway'): SAGEConv(
                (in_channels_dict['protein'], in_channels_dict['pathway']),
                hidden_channels,
            ),
            ('drug', 'metabolite_targets', 'protein'): SAGEConv(
                (in_channels_dict['drug'], in_channels_dict['protein']),
                hidden_channels,
            ),
        }, aggr='sum')

        # ---- Layer 2 ----
        # Same structure, but now every node type's INPUT dimension is
        # hidden_channels (the output of layer 1), regardless of what its
        # original raw feature dimension was -- this is why HeteroData is
        # useful: each node type keeps its own native dimensionality only
        # until the first layer projects everything into a shared space.
        self.conv2 = HeteroConv({
            ('drug', 'ddi', 'drug'): RGCNConv(
                hidden_channels,
                out_channels,
                num_relations=num_ddi_relations,
                num_bases=num_bases,
            ),
            ('drug', 'targets', 'protein'): SAGEConv(
                (hidden_channels, hidden_channels),
                out_channels,
            ),
            ('protein', 'interacts', 'protein'): SAGEConv(
                hidden_channels,
                out_channels,
            ),
            ('protein', 'in', 'pathway'): SAGEConv(
                (hidden_channels, hidden_channels),
                out_channels,
            ),
            ('drug', 'metabolite_targets', 'protein'): SAGEConv(
                (hidden_channels, hidden_channels),
                out_channels,
            ),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_type_dict=None):
        """
        x_dict:          {'drug': Tensor[num_drugs, in_dim], 'protein': ..., 'pathway': ...}
        edge_index_dict: {('drug','ddi','drug'): LongTensor[2, num_edges], ...}
        edge_type_dict:  {('drug','ddi','drug'): LongTensor[num_edges]}  -- ONLY
                         needed for the ddi relation, since that's the only
                         edge type RGCNConv is used for.

        Returns: dict of final node embeddings, same keys as x_dict,
                 each shaped [num_nodes_of_that_type, out_channels]
        """
        # PyG's HeteroConv requires extra per-edge-type kwargs to be passed
        # under a name ending in '_dict' (it strips that suffix internally
        # and routes each entry to the matching edge type's conv module).
        extra_kwargs = {}
        if edge_type_dict is not None:
            extra_kwargs['edge_type_dict'] = edge_type_dict

        x_dict = self.conv1(x_dict, edge_index_dict, **extra_kwargs)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict, **extra_kwargs)
        # no activation on the final layer -- these are the embeddings that
        # feed into gating/cross-attention next, not a classification logit

        return x_dict


if __name__ == "__main__":
    # --- Minimal smoke test with dummy tensors, just to confirm shapes work ---
    # This does NOT use real data -- it's here so the model's structural
    # correctness can be verified before wiring in real tensors tomorrow.

    num_drugs, num_proteins, num_pathways = 645, 500, 50
    num_ddi_relations = 963

    in_channels_dict = {'drug': 768, 'protein': 256, 'pathway': 64}
    hidden_channels = 128
    out_channels = 64

    x_dict = {
        'drug': torch.randn(num_drugs, in_channels_dict['drug']),
        'protein': torch.randn(num_proteins, in_channels_dict['protein']),
        'pathway': torch.randn(num_pathways, in_channels_dict['pathway']),
    }

    edge_index_dict = {
        ('drug', 'ddi', 'drug'): torch.randint(0, num_drugs, (2, 2000)),
        ('drug', 'targets', 'protein'): torch.stack([
            torch.randint(0, num_drugs, (500,)),
            torch.randint(0, num_proteins, (500,)),
        ]),
        ('protein', 'interacts', 'protein'): torch.randint(0, num_proteins, (2, 3000)),
        ('protein', 'in', 'pathway'): torch.stack([
            torch.randint(0, num_proteins, (800,)),
            torch.randint(0, num_pathways, (800,)),
        ]),
        ('drug', 'metabolite_targets', 'protein'): torch.stack([
            torch.randint(0, num_drugs, (222,)),
            torch.randint(0, num_proteins, (222,)),
        ]),
    }
    edge_type_dict = {
        ('drug', 'ddi', 'drug'): torch.randint(0, num_ddi_relations, (2000,)),
    }

    model = PharmaconHeteroEncoder(
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_ddi_relations=num_ddi_relations,
        num_bases=30,
    )

    out = model(x_dict, edge_index_dict, edge_type_dict)

    print("Forward pass succeeded. Output shapes:")
    for node_type, emb in out.items():
        print(f"  {node_type}: {emb.shape}")
