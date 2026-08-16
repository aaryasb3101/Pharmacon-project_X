"""
Pharmacon biological branch -- heterogeneous graph encoder.

Node types (4, each fully independent -- metabolite info is NEVER folded
into the drug node's features anywhere in this graph):
    - drug
    - metabolite
    - protein
    - pathway

Edge types:
    - ('drug', 'ddi', 'drug')                  -- ~963 relation types, RGCNConv + basis decomposition
    - ('drug', 'targets', 'protein')            -- drug-protein interactions (binary)
    - ('protein', 'interacts', 'protein')        -- protein-protein interactions (binary)
    - ('protein', 'in', 'pathway')               -- pathway membership (binary)
    - ('drug', 'produces', 'metabolite')         -- which metabolite belongs to which parent drug
    - ('metabolite', 'targets', 'protein')       -- 222 metabolite-protein interactions (binary)

Confirmed design decisions (mentor-approved):
    - metabolite and pathway do NOT need 3+ edge types each -- the "each
      node type needs >=3 edges" rule does not strictly apply
    - all non-DDI relations are binary (exists / doesn't) -- no per-relation
      weighting needed there
    - SAGEConv for all binary-relation edges (chosen over GCNConv for its
      separate self/neighbor transform, chosen over GATConv for simplicity
      given time constraints -- GATConv is a reasonable future upgrade if a
      given relation shows meaningful within-relation neighbor variation)
    - HeteroConv aggregation = sum
    - 2 layers
    - num_bases < num_relations for RGCNConv (exact value still open/tunable,
      defaulted to 30 here)
    - no special handling for the 222 metabolite-protein edges specifically
    - no interaction-severity weighting (explicitly out of scope for now --
      no severity data currently available)

This file defines ONLY the encoder -- gating, cross-attention, and the MLP
classifier are separate, later work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, SAGEConv, HeteroConv


class PharmaconHeteroEncoder(nn.Module):
    """
    Two-layer heterogeneous encoder producing final embeddings for every
    node (drug, metabolite, protein, pathway) in the graph.

    Args:
        in_channels_dict: e.g. {'drug': 768, 'metabolite': 768, 'protein': 256, 'pathway': 64}
        hidden_channels: dimensionality used for the intermediate layer
        out_channels: dimensionality of the FINAL node embeddings (feeds into
                      gating/cross-attention later -- not built here)
        num_ddi_relations: number of distinct drug-drug interaction types (~963)
        num_bases: basis-decomposition parameter for the drug-drug RGCNConv layer.
                   Must be < num_ddi_relations. Keeps parameter count tractable --
                   without it, RGCNConv needs a full d x d matrix PER relation,
                   infeasible at ~963 relations. This is the direct analogue of
                   Decagon's "effective sharing of model parameters across edge
                   types" (paper Section 4.1). Exact value still open/tunable --
                   defaulted to 30, flagged as unresolved with mentor.
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

        assert num_bases < num_ddi_relations, \
            "num_bases must be less than num_ddi_relations (mentor-confirmed constraint)"

        self.in_channels_dict = in_channels_dict
        self.num_ddi_relations = num_ddi_relations

        # ---- Layer 1 ----
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
            ('drug', 'produces', 'metabolite'): SAGEConv(
                (in_channels_dict['drug'], in_channels_dict['metabolite']),
                hidden_channels,
            ),
            ('metabolite', 'targets', 'protein'): SAGEConv(
                (in_channels_dict['metabolite'], in_channels_dict['protein']),
                hidden_channels,
            ),
        }, aggr='sum')

        # ---- Layer 2 ----
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
            ('drug', 'produces', 'metabolite'): SAGEConv(
                (hidden_channels, hidden_channels),
                out_channels,
            ),
            ('metabolite', 'targets', 'protein'): SAGEConv(
                (hidden_channels, hidden_channels),
                out_channels,
            ),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_type_dict=None):
        """
        x_dict:          {'drug': Tensor[num_drugs, in_dim], 'metabolite': ...,
                           'protein': ..., 'pathway': ...}
        edge_index_dict: {('drug','ddi','drug'): LongTensor[2, num_edges], ...}
        edge_type_dict:  {('drug','ddi','drug'): LongTensor[num_edges]}  -- ONLY
                         needed for the ddi relation (the only RGCNConv edge type)

        Returns: dict of final node embeddings, same keys as x_dict,
                 each shaped [num_nodes_of_that_type, out_channels]
        """
        extra_kwargs = {}
        if edge_type_dict is not None:
            extra_kwargs['edge_type_dict'] = edge_type_dict

        x_dict = self.conv1(x_dict, edge_index_dict, **extra_kwargs)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict, **extra_kwargs)
        # no activation on the final layer -- these are embeddings, not logits

        return x_dict


if __name__ == "__main__":
    num_drugs, num_metabolites, num_proteins, num_pathways = 645, 557, 500, 50
    num_ddi_relations = 963

    in_channels_dict = {'drug': 768, 'metabolite': 768, 'protein': 256, 'pathway': 64}
    hidden_channels = 128
    out_channels = 64

    x_dict = {
        'drug': torch.randn(num_drugs, in_channels_dict['drug']),
        'metabolite': torch.randn(num_metabolites, in_channels_dict['metabolite']),
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
        ('drug', 'produces', 'metabolite'): torch.stack([
            torch.randint(0, num_drugs, (557,)),
            torch.randint(0, num_metabolites, (557,)),
        ]),
        ('metabolite', 'targets', 'protein'): torch.stack([
            torch.randint(0, num_metabolites, (222,)),
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
