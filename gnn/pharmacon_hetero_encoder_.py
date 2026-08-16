# hetero gnn encoder for the bio branch - drug/metabolite/protein/pathway nodes
# ddi edges use rgcn (too many rel types for normal conv), everything else uses sage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, SAGEConv, HeteroConv


class PharmaconHeteroEncoder(nn.Module):
    # 2 layer encoder, outputs one embedding per node (drug/met/protein/pathway)
    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int,
        out_channels: int,
        num_ddi_relations: int,
        num_bases: int = 30,
    ):
        super().__init__()

        # if num_bases >= num_relations, basis decomp does nothing, no point using rgcn then
        assert num_bases < num_ddi_relations, \
            "num_bases must be less than num_ddi_relations (mentor-confirmed constraint)"

        self.in_channels_dict = in_channels_dict
        self.num_ddi_relations = num_ddi_relations

        # layer 1 - one conv per edge type, heteroconv runs them all + sums per node
        self.conv1 = HeteroConv({
            ('drug', 'ddi', 'drug'): RGCNConv(
                in_channels_dict['drug'],
                hidden_channels,
                num_relations=num_ddi_relations,
                num_bases=num_bases,  # shared weight basis, keeps param count sane at ~963 rels
            ),
            ('drug', 'targets', 'protein'): SAGEConv(
                (in_channels_dict['drug'], in_channels_dict['protein']),  # tuple bc bipartite, diff dims each side
                hidden_channels,
            ),
            ('protein', 'interacts', 'protein'): SAGEConv(
                in_channels_dict['protein'],  # same node type both sides, single dim is fine
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
        }, aggr='sum')  # if a node gets msgs from multiple rel types, just add them up

        # layer 2 - same idea, but now everything's already hidden_channels dim from layer 1
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
        # x_dict = raw node feats per type, edge_index_dict = edges per type
        # edge_type_dict only needed for ddi edges (rgcn needs to know which of the 963 rels)

        # rgcn needs edge_type as extra arg but sageconv doesn't take it at all
        # pyg heteroconv only routes it right if the kwarg name ends in _dict
        extra_kwargs = {}
        if edge_type_dict is not None:
            extra_kwargs['edge_type_dict'] = edge_type_dict

        x_dict = self.conv1(x_dict, edge_index_dict, **extra_kwargs)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # relu so stacking layers actually means smth

        x_dict = self.conv2(x_dict, edge_index_dict, **extra_kwargs)
        # no relu here, this is the final embedding not a logit, dont want to zero stuff out

        return x_dict


if __name__ == "__main__":
    # quick smoke test w dummy data just to check shapes work, not real data
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

    print("fwd pass ok, output shapes:")
    for node_type, emb in out.items():
        print(f"  {node_type}: {emb.shape}")
