# hetero gnn encoder for the bio branch - drug/metabolite/protein/pathway nodes
# ddi edges use rgcn (too many rel types for normal conv), everything else uses sage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, SAGEConv, HeteroConv


class PharmaconHeteroEncoder(nn.Module):
    # 2 layer encoder, outputs one embedding per node type (drug/met/protein/pathway)
    def __init__(
        self,
        in_channels_dict: dict,   # raw feature dim per node type - diff sizes, not shared
        hidden_channels: int,    
        out_channels: int,       
        num_ddi_relations: int,   
        num_bases: int = 30,      
    ):
        super().__init__()

        # if num_bases >= num_relations, basis decomp does nothing, no point using rgcn then
        assert num_bases < num_ddi_relations, \
            "num_bases must be less than num_ddi_relations, else no compression happening"

        self.in_channels_dict = in_channels_dict
        self.num_ddi_relations = num_ddi_relations

        # layer 1 
        # one conv per edge type, heteroconv runs them all + sums per node
        self.conv1 = HeteroConv({
            
            # basis decomp = share 30 "base" weight mats instead of 963 full ones (param blowup otherwise)
            ('drug', 'ddi', 'drug'): RGCNConv(
                in_channels_dict['drug'],
                hidden_channels,
                num_relations=num_ddi_relations,
                num_bases=num_bases,
            ),
            # bipartite edge 
            ('drug', 'targets', 'protein'): SAGEConv(
                (in_channels_dict['drug'], in_channels_dict['protein']), # tuple
                hidden_channels,
            ),
            # no tuple needed
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

        # layer 2 
        # same structure, but input dim is now hidden_channels for everyone 
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
        # when you acc run the model on data 
        # x_dict = raw node features per type, edge_index_dict = edges per type
               # edge_type_dict only needed for ddi edges (rgcn needs to know which one it is from the setof the 963 rels)
       
               # rgcn needs edge_type as extra arg- use pyg heteroconv
        extra_kwargs = {}
        if edge_type_dict is not None:
            extra_kwargs['edge_type_dict'] = edge_type_dict

        x_dict = self.conv1(x_dict, edge_index_dict, **extra_kwargs)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # nonlinearity btwn layers, stops collapse into 1 linear layer

        x_dict = self.conv2(x_dict, edge_index_dict, **extra_kwargs)
        # no relu here - final emb goes to downstream decoder, negative vals still useful (e.g. dot product decoder)

        return x_dict
