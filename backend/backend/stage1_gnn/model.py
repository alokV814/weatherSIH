"""
StormTrace AI - PyTorch Spherical ST-GNN Model 
Combines Multi-Head Graph Attention Network (GATv2) with Gated Recurrent Unit (GRU) / Temporal Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SphericalGATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_features // heads
        self.W_src = nn.Linear(in_features, out_features, bias=False)
        self.W_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(self, x, edge_index):
        N = x.size(0)
        h_src = self.W_src(x).view(N, self.heads, self.head_dim)
        h_dst = self.W_dst(x).view(N, self.heads, self.head_dim)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        cat_feat = h_src[src_idx] + h_dst[dst_idx]
        scores = (self.leaky_relu(cat_feat) * self.attn_vec).sum(dim=-1)
        alpha = torch.exp(scores - scores.max())
        denom = torch.zeros(N, self.heads, device=x.device).scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(alpha), alpha) + 1e-8
        alpha = alpha / denom[dst_idx]
        
        msg = h_src[src_idx] * alpha.unsqueeze(-1)
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device).scatter_add_(0, dst_idx.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)
        return F.elu(out.view(N, -1))

class PyTorchSTGNNModel(nn.Module):
    """
    Spatio-Temporal GNN combining GATv2 Spatial Graph Attention with GRU Temporal Modeling.
    """
    def __init__(self, in_channels=6, hidden_dim=64, out_channels=2):
        super().__init__()
        self.gat1 = SphericalGATv2Layer(in_channels, hidden_dim)
        self.gat2 = SphericalGATv2Layer(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_track = nn.Linear(hidden_dim, out_channels) # [lat_delta, lon_delta]
        self.fc_intensity = nn.Linear(hidden_dim, 1) # intensity prediction

    def forward(self, x_seq, edge_index):
        # x_seq: [T, N, in_channels]
        T, N, C = x_seq.shape
        spatial_embeddings = []

        for t in range(T):
            h = self.gat1(x_seq[t], edge_index)
            h = self.gat2(h, edge_index)
            spatial_embeddings.append(h)

        spatial_seq = torch.stack(spatial_embeddings, dim=0) # [T, N, hidden_dim]
        graph_pooled = spatial_seq.mean(dim=1).unsqueeze(0) # [1, T, hidden_dim]

        gru_out, _ = self.gru(graph_pooled) # [1, T, hidden_dim]
        gru_out = gru_out.squeeze(0) # [T, hidden_dim]

        track_pred = self.fc_track(gru_out)
        intensity_pred = self.fc_intensity(gru_out)
        return track_pred, intensity_pred
