import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv
from torch_geometric.utils import dense_to_sparse

class TIN(nn.Module):
    def __init__(self, hidden_dim):
        super(TIN, self).__init__()
        self.hidden_dim = hidden_dim
        print(f"TIN initialized with hidden_dim: {hidden_dim}")

        # Define residual connections and LayerNorm layers
        self.residual_layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))

        self.GatedGCN = GatedGCN(hidden_dim, hidden_dim)

        # Fusion layer
        self.lstm = nn.LSTM(self.hidden_dim*2, self.hidden_dim, 2, batch_first=True,
                            bidirectional=True)

        # MLP
        self.feature_fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))

    def forward(self, h_feature, h_syn_ori, h_syn_feature, h_sem_ori, h_sem_feature, adj_sem_ori, adj_sem_gcn):
        print(f"TIN forward - h_feature shape: {h_feature.shape}")

        # (batch_size, sequence_length, hidden_dim)
        assert h_feature.shape == h_syn_feature.shape == h_sem_ori.shape == h_sem_feature.shape

        # residual layer
        h_syn_origin = self.residual_layer1(h_feature + h_syn_ori)
        h_syn_feature = self.residual_layer2(h_feature + h_syn_feature)
        h_sem_origin = self.residual_layer3(h_feature + h_sem_ori)
        h_sem_feature = self.residual_layer4(h_feature + h_sem_feature)

        h_syn_origin, h_syn_feature = self.GatedGCN(h_syn_origin, h_syn_feature, adj_sem_ori, adj_sem_gcn)
        h_sem_origin, h_sem_feature = self.GatedGCN(h_sem_origin, h_sem_feature, adj_sem_ori, adj_sem_gcn)

        concat = torch.cat([h_syn_feature, h_sem_feature], dim=2)
        output, _ = self.lstm(concat)
        h_fusion = self.feature_fusion(output)

        return h_fusion

class FeatureStacking(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureStacking, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input1, input2):
        stacked_input = torch.stack([input1, input2], dim=3)
        pooled_input, _ = torch.max(stacked_input, dim=3, keepdim=True)
        output = pooled_input.squeeze(3)
        return output

class GatedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, gated_layers=2):
        super(GatedGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gated_layers = gated_layers
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv3 = GatedGraphConv(self.hidden_dim, self.gated_layers)

    def forward(self, input1, input2, adj_sem_ori, adj_sem_gcn):
        print(f"GatedGCN forward - input1 shape: {input1.shape}, input2 shape: {input2.shape}")

        # Build graph data structures
        input1_ = input1.view(-1, self.input_dim)
        input2_ = input2.view(-1, self.input_dim)
        features = torch.cat([input1_, input2_], dim=0)
        batch_size, seq_len, _ = input1.shape
        
        edge_index, _ = dense_to_sparse(torch.ones((seq_len*2, seq_len*2)).to(input1.device))
        edge_attr = compute_cosine_similarity(features, edge_index)

        x = features
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = Multi_Head_S_Pool(x, adj_sem_ori, adj_sem_gcn)
        x = F.relu(self.conv3(x, edge_index))
        
        x = x.view(2, batch_size, seq_len, self.hidden_dim)
        h_fusion_1, h_fusion_2 = x[0], x[1]
        
        return h_fusion_1, h_fusion_2

def Multi_Head_S_Pool(x, adj_sem_ori, adj_sem_gcn):
    # Average pooling
    adj = torch.cat([adj_sem_ori, adj_sem_gcn], dim=0)
    S_Mean = adj.mean(dim=2)  # Compute the mean along the seq_len dimension
    S_Mean = S_Mean.view(x.size(0), -1)  # Reshape to match the dimensions of x

    # Max pooling
    S_Max, _ = adj.max(dim=2)  # Take the max along the seq_len dimension
    S_Max = S_Max.view(x.size(0), -1)  # Reshape to match the dimensions of x

    # Calculate Z_1
    Z_1 = F.relu(x * (1 + S_Mean + S_Max))

    return Z_1

def edge_weight(x, edge_index):
    """Calculate the edge weights, i.e. Euclidean distance"""
    row, col = edge_index
    edge_attr = (x[row] - x[col]).norm(p=2, dim=-1).view(edge_index.size(1), -1)
    return edge_attr

def compute_cosine_similarity(x, edge_index):
    row, col = edge_index
    x_row = x[row]
    x_col = x[col]
    similarity = F.cosine_similarity(x_row, x_col, dim=1)
    min_value = similarity.min()
    max_value = similarity.max()
    similarity = (similarity - min_value) / (max_value - min_value)
    return similarity

def compute_pearson_correlation(x, edge_index):
    mean_x = torch.mean(x, dim=1)
    diff_x = x - mean_x[:, None]
    sum_squared_diff_x = torch.sum(diff_x ** 2, dim=1)
    sqrt_sum_squared_diff_x = torch.sqrt(sum_squared_diff_x)
    product_sqrt_diff_x = sqrt_sum_squared_diff_x[edge_index[0]] * sqrt_sum_squared_diff_x[edge_index[1]]
    sum_multiplied_diff = torch.sum(diff_x[edge_index[0]] * diff_x[edge_index[1]], dim=1)
    pearson_corr = sum_multiplied_diff / product_sqrt_diff_x
    return pearson_corr