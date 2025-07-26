import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torch_geometric.utils import dense_to_sparse

class AdvancedSynGCN(nn.Module):
    """
    Advanced Syntactic GCN with multiple GCN convolution methods
    Similar to TIN upgrade from GCNConv/GatedGraphConv to GATv2Conv
    """

    def __init__(self, emb_dim=768, num_layers=2, gcn_dropout=0.1, gcn_type='gatv2', **kwargs):
        super(AdvancedSynGCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.gcn_type = gcn_type
        self.gcn_dropout = gcn_dropout
        
        # GCN layers based on type
        self.gcn_layers = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.gcn_layers.append(self._create_gcn_layer(gcn_type, input_dim, self.out_dim, **kwargs))
        
        # Layer normalization for better training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.out_dim) for _ in range(self.layers)
        ])
        
        # Residual connections
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.layers)
        ])
        
        # Edge feature learning
        self.edge_encoder = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh()
        )
        
        # Multi-scale processing
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(emb_dim, emb_dim, kernel_size=2**i, padding=2**(i-1))
            for i in range(1, 4)  # 3 scales: 2, 4, 8
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def _create_gcn_layer(self, gcn_type, input_dim, output_dim, **kwargs):
        """Create different types of GCN layers"""
        if gcn_type == 'gatv2':
            return GATv2Conv(input_dim, output_dim, heads=kwargs.get('heads', 8), concat=True)
        elif gcn_type == 'gcn':
            return GCNConv(input_dim, output_dim)
        elif gcn_type == 'sage':
            return SAGEConv(input_dim, output_dim, aggr=kwargs.get('aggr', 'mean'))
        elif gcn_type == 'gin':
            return GINConv(input_dim, output_dim, eps=kwargs.get('eps', 0.0))
        elif gcn_type == 'chebyshev':
            return ChebyshevGCN(input_dim, output_dim, K=kwargs.get('K', 3))
        elif gcn_type == 'dynamic':
            return DynamicGCN(input_dim, output_dim)
        elif gcn_type == 'edge_conv':
            return EdgeConv(input_dim, output_dim)
        elif gcn_type == 'hybrid':
            return HybridGCN(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown GCN type: {gcn_type}")

    def forward(self, adj, inputs):
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Enhanced adjacency matrix processing
        adj_enhanced = self._enhance_adjacency_matrix(adj, inputs)
        
        # Multi-scale feature extraction
        multi_scale_features = self._multi_scale_processing(inputs, adj_enhanced)
        
        # GCN layers with residual connections
        outputs = inputs
        residual_outputs = []
        
        for l in range(self.layers):
            gcn_layer = self.gcn_layers[l]
            
            # Convert dense adjacency to edge_index for PyTorch Geometric layers
            if self.gcn_type in ['gatv2', 'gcn', 'sage', 'hybrid']:
                edge_index, _ = dense_to_sparse(adj_enhanced)
                layer_output = gcn_layer(outputs, edge_index)
            else:
                layer_output = gcn_layer(outputs, adj_enhanced)
            
            # Residual connection
            if l > 0:
                residual_weight = torch.sigmoid(self.residual_weights[l])
                layer_output = residual_weight * layer_output + (1 - residual_weight) * residual_outputs[-1]
            
            # Layer normalization
            layer_output = self.layer_norms[l](layer_output)
            layer_output = F.relu(layer_output)
            
            # Dropout
            if l < self.layers - 1:
                layer_output = F.dropout(layer_output, p=self.gcn_dropout, training=self.training)
            
            outputs = layer_output
            residual_outputs.append(outputs)
        
        # Feature fusion with multi-scale features
        final_outputs = torch.cat([outputs, multi_scale_features], dim=-1)
        
        # Create mask for output
        mask = (adj_enhanced.sum(2) + adj_enhanced.sum(1)).eq(0).unsqueeze(2)
        
        return final_outputs, mask

    def _enhance_adjacency_matrix(self, adj, inputs):
        """Enhance adjacency matrix with learned edge features"""
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Create edge features from node pairs
        node_i = inputs.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j = inputs.unsqueeze(1).expand(-1, seq_len, -1, -1)
        edge_features = torch.cat([node_i, node_j], dim=-1)
        
        # Learn edge weights
        edge_weights = self.edge_encoder(edge_features)
        edge_weights = torch.sigmoid(edge_weights.mean(dim=-1))
        
        # Apply edge weights to adjacency matrix
        enhanced_adj = adj * edge_weights
        
        # Add self-loops and normalize
        enhanced_adj = enhanced_adj + torch.eye(seq_len, device=adj.device).unsqueeze(0)
        enhanced_adj = F.softmax(enhanced_adj, dim=-1)
        
        return enhanced_adj

    def _multi_scale_processing(self, inputs, adj):
        """Multi-scale feature processing"""
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Multi-scale convolution
        conv_inputs = inputs.transpose(1, 2)  # (batch, hidden, seq)
        scale_outputs = []
        
        for conv in self.multi_scale_conv:
            conv_out = conv(conv_inputs)
            conv_out = F.relu(conv_out)
            scale_outputs.append(conv_out.transpose(1, 2))  # (batch, seq, hidden)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        
        # Graph-based aggregation
        graph_enhanced = adj.bmm(multi_scale_features)
        
        # Scale fusion
        fused_features = self.scale_fusion(graph_enhanced)
        
        return fused_features


class GINConv(nn.Module):
    """Graph Isomorphism Network Convolution"""
    
    def __init__(self, input_dim, hidden_dim, eps=0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, adj):
        # GIN convolution: h_v = MLP((1 + eps) * h_v + sum(h_u for u in N(v)))
        neighbor_sum = torch.matmul(adj, x)
        out = (1 + self.eps) * x + neighbor_sum
        return self.mlp(out)


class ChebyshevGCN(nn.Module):
    """Chebyshev Graph Convolutional Network"""
    
    def __init__(self, input_dim, hidden_dim, K=3):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x, adj):
        # Chebyshev polynomial approximation
        Tx_0 = x
        Tx_1 = torch.matmul(adj, x)
        out = torch.matmul(Tx_0, self.weight[0]) + torch.matmul(Tx_1, self.weight[1])
        
        if self.K > 2:
            for k in range(2, self.K):
                Tx_2 = 2 * torch.matmul(adj, Tx_1) - Tx_0
                out += torch.matmul(Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2
                
        return out + self.bias


class DynamicGCN(nn.Module):
    """Dynamic Graph Convolutional Network with learnable edge weights"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.edge_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.gcn = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, base_adj):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Predict dynamic edge weights based on node features
        node_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        node_pairs = torch.cat([node_i, node_j], dim=-1)
        
        edge_weights = self.edge_predictor(node_pairs).squeeze(-1)
        dynamic_adj = base_adj * edge_weights
        
        # Add self-loops and normalize
        dynamic_adj = dynamic_adj + torch.eye(seq_len, device=x.device).unsqueeze(0)
        dynamic_adj = F.softmax(dynamic_adj, dim=-1)
        
        # Apply GCN
        return self.gcn(torch.matmul(dynamic_adj, x))


class EdgeConv(nn.Module):
    """Edge Convolution for learning edge features"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x, adj):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Create edge features from node pairs
        node_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        edge_features = torch.cat([node_i, node_j], dim=-1)
        
        # Process edge features
        edge_processed = self.edge_mlp(edge_features)
        
        # Aggregate using adjacency matrix
        output = torch.matmul(adj, edge_processed)
        return output


class HybridGCN(nn.Module):
    """Hybrid approach combining GATv2 and GIN"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gat = GATv2Conv(input_dim, hidden_dim // 2, heads=4, concat=True)
        self.gin = GINConv(input_dim, hidden_dim // 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x, edge_index):
        # GAT processing
        gat_out = self.gat(x, edge_index)
        
        # GIN processing (need to convert edge_index to adj)
        adj = torch.zeros(x.size(0), x.size(0), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        gin_out = self.gin(x, adj)
        
        # Fusion
        combined = torch.cat([gat_out, gin_out], dim=-1)
        return self.fusion(combined)


# Backward compatibility - original GCN class
class GCN(nn.Module):
    """Original GCN implementation for backward compatibility"""
    
    def __init__(self, emb_dim=768, num_layers=2, gcn_dropout=0.1):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)

    def forward(self, adj, inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](inputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return inputs, mask 