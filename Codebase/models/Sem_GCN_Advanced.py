import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torch_geometric.utils import dense_to_sparse

class AdvancedSemGCN(nn.Module):
    """
    Advanced Semantic GCN with multiple GCN convolution methods and enhanced attention
    Similar to TIN upgrade from GCNConv/GatedGraphConv to GATv2Conv
    """

    def __init__(self, args, emb_dim=768, num_layers=2, gcn_dropout=0.1, gcn_type='gatv2', **kwargs):
        super(AdvancedSemGCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.gcn_type = gcn_type
        self.gcn_dropout = gcn_dropout
        
        # Enhanced attention heads (increased from default)
        self.attention_heads = max(8, getattr(args, 'attention_heads', 8))
        self.mem_dim = self.args.hidden_dim
        
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
        
        # Enhanced attention mechanism
        self.attn = AdvancedMultiHeadAttention(self.attention_heads, self.mem_dim * 2)
        
        # Global context modeling
        self.global_context = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Position encoding for better sequence modeling
        self.pos_encoding = PositionalEncoding(self.emb_dim, max_len=512)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim)
        )
        
        # Multi-scale attention fusion
        self.multi_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(emb_dim, 4, dropout=0.1, batch_first=True)
            for _ in range(3)  # 3 scales
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

    def forward(self, inputs, encoding, seq_lens):
        tok = encoding
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = seq_lens
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        # Add positional encoding
        inputs = self.pos_encoding(inputs)
        
        gcn_inputs = inputs
        
        # Enhanced attention mechanism
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        
        # Multi-scale attention fusion
        adj_ag = None
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag_new = adj_ag.clone()
        adj_ag_new /= self.attention_heads

        # Enhanced adjacency matrix processing
        for j in range(adj_ag_new.size(0)):
            adj_ag_new[j] -= torch.diag(torch.diag(adj_ag_new[j]))
            adj_ag_new[j] += torch.eye(adj_ag_new[j].size(0)).cuda()
            # Add edge weight normalization
            adj_ag_new[j] = F.softmax(adj_ag_new[j], dim=-1)
        adj_ag_new = mask_ * adj_ag_new

        # Multi-scale attention processing
        multi_scale_features = self._multi_scale_attention_processing(inputs, adj_ag_new)

        # GCN layers with residual connections
        denom_ag = adj_ag_new.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        residual_outputs = []

        for l in range(self.layers):
            gcn_layer = self.gcn_layers[l]
            
            # Convert dense adjacency to edge_index for PyTorch Geometric layers
            if self.gcn_type in ['gatv2', 'gcn', 'sage', 'hybrid']:
                edge_index, _ = dense_to_sparse(adj_ag_new)
                layer_output = gcn_layer(outputs, edge_index)
            else:
                layer_output = gcn_layer(outputs, adj_ag_new)
            
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

        # Global context modeling
        global_outputs, _ = self.global_context(
            outputs, outputs, outputs,
            key_padding_mask=~mask_.squeeze(-1).bool()
        )
        
        # Feature fusion
        fused_outputs = self.feature_fusion(torch.cat([outputs, global_outputs], dim=-1))
        
        # Combine with multi-scale features
        final_outputs = torch.cat([fused_outputs, multi_scale_features], dim=-1)
        
        return final_outputs, adj_ag_new

    def _multi_scale_attention_processing(self, inputs, adj):
        """Multi-scale attention processing"""
        batch_size, seq_len, hidden_dim = inputs.shape
        scale_outputs = []
        
        for attention_layer in self.multi_scale_attention:
            # Apply attention at different scales
            scale_out, _ = attention_layer(
                inputs, inputs, inputs,
                key_padding_mask=~adj.sum(dim=-1).bool()
            )
            scale_outputs.append(scale_out)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        
        # Scale fusion
        fused_features = self.scale_fusion(multi_scale_features)
        
        return fused_features


class AdvancedMultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with relative position encoding"""
    
    def __init__(self, h, d_model, dropout=0.1):
        super(AdvancedMultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)  # Query, Key, Value
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Add relative position encoding
        self.relative_position_encoding = RelativePositionEncoding(self.d_k, max_len=512)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        
        # Apply linear transformations
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, key))]

        # Add relative position encoding
        query = self.relative_position_encoding(query)
        key = self.relative_position_encoding(key)

        attn = improved_attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Concatenate and apply final linear transformation
        attn = attn.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        return attn


class RelativePositionEncoding(nn.Module):
    """Relative Position Encoding for attention"""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create relative position embeddings
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_len - 1, d_model))
        
    def forward(self, x):
        batch_size, num_heads, seq_len, d_model = x.size()
        
        # Create relative position indices
        pos_indices = torch.arange(seq_len, device=x.device)
        rel_pos_indices = pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0)
        rel_pos_indices += self.max_len - 1  # Shift to non-negative indices
        
        # Get relative position embeddings
        rel_pos_emb = self.rel_pos_emb[rel_pos_indices]
        
        # Add to input
        return x + rel_pos_emb.unsqueeze(1)  # Add head dimension


class PositionalEncoding(nn.Module):
    """Positional Encoding for sequence modeling"""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


def improved_attention(query, key, value, mask=None, dropout=None):
    """Improved attention function with better numerical stability"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Use scaled softmax for better numerical stability
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# GCN Layer Implementations
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


# Backward compatibility - original SemGCN class
class SemGCN(nn.Module):
    """Original SemGCN implementation for backward compatibility"""

    def __init__(self, args, emb_dim=768, num_layers=2, gcn_dropout=0.1):
        super(SemGCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.attention_heads = self.args.attention_heads
        self.mem_dim = self.args.hidden_dim
        
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim * 2)

    def forward(self, inputs, encoding, seq_lens):
        tok = encoding
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = seq_lens
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        gcn_inputs = inputs
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None

        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag_new = adj_ag.clone()
        adj_ag_new /= self.attention_heads

        for j in range(adj_ag_new.size(0)):
            adj_ag_new[j] -= torch.diag(torch.diag(adj_ag_new[j]))
            adj_ag_new[j] += torch.eye(adj_ag_new[j].size(0)).cuda()
        adj_ag_new = mask_ * adj_ag_new

        # gcn layer
        denom_ag = adj_ag_new.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs

        for l in range(self.layers):
            Ax = adj_ag_new.bmm(outputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom_ag
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return outputs, adj_ag_new


class MultiHeadAttention(nn.Module):
    """Original MultiHeadAttention for backward compatibility"""

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn 