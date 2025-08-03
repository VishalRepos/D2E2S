import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class ImprovedGCN(nn.Module):
    def __init__(self, emb_dim=768, num_layers=3, gcn_dropout=0.1):
        super(ImprovedGCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        
        # Enhanced GCN layers with different convolution types
        self.W = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_weights = nn.ParameterList()
        self.attention_weights = nn.ParameterList()
        
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
            self.layer_norms.append(nn.LayerNorm(input_dim))
            self.residual_weights.append(nn.Parameter(torch.ones(1)))
            self.attention_weights.append(nn.Parameter(torch.ones(1)))
            
        self.gcn_drop = nn.Dropout(gcn_dropout)
        
        # Edge feature learning
        self.edge_encoder = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh()
        )
        
        # Multi-scale feature aggregation
        self.multi_scale_agg = MultiScaleAggregation(emb_dim)
        
        # Graph attention mechanism
        self.graph_attention = GraphAttention(emb_dim, num_heads=8)
        
        # Feature enhancement layer
        self.feature_enhancement = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, adj, inputs):
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Enhanced adjacency matrix processing
        adj_enhanced = self._enhance_adjacency_matrix(adj, inputs)
        
        # Multi-scale feature extraction
        multi_scale_features = self.multi_scale_agg(inputs, adj_enhanced)
        
        # GCN layers with residual connections and attention
        denom = adj_enhanced.sum(2).unsqueeze(2) + 1
        mask = (adj_enhanced.sum(2) + adj_enhanced.sum(1)).eq(0).unsqueeze(2)
        
        outputs = inputs
        residual_outputs = []
        
        for l in range(self.layers):
            # Standard GCN operation
            Ax = adj_enhanced.bmm(outputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](outputs)  # Self loop
            AxW = AxW / denom
            
            # Graph attention enhancement
            attn_outputs = self.graph_attention(outputs, adj_enhanced)
            attn_weight = torch.sigmoid(self.attention_weights[l])
            AxW = attn_weight * AxW + (1 - attn_weight) * attn_outputs
            
            # Residual connection
            if l > 0:
                residual_weight = torch.sigmoid(self.residual_weights[l])
                AxW = residual_weight * AxW + (1 - residual_weight) * residual_outputs[-1]
            
            gAxW = F.relu(AxW)
            gAxW = self.layer_norms[l](gAxW)
            
            if l < self.layers - 1:
                outputs = self.gcn_drop(gAxW)
            else:
                outputs = gAxW
                
            residual_outputs.append(outputs)
        
        # Feature fusion with multi-scale features
        final_outputs = self.feature_enhancement(
            torch.cat([outputs, multi_scale_features], dim=-1)
        )
        
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


class MultiScaleAggregation(nn.Module):
    def __init__(self, hidden_dim, num_scales=3):
        super(MultiScaleAggregation, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Different kernel sizes for multi-scale processing
        self.scale_conv = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2**i, padding=2**(i-1))
            for i in range(1, num_scales + 1)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, inputs, adj):
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Multi-scale convolution
        conv_inputs = inputs.transpose(1, 2)  # (batch, hidden, seq)
        scale_outputs = []
        
        for conv in self.scale_conv:
            conv_out = conv(conv_inputs)
            conv_out = F.relu(conv_out)
            # Ensure output has same sequence length as input
            if conv_out.size(-1) != seq_len:
                conv_out = F.interpolate(conv_out, size=seq_len, mode='linear', align_corners=False)
            scale_outputs.append(conv_out.transpose(1, 2))  # (batch, seq, hidden)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        
        # Ensure dimensions match for matrix multiplication
        if multi_scale_features.size(1) != adj.size(1):
            # Resize multi_scale_features to match adjacency matrix
            multi_scale_features = F.interpolate(
                multi_scale_features.transpose(1, 2), 
                size=adj.size(1), 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # Graph-based aggregation
        graph_enhanced = adj.bmm(multi_scale_features)
        
        # Scale fusion
        fused_features = self.scale_fusion(graph_enhanced)
        
        return fused_features


class GraphAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # Attention parameters
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, inputs, adj):
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Linear transformations
        Q = self.W_q(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply adjacency mask
        adj_mask = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(adj_mask == 0, -1e9)
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Output projection
        output = self.W_o(context)
        output = self.layer_norm(output + inputs)  # Residual connection
        
        return output


class AdaptiveGCN(nn.Module):
    """Adaptive Graph Convolution Network with dynamic edge weights"""
    
    def __init__(self, emb_dim=768, num_layers=3, gcn_dropout=0.1):
        super(AdaptiveGCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        
        # Adaptive edge weight learning
        self.edge_predictor = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # GCN layers
        self.W = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.emb_dim
            self.W.append(nn.Linear(input_dim, input_dim))
            self.layer_norms.append(nn.LayerNorm(input_dim))
            
        self.gcn_drop = nn.Dropout(gcn_dropout)
        
    def forward(self, adj, inputs):
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Learn adaptive edge weights
        adaptive_adj = self._learn_adaptive_edges(inputs, adj)
        
        # GCN processing
        denom = adaptive_adj.sum(2).unsqueeze(2) + 1
        mask = (adaptive_adj.sum(2) + adaptive_adj.sum(1)).eq(0).unsqueeze(2)
        
        outputs = inputs
        
        for l in range(self.layers):
            Ax = adaptive_adj.bmm(outputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](outputs)  # Self loop
            AxW = AxW / denom
            
            gAxW = F.relu(AxW)
            gAxW = self.layer_norms[l](gAxW)
            
            if l < self.layers - 1:
                outputs = self.gcn_drop(gAxW)
            else:
                outputs = gAxW
                
        return outputs, mask
    
    def _learn_adaptive_edges(self, inputs, base_adj):
        """Learn adaptive edge weights based on node features"""
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Create node pairs
        node_i = inputs.unsqueeze(2).expand(-1, -1, seq_len, -1)
        node_j = inputs.unsqueeze(1).expand(-1, seq_len, -1, -1)
        node_pairs = torch.cat([node_i, node_j], dim=-1)
        
        # Predict edge weights
        edge_weights = self.edge_predictor(node_pairs).squeeze(-1)
        
        # Combine with base adjacency
        adaptive_adj = base_adj * edge_weights
        
        # Add self-loops and normalize
        adaptive_adj = adaptive_adj + torch.eye(seq_len, device=inputs.device).unsqueeze(0)
        adaptive_adj = F.softmax(adaptive_adj, dim=-1)
        
        return adaptive_adj 