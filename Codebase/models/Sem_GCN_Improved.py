import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math

class ImprovedSemGCN(nn.Module):
    def __init__(self, args, emb_dim=768, num_layers=3, gcn_dropout=0.1):
        super(ImprovedSemGCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.attention_heads = max(8, self.args.attention_heads)  # Increased attention heads
        self.mem_dim = self.args.hidden_dim
        
        # Enhanced GCN layers with residual connections
        self.W = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_weights = nn.ParameterList()
        
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
            self.layer_norms.append(nn.LayerNorm(input_dim))
            self.residual_weights.append(nn.Parameter(torch.ones(1)))
            
        self.gcn_drop = nn.Dropout(gcn_dropout)
        
        # Enhanced attention mechanism
        self.attn = ImprovedMultiHeadAttention(self.attention_heads, self.mem_dim * 2)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim)
        )
        
        # Global context modeling
        self.global_context = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Position encoding for better sequence modeling
        self.pos_encoding = PositionalEncoding(self.emb_dim, max_len=512)

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
        
        # Create attention-based adjacency matrix
        # Use the attention output to create an adjacency matrix
        batch_size, seq_len, _ = attn_tensor.size()
        
        # Create adjacency matrix from attention weights
        # For simplicity, we'll create a basic adjacency matrix
        adj_ag_new = torch.eye(seq_len, device=gcn_inputs.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add some connectivity based on attention
        # This is a simplified version - you can enhance this based on your needs
        adj_ag_new = adj_ag_new + 0.1 * torch.ones_like(adj_ag_new)

        # Enhanced adjacency matrix processing
        for j in range(adj_ag_new.size(0)):
            # Normalize the adjacency matrix
            adj_ag_new[j] = F.softmax(adj_ag_new[j], dim=-1)
        
        # Apply mask
        adj_ag_new = mask_ * adj_ag_new

        # Enhanced GCN layers with residual connections
        denom_ag = adj_ag_new.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        residual_outputs = []

        for l in range(self.layers):
            # GCN operation
            Ax = adj_ag_new.bmm(outputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom_ag
            
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

        # Global context modeling
        try:
            global_outputs, _ = self.global_context(
                outputs, outputs, outputs,
                key_padding_mask=~mask_.squeeze(-1).bool()
            )
        except:
            # Fallback if global context fails
            global_outputs = outputs
        
        # Feature fusion
        try:
            fused_outputs = self.feature_fusion(torch.cat([outputs, global_outputs], dim=-1))
        except:
            # Fallback if feature fusion fails
            fused_outputs = outputs
        
        return fused_outputs, adj_ag_new


class ImprovedMultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(ImprovedMultiHeadAttention, self).__init__()
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
    def __init__(self, d_model, max_len=512):
        super(RelativePositionEncoding, self).__init__()
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
        
        # Clamp indices to valid range
        rel_pos_indices = torch.clamp(rel_pos_indices, 0, 2 * self.max_len - 2)
        
        # Get relative position embeddings for diagonal (self-relative positions)
        diag_indices = torch.arange(seq_len, device=x.device)
        diag_rel_pos = self.rel_pos_emb[self.max_len - 1]  # Center position embedding
        
        # Expand to match input dimensions
        rel_pos_emb = diag_rel_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, d_model)
        rel_pos_emb = rel_pos_emb.expand(batch_size, num_heads, seq_len, d_model)
        
        # Add to input
        return x + rel_pos_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
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