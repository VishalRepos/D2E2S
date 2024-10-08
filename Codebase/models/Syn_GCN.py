import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, num_layers=2, gcn_dropout=0.1):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.gcn_dropout = gcn_dropout
        
        print(f"GCN initialized with input_dim: {input_dim}, hidden_dim: {hidden_dim}")
        
        # We'll create the layers dynamically in the forward pass

    def forward(self, adj, inputs):
        print(f"GCN forward - adj shape: {adj.shape}, inputs shape: {inputs.shape}")
        
        # Dynamically create or adjust layers based on input size
        if not hasattr(self, 'W') or self.W[0].in_features != inputs.shape[-1]:
            self.input_dim = inputs.shape[-1]
            self.W = nn.ModuleList()
            self.W.append(nn.Linear(self.input_dim, self.hidden_dim))
            for _ in range(1, self.layers):
                self.W.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            print(f"Adjusted GCN layers for input_dim: {self.input_dim}")

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        
        for l in range(self.layers):
            Ax = adj.bmm(inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](inputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            inputs = F.dropout(gAxW, self.gcn_dropout, training=self.training) if l < self.layers - 1 else gAxW

        return inputs, mask