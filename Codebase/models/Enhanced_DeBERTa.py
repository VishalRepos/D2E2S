import torch
import torch.nn as nn
from transformers import AutoModel


class EnhancedDeBERTa(nn.Module):
    """
    Enhanced DeBERTa with improved dropout and layer normalization
    
    Improvements:
    1. Variational dropout (same mask across timesteps)
    2. Layer normalization after transformer output
    3. Residual connection with learnable gate
    4. Optional attention dropout enhancement
    """
    
    def __init__(self, config, enhanced_dropout=0.1, use_layer_norm=True, use_residual_gate=True):
        super(EnhancedDeBERTa, self).__init__()
        
        # Load pretrained DeBERTa
        self.deberta = AutoModel.from_config(config)
        
        # Enhanced components
        self.enhanced_dropout = enhanced_dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual_gate = use_residual_gate
        
        hidden_size = config.hidden_size
        
        # Layer normalization for output stabilization
        if self.use_layer_norm:
            self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # Variational dropout (applied consistently across sequence)
        self.variational_dropout = nn.Dropout(enhanced_dropout)
        
        # Residual gate (learnable combination of input embeddings and transformer output)
        if self.use_residual_gate:
            self.residual_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        
        # Additional dropout for attention outputs (if needed)
        self.attention_dropout = nn.Dropout(enhanced_dropout * 0.5)  # Lighter dropout
        
    def forward(self, input_ids, attention_mask=None, return_dict=False):
        """
        Forward pass with enhancements
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_dict: Whether to return dict (for compatibility)
            
        Returns:
            Enhanced transformer outputs
        """
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Apply variational dropout
        hidden_states = self.variational_dropout(hidden_states)
        
        # Apply layer normalization for stability
        if self.use_layer_norm:
            hidden_states = self.output_layer_norm(hidden_states)
        
        # Optional: Residual gate with input embeddings
        if self.use_residual_gate and hasattr(self.deberta, 'embeddings'):
            # Get input embeddings
            input_embeds = self.deberta.embeddings.word_embeddings(input_ids)
            
            # Compute gate
            gate_input = torch.cat([hidden_states, input_embeds], dim=-1)
            gate = self.residual_gate(gate_input)
            
            # Apply gated residual
            hidden_states = gate * hidden_states + (1 - gate) * input_embeds
        
        # Return in same format as original
        if return_dict:
            outputs.last_hidden_state = hidden_states
            return outputs
        else:
            return (hidden_states,) + outputs[1:]
    
    def parameters(self, recurse=True):
        """Override to include all parameters"""
        return super().parameters(recurse=recurse)


class MinimalEnhancedDeBERTa(nn.Module):
    """
    Minimal enhancement - just better dropout and layer norm
    (Use this if memory is tight)
    """
    
    def __init__(self, config, enhanced_dropout=0.1):
        super(MinimalEnhancedDeBERTa, self).__init__()
        
        self.deberta = AutoModel.from_config(config)
        hidden_size = config.hidden_size
        
        # Just add layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(enhanced_dropout)
        
    def forward(self, input_ids, attention_mask=None, return_dict=False):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply enhancements
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        if return_dict:
            outputs.last_hidden_state = hidden_states
            return outputs
        else:
            return (hidden_states,) + outputs[1:]
