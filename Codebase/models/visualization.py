import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

class AttentionVisualizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def plot_attention_weights(self, attention_weights, input_ids, layer_index=-1, head_index=None, 
                             title=None, figsize=(10, 8)):
        """
        Plot attention weights from a transformer model.
        
        Args:
            attention_weights: Tuple of attention tensors from the model
            input_ids: Input token IDs
            layer_index: Which layer to visualize (-1 for last layer)
            head_index: Which attention head to visualize (None for average across heads)
            title: Title for the plot
            figsize: Size of the figure (width, height)
        """
        if not isinstance(attention_weights, tuple):
            attention_weights = (attention_weights,)
            
        # Get attention weights for specified layer
        layer_weights = attention_weights[layer_index]  # shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Convert to numpy and move to CPU if needed
        if isinstance(layer_weights, torch.Tensor):
            layer_weights = layer_weights.detach().cpu().numpy()
        
        # Take first example from batch
        layer_weights = layer_weights[0]  # shape: [num_heads, seq_len, seq_len]
        
        # Average across heads if head_index not specified
        if head_index is None:
            attn_map = layer_weights.mean(axis=0)  # shape: [seq_len, seq_len]
        else:
            attn_map = layer_weights[head_index]  # shape: [seq_len, seq_len]
            
        # Get token labels
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.heatmap(attn_map, 
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='viridis')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Set title
        if title is None:
            title = f"Attention Weights (Layer {layer_index+1})"
            if head_index is not None:
                title += f" (Head {head_index+1})"
        plt.title(title)
        
        plt.tight_layout()
        return plt.gcf()
        
    def plot_all_heads(self, attention_weights, input_ids, layer_index=-1, max_heads=4, 
                      ncols=2, figsize=(15, 15)):
        """
        Plot attention patterns for all heads in a given layer.
        
        Args:
            attention_weights: Tuple of attention tensors from the model
            input_ids: Input token IDs
            layer_index: Which layer to visualize (-1 for last layer)
            max_heads: Maximum number of heads to plot
            ncols: Number of columns in the subplot grid
            figsize: Size of the figure (width, height)
        """
        if not isinstance(attention_weights, tuple):
            attention_weights = (attention_weights,)
            
        layer_weights = attention_weights[layer_index][0]  # Get first batch, specified layer
        num_heads = min(layer_weights.shape[0], max_heads)
        nrows = (num_heads + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        for i in range(num_heads):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            attn_map = layer_weights[i].detach().cpu().numpy()
            sns.heatmap(attn_map, xticklabels=tokens, yticklabels=tokens, 
                       cmap='viridis', ax=ax)
            ax.set_title(f'Head {i+1}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        # Remove empty subplots
        for i in range(num_heads, nrows * ncols):
            row = i // ncols
            col = i % ncols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout()
        return fig
        
    def save_attention_plot(self, fig, filepath):
        """Save attention visualization to file."""
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
