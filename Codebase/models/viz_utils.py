import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionVisualizer:
    """Utility class for visualizing attention weights in the D2E2S model."""
    
    def __init__(self, save_dir="./attention_viz/"):
        self.save_dir = save_dir
        
    def plot_attention_weights(self, attention_weights, tokens, title="Attention Weights"):
        """Plot attention weights as a heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            annot=True if len(tokens) < 20 else False
        )
        plt.title(title)
        plt.xlabel("Target Tokens")
        plt.ylabel("Source Tokens")
        plt.tight_layout()
        
        # Return figure for saving or displaying
        return plt.gcf()
        
    def plot_all_heads(self, attention_heads, tokens, layer_name=""):
        """Plot attention weights for all heads in a layer."""
        n_heads = attention_heads.shape[0]
        fig, axes = plt.subplots(
            n_heads // 2, 2, 
            figsize=(15, 4 * (n_heads // 2)),
            squeeze=False
        )
        
        for idx in range(n_heads):
            i, j = idx // 2, idx % 2
            sns.heatmap(
                attention_heads[idx],
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd',
                ax=axes[i,j],
                annot=True if len(tokens) < 20 else False
            )
            axes[i,j].set_title(f"{layer_name} Head {idx+1}")
            
        plt.tight_layout()
        return fig
        
    @staticmethod
    def get_attention_stats(attention_weights):
        """Get statistics about attention weights."""
        return {
            "mean": attention_weights.mean().item(),
            "std": attention_weights.std().item(),
            "max": attention_weights.max().item(),
            "min": attention_weights.min().item(),
            "sparsity": (attention_weights < 0.1).float().mean().item()
        }
        
    def save_attention_plot(self, fig, filename):
        """Save the attention visualization plot."""
        import os
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        fig.savefig(os.path.join(self.save_dir, filename))
        plt.close(fig)
        
    def visualize_model_attention(self, deberta_attn, lstm_attn=None, gcn_attn=None, 
                                tokens=None, save_prefix=""):
        """Visualize attention from different components of the model."""
        if tokens is None:
            tokens = [f"Token_{i}" for i in range(deberta_attn.shape[-1])]
            
        # DeBERTa attention
        fig = self.plot_attention_weights(
            deberta_attn.mean(dim=0).cpu().numpy(),
            tokens,
            "DeBERTa Layer Attention"
        )
        if save_prefix:
            self.save_attention_plot(fig, f"{save_prefix}_deberta_attn.png")
            
        # LSTM attention if available 
        if lstm_attn is not None:
            fig = self.plot_attention_weights(
                lstm_attn.cpu().numpy(),
                tokens,
                "LSTM Attention"
            )
            if save_prefix:
                self.save_attention_plot(fig, f"{save_prefix}_lstm_attn.png")
                
        # GCN attention if available
        if gcn_attn is not None:
            fig = self.plot_attention_weights(
                gcn_attn.cpu().numpy(),
                tokens,
                "GCN Attention"
            )
            if save_prefix:
                self.save_attention_plot(fig, f"{save_prefix}_gcn_attn.png")
