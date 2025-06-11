import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """Utility class for visualizing attention weights in the D2E2S model."""
    
    def __init__(self, save_dir="./attention_viz/"):
        self.save_dir = save_dir
        logger.info(f"Initialized AttentionVisualizer with save_dir: {save_dir}")
        
    def plot_attention_weights(self, attention_weights, tokens, title="Attention Weights"):
        """Plot attention weights as a heatmap."""
        try:
            logger.debug(f"Original attention weights shape: {attention_weights.shape if hasattr(attention_weights, 'shape') else 'None'}")
            
            if attention_weights is None:
                logger.error("Received None attention weights")
                raise ValueError("Cannot plot None attention weights")
                
            # Convert to numpy if needed
            if not isinstance(attention_weights, np.ndarray):
                logger.debug("Converting attention weights to numpy array")
                attention_weights = np.array(attention_weights)
            
            # Handle multi-dimensional attention tensors
            if len(attention_weights.shape) > 2:
                logger.info(f"Reducing {len(attention_weights.shape)}D attention tensor to 2D")
                if len(attention_weights.shape) == 3:
                    # For 3D tensor (e.g., batch x seq_len x seq_len)
                    # Take the first item if it's batch dimension
                    if attention_weights.shape[0] < attention_weights.shape[1]:
                        attention_weights = attention_weights[0]
                    # Or average across heads if it's multi-head attention
                    else:
                        attention_weights = attention_weights.mean(axis=0)
                elif len(attention_weights.shape) == 4:
                    # For 4D tensor (e.g., batch x heads x seq_len x seq_len)
                    # Take first batch and average across heads
                    attention_weights = attention_weights[0].mean(axis=0)
            
            logger.debug(f"Processed attention weights shape: {attention_weights.shape}")
            logger.debug(f"Attention weights statistics: min={attention_weights.min():.4f}, max={attention_weights.max():.4f}, mean={attention_weights.mean():.4f}")
                
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
            
            logger.debug(f"Successfully created attention heatmap for: {title}")
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting attention weights: {str(e)}")
            raise
        
    def plot_all_heads(self, attention_heads, tokens, layer_name=""):
        """Plot attention weights for all heads in a layer."""
        logger.debug(f"Plotting all attention heads for {layer_name}")
        logger.debug(f"Attention heads shape: {attention_heads.shape}")
        
        n_heads = attention_heads.shape[0]
        fig, axes = plt.subplots(
            n_heads // 2, 2, 
            figsize=(15, 4 * (n_heads // 2)),
            squeeze=False
        )
        
        for idx in range(n_heads):
            logger.debug(f"Plotting head {idx+1}/{n_heads}")
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
        logger.info(f"Successfully plotted all {n_heads} attention heads for {layer_name}")
        return fig
        
    @staticmethod
    def get_attention_stats(attention_weights):
        """Get statistics about attention weights."""
        stats = {
            "mean": attention_weights.mean().item(),
            "std": attention_weights.std().item(),
            "max": attention_weights.max().item(),
            "min": attention_weights.min().item(),
            "sparsity": (attention_weights < 0.1).float().mean().item()
        }
        logger.debug(f"Attention statistics: {stats}")
        return stats
        
    def save_attention_plot(self, fig, filename):
        """Save the attention visualization plot."""
        if not os.path.exists(self.save_dir):
            logger.info(f"Creating directory: {self.save_dir}")
            os.makedirs(self.save_dir)
            
        save_path = os.path.join(self.save_dir, filename)
        logger.info(f"Saving attention plot to: {save_path}")
        try:
            fig.savefig(save_path)
            plt.close(fig)
            logger.info(f"Successfully saved attention plot: {filename}")
        except Exception as e:
            logger.error(f"Failed to save attention plot: {e}")
            raise
        
    def visualize_model_attention(self, deberta_attn, lstm_attn=None, gcn_attn=None, 
                                tokens=None, save_prefix=""):
        """Visualize attention from different components of the model."""
        logger.info("Starting model attention visualization")
        
        if tokens is None and deberta_attn is not None:
            logger.debug("No tokens provided, generating default token labels")
            tokens = [f"Token_{i}" for i in range(deberta_attn.shape[-1])]
        elif tokens is None:
            logger.warning("No tokens and no DeBERTa attention provided, cannot visualize")
            return

        if deberta_attn is not None:
            logger.info("Visualizing DeBERTa attention")
            logger.debug(f"DeBERTa attention tensor shape: {deberta_attn.shape}")
            
            try:
                averaged_attn = deberta_attn.mean(dim=0)
                logger.debug(f"Averaged attention shape: {averaged_attn.shape}")
                
                fig = self.plot_attention_weights(
                    averaged_attn.cpu().numpy(),
                    tokens,
                    "DeBERTa Layer Attention"
                )
                
                if save_prefix:
                    self.save_attention_plot(fig, f"{save_prefix}_deberta_attn.png")
                    
            except Exception as e:
                logger.error(f"Failed to visualize DeBERTa attention: {str(e)}")
                logger.error(f"DeBERTa attention type: {type(deberta_attn)}")
                if isinstance(deberta_attn, torch.Tensor):
                    logger.error(f"DeBERTa attention device: {deberta_attn.device}")
                raise
                
        # LSTM attention if available 
        if lstm_attn is not None:
            logger.info("Visualizing LSTM attention")
            logger.debug(f"LSTM attention tensor shape: {lstm_attn.shape}")
            
            try:
                fig = self.plot_attention_weights(
                    lstm_attn.cpu().numpy(),
                    tokens,
                    "LSTM Attention"
                )
                if save_prefix:
                    self.save_attention_plot(fig, f"{save_prefix}_lstm_attn.png")
            except Exception as e:
                logger.error(f"Failed to visualize LSTM attention: {e}")
                raise
                
        # GCN attention if available
        if gcn_attn is not None:
            logger.info("Visualizing GCN attention")
            logger.debug(f"GCN attention tensor shape: {gcn_attn.shape}")
            
            try:
                fig = self.plot_attention_weights(
                    gcn_attn.cpu().numpy(),
                    tokens, 
                    "GCN Attention"
                )
                if save_prefix:
                    self.save_attention_plot(fig, f"{save_prefix}_gcn_attn.png")
            except Exception as e:
                logger.error(f"Failed to visualize GCN attention: {e}")
                raise
        
        logger.info("Completed model attention visualization")
        
    def visualize_model_attention(self, deberta_attn, sem_gcn_attn, syn_gcn_attn, tokens, save_prefix):
        """Visualize different types of attention patterns from the model."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"Saving visualizations to: {self.save_dir}")

            # Use higher DPI and SVG format for better quality
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['svg.fonttype'] = 'none'  # This ensures text remains as text in SVG
            
            if deberta_attn is not None:
                try:
                    fig = self.plot_attention_weights(deberta_attn, tokens, "DeBERTa Attention")
                    save_path = os.path.join(self.save_dir, f"{save_prefix}_deberta_attn.svg")
                    fig.savefig(save_path, format='svg', bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Saved DeBERTa attention visualization to {save_path}")
                    
                    # Also save as HTML file for easier viewing
                    html_path = os.path.join(self.save_dir, f"{save_prefix}_deberta_attn.html")
                    self._save_attention_html(deberta_attn, tokens, "DeBERTa Attention", html_path)
                except Exception as e:
                    logger.warning(f"Could not visualize DeBERTa attention: {str(e)}")
            
            if sem_gcn_attn is not None:
                try:
                    fig = self.plot_attention_weights(sem_gcn_attn, tokens, "Semantic GCN Attention")
                    save_path = os.path.join(self.save_dir, f"{save_prefix}_sem_gcn_attn.svg")
                    fig.savefig(save_path, format='svg', bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Saved Semantic GCN attention visualization to {save_path}")
                    
                    # Also save as HTML file
                    html_path = os.path.join(self.save_dir, f"{save_prefix}_sem_gcn_attn.html")
                    self._save_attention_html(sem_gcn_attn, tokens, "Semantic GCN Attention", html_path)
                except Exception as e:
                    logger.warning(f"Could not visualize Semantic GCN attention: {str(e)}")
            
            if syn_gcn_attn is not None:
                try:
                    fig = self.plot_attention_weights(syn_gcn_attn, tokens, "Syntactic GCN Attention")
                    save_path = os.path.join(self.save_dir, f"{save_prefix}_syn_gcn_attn.svg")
                    fig.savefig(save_path, format='svg', bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Saved Syntactic GCN attention visualization to {save_path}")
                    
                    # Also save as HTML file
                    html_path = os.path.join(self.save_dir, f"{save_prefix}_syn_gcn_attn.html")
                    self._save_attention_html(syn_gcn_attn, tokens, "Syntactic GCN Attention", html_path)
                except Exception as e:
                    logger.warning(f"Could not visualize Syntactic GCN attention: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error in visualize_model_attention: {str(e)}")
            raise
    
    def _save_attention_html(self, attention_weights, tokens, title, save_path):
        """Save attention weights as an interactive HTML heatmap."""
        try:
            # Convert attention weights to 2D if needed
            if len(attention_weights.shape) > 2:
                if len(attention_weights.shape) == 3:
                    attention_weights = attention_weights.mean(axis=0)
                elif len(attention_weights.shape) == 4:
                    attention_weights = attention_weights[0].mean(axis=0)
            
            # Convert to numpy if needed
            if not isinstance(attention_weights, np.ndarray):
                attention_weights = np.array(attention_weights)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    .attention-heatmap {{
                        border-collapse: collapse;
                        margin: 20px;
                    }}
                    .attention-heatmap td {{
                        width: 30px;
                        height: 30px;
                        text-align: center;
                        padding: 5px;
                    }}
                    .token-label {{
                        font-weight: bold;
                        padding: 5px;
                    }}
                </style>
            </head>
            <body>
                <h2>{title}</h2>
                <table class="attention-heatmap">
                    <tr>
                        <td></td>
                        {''.join(f'<td class="token-label">{t}</td>' for t in tokens)}
                    </tr>
            """
            
            # Add attention values
            for i, token_row in enumerate(tokens):
                html_content += f'<tr><td class="token-label">{token_row}</td>'
                for j in range(attention_weights.shape[1]):
                    value = attention_weights[i, j]
                    # Convert value to RGB color (red intensity)
                    color = f"rgb(255, {int(255 * (1 - value))}, {int(255 * (1 - value))})"
                    html_content += f'<td style="background-color: {color}">{value:.2f}</td>'
                html_content += '</tr>'
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            # Save HTML file
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Saved HTML visualization to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving HTML visualization: {str(e)}")
            raise
