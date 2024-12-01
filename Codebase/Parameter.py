import argparse
import torch

def train_argparser():
    dataset_files = {
        '14lap': {
            'train': './data/14lap/train_dep_triple_polarity_result.json',
            'test': './data/14lap/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        },
        '14res': {
            'train': './data/14res/train_dep_triple_polarity_result.json',
            'test': './data/14res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        },
        '15res': {
            'train': './data/15res/train_dep_triple_polarity_result.json',
            'test': './data/15res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        },
        '16res': {
            'train': './data/16res/train_dep_triple_polarity_result.json',
            'test': './data/16res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        }
    }

    # Model arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='14res', type=str, help='14res, 15res, 16res, 14lap')

    # Model architecture
    parser.add_argument('--drop_out_rate', type=float, default=0.1, help='drop out rate')
    parser.add_argument('--is_bidirect', default=True, help='Use bi-RNN layer')
    parser.add_argument('--use_gated', default=False, help='Use gcnconv and gatedgraphconv')
    parser.add_argument('--hidden_dim', type=int, default=384, help='Hidden layer dimension')  # Updated for DeBERTa V2
    parser.add_argument('--emb_dim', type=int, default=768, help='Word embedding dimension')  # Updated for DeBERTa V2
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--lstm_dim', type=int, default=384, help='Dimension of lstm cell')
    
    # Path and data settings
    parser.add_argument('--prefix', type=str, default="data/", help='Dataset and embedding path prefix')
    parser.add_argument('--span_generator', type=str, default="Max", choices=["Max", "Average"], 
                       help='Span generation method: Max or Average')
    
    # Network parameters
    parser.add_argument('--attention_heads', default=12, type=int, help='Number of attention heads')  # Updated for DeBERTa V2
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--mem_dim', type=int, default=768, help='Mutual biaffine memory dimension')  # Updated for DeBERTa V2
    parser.add_argument('--gcn_dropout', type=float, default=0.2, help='GCN layer dropout rate')
    parser.add_argument('--pooling', default='avg', type=str, help='Pooling type: max, avg, sum')
    parser.add_argument('--gcn_dim', type=int, default=300, help='Dimension of GCN')
    parser.add_argument('--bert_feature_dim', type=int, default=768, help='Dimension of pretrained features')  # Updated for DeBERTa V2
    
    # Training settings
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")
    parser.add_argument('--max_span_size', type=int, default=8, help="Maximum size of spans")
    parser.add_argument('--lowercase', action='store_true', default=True, help="Case-sensitive training if False")
    parser.add_argument('--max_pairs', type=int, default=1000, help="Maximum number of entity pairs for processing")
    parser.add_argument('--sen_filter_threshold', type=float, default=0.4, help="Sentiment triplet filter threshold")
    parser.add_argument('--sampling_limit', type=int, default=100, help="Maximum number of samples in queue")
    parser.add_argument('--neg_entity_count', type=int, default=100, help="Number of negative entity samples per sample")
    parser.add_argument('--neg_triple_count', type=int, default=100, help="Number of negative triplet samples per sample")

    # Model paths and settings
    parser.add_argument('--tokenizer_path', default='microsoft/deberta-base', type=str, help="Path to tokenizer")
    parser.add_argument('--pretrained_bert_name', default='microsoft/deberta-base', type=str, help="Pretrained model name")
    
    # Hardware settings
    parser.add_argument('--cpu', action='store_true', default=False, 
                       help="Use CPU even if CUDA is available")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda/cpu)")
    
    # Training hyperparameters
    parser.add_argument('--size_embedding', type=int, default=25, help="Size of embedding dimension")
    parser.add_argument('--sampling_processes', type=int, default=4, help="Number of sampling processes")
    parser.add_argument('--prop_drop', type=float, default=0.1, help="Dropout probability")
    parser.add_argument('--freeze_transformer', action='store_true', default=False, 
                       help="Freeze transformer parameters")
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size")  # Adjusted for DeBERTa V2
    parser.add_argument('--epochs', type=int, default=120, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")  # Adjusted for DeBERTa V2
    parser.add_argument('--lr_warmup', type=float, default=0.1, 
                       help="Warmup proportion for learning rate schedule")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")
    
    # Logging and evaluation
    parser.add_argument('--log_path', type=str, default="log/", help="Path for storing logs")
    parser.add_argument('--train_log_iter', type=int, default=1, help="Log frequency during training")
    parser.add_argument('--init_eval', action='store_true', default=False, 
                       help="Evaluate before training")
    parser.add_argument('--final_eval', action='store_true', default=False,
                       help="Evaluate only after training")
    parser.add_argument('--store_predictions', action='store_true', default=True,
                       help="Store predictions to disk")
    parser.add_argument('--store_examples', action='store_true', default=True,
                       help="Store evaluation examples to disk")
    parser.add_argument('--example_count', type=int, default=None,
                       help="Number of evaluation examples to store")
    
    # Model saving
    parser.add_argument('--save_path', type=str, default="data/save/",
                       help="Path for model checkpoints")
    parser.add_argument('--save_optimizer', action='store_true', default=False,
                       help="Save optimizer state")

    # Parse arguments
    opt = parser.parse_args()
    opt.label = opt.dataset
    opt.dataset_file = dataset_files[opt.dataset]
    
    # Set device
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    return opt