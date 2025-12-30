import argparse
import torch


def train_argparser_optimized():
    """
    Optimized hyperparameters based on old codebase that achieved 75-80% F1
    Key changes from current:
    - Learning rate: 5e-6 (was 5e-5, 10x lower)
    - Batch size: 16 (stable)
    - Model: deberta-v2-xxlarge (1.5B params)
    - emb_dim: 1536, hidden_dim: 768 (matching xxlarge output)
    - Conservative dropout and regularization
    """

    dataset_files = {
        "14lap": {
            "train": "./data/14lap/train_dep_triple_polarity_result.json",
            "test": "./data/14lap/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        },
        "14res": {
            "train": "./data/14res/train_dep_triple_polarity_result.json",
            "test": "./data/14res/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        },
        "15res": {
            "train": "./data/15res/train_dep_triple_polarity_result.json",
            "test": "./data/15res/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        },
        "16res": {
            "train": "./data/16res/train_dep_triple_polarity_result.json",
            "test": "./data/16res/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="14res", type=str, help="14res, 15res, 16res, 14lap"
    )

    # CRITICAL: Learning rate - 60x lower than previous attempts
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (OLD CODEBASE: 5e-6)")
    
    # Model configuration - matching old codebase
    parser.add_argument(
        "--pretrained_deberta_name", default="microsoft/deberta-v2-xxlarge", type=str,
        help="Use xxlarge model (1.5B params) like old codebase"
    )
    parser.add_argument(
        "--deberta_feature_dim", type=int, default=1536,
        help="xxlarge output dimension (OLD: 1536)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=768,
        help="Hidden dimension (OLD: 768)"
    )
    parser.add_argument(
        "--emb_dim", type=int, default=1536,
        help="Embedding dimension matching xxlarge (OLD: 1536)"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (OLD: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=120,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr_warmup", type=float, default=0.1,
        help="Warmup proportion"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Gradient clipping"
    )
    
    # Dropout - conservative like old codebase
    parser.add_argument(
        "--drop_out_rate", type=float, default=0.3,
        help="LSTM dropout"
    )
    parser.add_argument(
        "--prop_drop", type=float, default=0.05,
        help="Classifier dropout (OLD: 0.05)"
    )
    parser.add_argument(
        "--gcn_dropout", type=float, default=0.1,
        help="GCN dropout (OLD: 0.1)"
    )
    
    # GCN configuration - keep basic like old codebase
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="GCN layers (OLD: 2)"
    )
    parser.add_argument(
        "--gcn_dim", type=int, default=768,
        help="GCN dimension"
    )
    parser.add_argument(
        "--attention_heads", default=8, type=int,
        help="Attention heads"
    )
    
    # LSTM configuration
    parser.add_argument(
        "--lstm_layers", type=int, default=2,
        help="LSTM layers"
    )
    parser.add_argument(
        "--lstm_dim", type=int, default=384,
        help="LSTM dimension"
    )
    parser.add_argument(
        "--is_bidirect", default=True,
        help="Bidirectional LSTM"
    )
    
    # Span and sampling configuration
    parser.add_argument(
        "--max_span_size", type=int, default=6,
        help="Maximum span size (OLD: 6)"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=800,
        help="Max entity pairs"
    )
    parser.add_argument(
        "--neg_entity_count", type=int, default=50,
        help="Negative entity samples"
    )
    parser.add_argument(
        "--neg_triple_count", type=int, default=50,
        help="Negative triplet samples"
    )
    parser.add_argument(
        "--sampling_limit", type=int, default=100,
        help="Sampling queue limit"
    )
    parser.add_argument(
        "--sampling_processes", type=int, default=4,
        help="Sampling processes"
    )
    
    # Other parameters
    parser.add_argument(
        "--size_embedding", type=int, default=25,
        help="Size embedding dimension"
    )
    parser.add_argument(
        "--mem_dim", type=int, default=768,
        help="Memory dimension"
    )
    parser.add_argument(
        "--pooling", default="avg", type=str,
        help="Pooling method"
    )
    parser.add_argument(
        "--span_generator", type=str, default="Max",
        choices=["Max", "Average"],
        help="Span generator"
    )
    parser.add_argument(
        "--sen_filter_threshold", type=float, default=0.5,
        help="Sentiment filter threshold"
    )
    
    # System configuration
    parser.add_argument(
        "--seed", default=42, type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False,
        help="Use CPU"
    )
    parser.add_argument(
        "--freeze_transformer", action="store_true", default=False,
        help="Freeze transformer weights"
    )
    
    # Paths
    parser.add_argument(
        "--tokenizer_path", default="bert-base-uncased", type=str,
        help="Tokenizer path"
    )
    parser.add_argument(
        "--prefix", type=str, default="data/",
        help="Data prefix"
    )
    parser.add_argument(
        "--log_path", type=str, default="log/",
        help="Log directory"
    )
    
    # Logging
    parser.add_argument(
        "--train_log_iter", type=int, default=1,
        help="Log interval"
    )
    parser.add_argument(
        "--store_predictions", action="store_true", default=True,
        help="Store predictions"
    )
    parser.add_argument(
        "--init_eval", action="store_true", default=False,
        help="Evaluate before training"
    )
    parser.add_argument(
        "--final_eval", action="store_true", default=False,
        help="Evaluate only after training"
    )
    
    # Case sensitivity
    parser.add_argument(
        "--lowercase", action="store_true", default=True,
        help="Lowercase text"
    )
    
    # Legacy parameters
    parser.add_argument(
        "--use_gated", default=False,
        help="Use gated GCN"
    )

    args = parser.parse_args()
    args.train_path = dataset_files[args.dataset]["train"]
    args.test_path = dataset_files[args.dataset]["test"]
    args.types_path = dataset_files[args.dataset]["types_path"]
    args.dataset_file = dataset_files[args.dataset]
    
    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    return args
