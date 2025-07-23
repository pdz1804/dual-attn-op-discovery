# Import necessary libraries
from __future__ import division
import logging
import sys
import argparse

# === Create a fresh log file every time ===
LOG_FILE_PATH = "pdzttb.log"

# Remove any existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up new logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),  # Overwrite each run
        logging.StreamHandler(sys.stdout),  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# Import from configs
from configs.hyperparams import *
from utils.seed_everything import set_seed

def main():
    parser = argparse.ArgumentParser(
        description='FullFlow Patent-Product Matching System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pipeline dual_attn
  python main.py --pipeline patent_product --mode train
  python main.py --pipeline patent_product --mode test --embedding_type sentence_transformer
  python main.py --pipeline patent_product --mode chat --use_rag
  python main.py --pipeline rag_only --query "machine learning algorithms"
        """
    )
    
    # Main pipeline selection
    parser.add_argument('--pipeline', 
                        choices=['dual_attn', 'patent_product', 'rag_only', 'clustering'], 
                        required=True,
                        help='Pipeline to run')
    
    # Mode for patent_product pipeline
    parser.add_argument('--mode', 
                        choices=['train', 'test', 'chat'],
                        help='Mode for patent_product pipeline (required for patent_product)')
    
    # Embedding configuration
    parser.add_argument('--embedding_type', 
                        choices=['fasttext', 'sentence_transformer'], 
                        default='fasttext',
                        help='Type of embeddings to use')
    
    parser.add_argument('--sentence_transformer_model', 
                        default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    
    # RAG configuration
    parser.add_argument('--use_rag', 
                        action='store_true',
                        help='Use RAG approach for testing/chatting')
    
    parser.add_argument('--rag_use_external_summaries', 
                        action='store_true',
                        help='Use external summaries for RAG instead of dual attention keywords')
    
    parser.add_argument('--rag_top_k', 
                        type=int, 
                        default=5,
                        help='Number of top results for RAG queries')
    
    # UI and flow configuration
    parser.add_argument('--ui_flow_type', 
                        choices=['ml', 'rag'], 
                        default='ml',
                        help='UI flow type')
    
    parser.add_argument('--enable_dual_attention', 
                        action='store_true', 
                        default=True,
                        help='Enable dual attention model')
    
    parser.add_argument('--enable_transformation_matrix', 
                        action='store_true', 
                        default=True,
                        help='Enable transformation matrix')
    
    # Force rebuild options
    parser.add_argument('--force_rebuild_rag', 
                        action='store_true',
                        help='Force rebuild RAG vector database')
    
    parser.add_argument('--force_rebuild_clustering', 
                        action='store_true',
                        help='Force rebuild clustering models')
    
    # Clustering configuration
    parser.add_argument('--enable_clustering', 
                        action='store_true',
                        help='Enable clustering analysis')
    
    # Query for rag_only pipeline
    parser.add_argument('--query', 
                        type=str,
                        help='Query string for rag_only pipeline')
    
    # Display configuration
    parser.add_argument('--max_keywords_display',
                        type=int,
                        default=MAX_KEYWORDS_DISPLAY,
                        help=f'Maximum number of keywords to display in results (default: {MAX_KEYWORDS_DISPLAY})')

    # new: choosing the model type for transformation from patent to product and vice versa
    parser.add_argument('--model_type', 
                    choices=['linear', 'mlp'],  # Add more later
                    default='linear',
                    help='Type of transformation model to train')
    
    # new: choosing the approximation method for matrix extraction
    parser.add_argument('--approx_method', 
                    choices=['sampling', 'polynomial', 'kernel'],  # Add more later
                    default='sampling',
                    help='Type of approximation method to use for matrix extraction')

    args = parser.parse_args()
    
    # Create configuration dictionary from arguments
    config = {
        'embedding_type': args.embedding_type,
        'sentence_transformer_model': args.sentence_transformer_model,
        'use_rag': args.use_rag,
        'rag_use_external_summaries': args.rag_use_external_summaries,
        'rag_top_k': args.rag_top_k,
        'ui_flow_type': args.ui_flow_type,
        'enable_dual_attention': args.enable_dual_attention,
        'enable_clustering': args.enable_clustering,
        'enable_transformation_matrix': args.enable_transformation_matrix,
        'force_rebuild_rag': args.force_rebuild_rag,
        'force_rebuild_clustering': args.force_rebuild_clustering,
        'enable_clustering': args.enable_clustering,
        'query': args.query,
        'max_keywords_display': args.max_keywords_display,
        'model_type': args.model_type, # new
        'approx_method': args.approx_method  # Default approximation method
    }
    
    set_seed(42)
    
    logger.info(f"Starting pipeline: {args.pipeline}")
    logger.info(f"Configuration: {config}")

    # Route to appropriate pipeline
    if args.pipeline == 'dual_attn':
        from pipelines.dual_attention_pipeline import dual_attention_pipeline
        dual_attention_pipeline(config)

    elif args.pipeline == 'patent_product':
        if args.mode is None:
            raise ValueError("Error: --mode is required for patent_product pipeline. Use --mode=train / test / chat.")
        
        from pipelines.patent_product_pipeline import train_pipeline, test_pipeline, chat_pipeline
        
        if args.mode == 'train':
            train_pipeline(config)
        elif args.mode == 'test':
            test_pipeline(config)
        elif args.mode == 'chat':
            chat_pipeline(config)
        
    elif args.pipeline == 'rag_only':
        from pipelines.patent_product_pipeline import rag_only_pipeline
        rag_only_pipeline(config)
    
    elif args.pipeline == 'clustering':
        from pipelines.clustering_pipeline import clustering_pipeline
        clustering_pipeline(config)

if __name__ == "__main__":
    main()
    exit(1)  # Exit after running the pipeline
    