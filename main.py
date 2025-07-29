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
                        choices=['dual_attn', 'patent_product', 'rag_only', 'clustering', 'product_suggestion'], 
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
    
    # Product suggestion configuration
    parser.add_argument('--enable_product_suggestions',
                        action='store_true',
                        help='Enable product suggestions in test/chat modes')
    
    parser.add_argument('--product_similarity_threshold',
                        type=float,
                        default=PRODUCT_SUGGESTION_CONFIG['similarity_threshold'],
                        help='Similarity threshold for product suggestions')
    
    parser.add_argument('--enable_openai_enhance',
                        action='store_true',
                        help='Use OpenAI to enhance product names (requires API key in .env)')
    
    parser.add_argument('--product_suggestions_only',
                        action='store_true',
                        help='Run only product suggestions on existing results')
    
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
        'model_type': args.model_type,          # New
        'approx_method': args.approx_method,    # Default approximation method
        'countries': COUNTRY,                    # Add countries for consistent clustering
        'enable_product_suggestions': args.enable_product_suggestions,
        'product_similarity_threshold': args.product_similarity_threshold,
        'enable_openai_enhance': args.enable_openai_enhance,
        'product_suggestions_only': args.product_suggestions_only
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
    
    elif args.pipeline == 'product_suggestion':
        if not config.get('query'):
            # Interactive mode if no query provided
            query = input("Enter patent abstract or product query: ").strip()
            if not query:
                raise ValueError("Error: Query is required for product suggestion pipeline.")
            config['query'] = query
        
        from pipelines.product_suggestion_pipeline import PatentProductSuggester
        
        # Enhanced demo data with multiple relevant companies
        sample_companies = [
            {
                'hojinid': 'DEMO001',
                'name': 'MedTech AI Solutions',
                'keywords': ['artificial intelligence', 'medical imaging', 'computer vision', 'machine learning', 'healthcare', 'diagnostic systems'],
                'patents': [
                    {
                        'patent_id': 'DEMO2023001',
                        'abstract': 'AI-powered medical imaging system for automated diagnosis using deep learning neural networks to analyze X-ray, MRI, and CT scans with high accuracy for healthcare professionals.'
                    },
                    {
                        'patent_id': 'DEMO2023002',
                        'abstract': 'Machine learning system for medical image analysis using artificial intelligence to detect anomalies and assist in clinical decision making.'
                    }
                ]
            },
            {
                'hojinid': 'DEMO002',
                'name': 'Vision AI Technologies',
                'keywords': ['computer vision', 'artificial intelligence', 'image processing', 'pattern recognition', 'automation'],
                'patents': [
                    {
                        'patent_id': 'DEMO2023003',
                        'abstract': 'Computer vision system using convolutional neural networks for real-time object detection and image classification in various applications.'
                    }
                ]
            },
            {
                'hojinid': 'DEMO003',
                'name': 'Healthcare Innovation Corp',
                'keywords': ['healthcare technology', 'medical devices', 'digital health', 'telemedicine', 'health informatics'],
                'patents': [
                    {
                        'patent_id': 'DEMO2023004',
                        'abstract': 'Digital healthcare platform integrating medical devices with cloud-based analytics for remote patient monitoring and diagnosis.'
                    }
                ]
            }
        ]
        
        # Initialize suggester with lower threshold for demo
        demo_config = {
            'similarity_threshold': 0.1,  # Lower threshold for demo
            'top_k_suggestions': 3,
            'enable_openai_enhance': config.get('enable_openai_enhance', False),
            'max_combinations': PRODUCT_SUGGESTION_CONFIG['max_combinations'],  # FIXED: Use config value
            'max_keywords': PRODUCT_SUGGESTION_CONFIG['max_keywords'],
            'debug_enabled': False  # Disable debug for cleaner output
        }
        suggester = PatentProductSuggester(demo_config)
        
        # Generate suggestions
        results = suggester.suggest_products(config['query'], sample_companies)
        
        # Display results
        print(f"\nProduct Suggestions for: '{config['query'][:60]}...'")
        print(f"Companies processed: {len(results['results'])}")
        print(f"Companies with suggestions: {len([r for r in results['results'] if r['products']])}")
        print(f"Total products suggested: {sum(len(r['products']) for r in results['results'])}")
        
        for i, company in enumerate(results['results'], 1):
            print(f"\n{i}. {company['company_name']} (ID: {company['hojinid']})")
            print(f"   Company Similarity: {company['company_similarity']:.3f}")
            print(f"   Keyword Similarity: {company['keyword_similarity']:.3f}")
            print(f"   Patent Similarity: {company['patent_similarity']:.3f}")
            print(f"   Suggested Products:")
            for j, product in enumerate(company['products'], 1):
                print(f"      {j}. {product['product_name']}")
                print(f"         Score: {product['score']:.3f} | Lexical: {product['lexical_similarity']:.3f} | Semantic: {product['semantic_similarity']:.3f}")
        
        if not results['results']:
            print(f"\nNo suggestions found for this query.")
        
        # Export results
        output_path = suggester.export_results(results)
        print(f"\nResults saved to:")
        print(f"  JSON: {output_path}")
        
        # Text file should have same timestamp but different directory
        timestamp = output_path.split('_')[-1].split('.')[0]  # Extract timestamp
        text_path = f"data/submissions/product_suggestions_{timestamp}.txt"
        print(f"  Text: {text_path}")

if __name__ == "__main__":
    main()
    exit(1)  # Exit after running the pipeline
    
    
    