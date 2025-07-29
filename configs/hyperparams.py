MAX_PAGE = 32
MAX_LEN = 864
BATCH_SIZE = 10

EPOCHS_DUAL_ATT = 10

# Config for Dual Attention Model
# Note that this is the best config so far
HIDDEN_DIM = 300
EMBEDDING_DIM = 300
LABEL_DIM = 1
ATTN_TYPE = 'dot'  
ATTN_WORD = False
ATTN_PAGE = False
SCALE = 10
PAGE_SCALE = 10
SELECTED_KEYWORDS_N = 60

# Config for Transformer Matrix Model
COUNTRY = ['US']
EPOCHS_TRANS_MATRIX = 100

# Get only partial of the data for training only
TEST_SIZE = None

# === NEW CONFIGURATION OPTIONS ===

# Embedding Configuration
EMBEDDING_TYPE = 'fasttext'  # Options: 'fasttext', 'sentence_transformer'
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model name

# RAG Configuration
USE_RAG = False  # Whether to use RAG instead of transformation matrix
RAG_CHUNK_SIZE = 512  # Size of text chunks for RAG
RAG_CHUNK_OVERLAP = 50  # Overlap between chunks
RAG_TOP_K = 5  # Number of documents to retrieve in RAG
RAG_USE_EXTERNAL_SUMMARIES = False  # Use external company summaries instead of keywords

# UI Flow Configuration
UI_FLOW_TYPE = 'ml'  # Options: 'ml' (Machine Learning), 'rag' (RAG approach)
ENABLE_DUAL_ATTENTION = True  # Whether to use dual attention model
ENABLE_TRANSFORMATION_MATRIX = True  # Whether to use transformation matrix

# Vector Database Configuration
VECTOR_DB_TYPE = 'chromadb'  # Options: 'chromadb' (recommended)
VECTOR_DB_PATH = 'data/vector_db'  # Path to store vector database

# Embedding Dimensions (automatically determined based on embedding type)
def get_embedding_dimension(embedding_type, model_name=None):
    """Get embedding dimension based on embedding type"""
    if embedding_type == 'fasttext':
        return 300  # Standard FastText dimension
    elif embedding_type == 'sentence_transformer':
        if model_name == 'all-MiniLM-L6-v2':
            return 384  # Dimension for all-MiniLM-L6-v2
        elif model_name == 'all-mpnet-base-v2':
            return 768  # Dimension for all-mpnet-base-v2
        else:
            return 384  # Default for sentence transformers
    return 300  # Default fallback

# === CLUSTERING CONFIGURATION ===

# Enable/disable clustering pipeline
ENABLE_CLUSTERING = True

# Clustering algorithm options: 'kmeans', 'hierarchical', 'dbscan'
CLUSTERING_ALGORITHM = 'kmeans'

# Range of cluster numbers to test for hyperparameter tuning
CLUSTER_NUMBERS_RANGE = list(range(3, 101, 5))  # [3, 8, 13, 18, 23, 28, ..., 98]

# Alternative: specific numbers to test (recommended for large datasets)
# CLUSTER_NUMBERS_RANGE = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

# Alternative: smaller range for testing (uncomment if above fails)
# CLUSTER_NUMBERS_RANGE = [5, 10, 15, 20, 25, 30]

# Clustering evaluation metrics to use
CLUSTERING_METRICS = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

# Primary metric for selecting best cluster number (NOTE: This is now deprecated in favor of multi-metric ranking)
PRIMARY_CLUSTERING_METRIC = 'silhouette'  # Kept for backward compatibility

# Random state for reproducible clustering
CLUSTERING_RANDOM_STATE = 42

# Maximum iterations for K-means
CLUSTERING_MAX_ITER = 300

# Number of K-means initializations
CLUSTERING_N_INIT = 10

# Minimum samples for DBSCAN
DBSCAN_MIN_SAMPLES = 5

# Eps parameter for DBSCAN (will be auto-tuned if None)
DBSCAN_EPS = None

# === DISPLAY CONFIGURATION ===

# Maximum number of keywords to display in results
MAX_KEYWORDS_DISPLAY = 50  # Maximum number of keywords to show in results summary
KEYWORDS_PER_COMPANY_CLUSTER = 10  # Keywords to show per company in cluster display
COMPANIES_PER_CLUSTER_DISPLAY = 3  # Number of sample companies to show per cluster
TOP_K_COMPANIES_IN_CLUSTER = 5  # Number of top-k most relevant companies to retrieve in cluster

# === PRODUCT SUGGESTION CONFIGURATION ===

# Enable/disable product suggestion pipeline
ENABLE_PRODUCT_SUGGESTIONS = True

# Product suggestion model parameters
PRODUCT_SUGGESTION_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Sentence transformer model
    'alpha': 0.3,  # Weight for keyword vs patent similarity (0-1)
    'beta': 0.3,   # Weight for lexical vs semantic similarity (0-1) - Lower value favors semantic similarity
    'similarity_threshold': 0.075,  # Minimum similarity threshold for suggestions (lowered for real data)
    'top_k_suggestions': 5,  # Number of product suggestions per company
    'max_keywords': 20,  # Maximum keywords to extract per text
    'max_combinations': 5,  # Maximum keyword combinations for product generation
    'use_patent_data': True,  # Whether to use company patent data
    'enable_openai_enhance': False,  # Use OpenAI to enhance product names
    'openai_model': 'gpt-4o-mini',  # OpenAI model for enhancement
    'output_directory': 'data/suggestions',  # Directory to save suggestion results
    'submissions_directory': 'data/submissions',  # Directory to save text output files
    'debug_enabled': False  # Enable debug logging for product suggestions
}

# Patent processing limits for product suggestions
MAX_PATENTS_PER_COMPANY = 20  # Maximum patents to process per company for similarity computation
MAX_PATENTS_FOR_ANALYSIS = 20  # Maximum patents to analyze for keyword extraction

# Keyword extraction limits for different contexts
KEYWORD_EXTRACTION_LIMITS = {
    'similarity_computation': 15,  # Keywords for similarity comparison
    'patent_analysis': 10,  # Keywords per patent for analysis
    'theme_extraction': 5,  # Keywords for theme generation
    'top_keywords_limit': 15,  # Top keywords to keep from frequency analysis
    'theme_combinations': 5  # Max theme combinations to generate
}

# Note: Product name templates are now domain-specific and defined within the pipeline
