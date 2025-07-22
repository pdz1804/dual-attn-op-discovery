import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "CN_JP_US_data")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# === DATA NEED TO EXIST ===
US_WEB_DATA = os.path.join(DATA_DIR, "us_web_with_company.csv")
US_PATENT_DATA = os.path.join(DATA_DIR, "us_patent202506.csv")
US_PATENT_DATA_CLEANED = os.path.join(DATA_DIR, "cleaned_patents.csv")
FASTTEXT_VEC = os.path.join(DATA_DIR, "fasttext_web_patent.vec")

# FASTTEXT_VEC_ORI = os.path.join(DATA_DIR, "cc.en.300.vec")
FASTTEXT_VEC_ORI = FASTTEXT_VEC
# === END ===

# === OUTPUT === 
DUAL_ATT_SAVE = os.path.join(MODEL_DIR, "dual_attn_model_state.pth")
DUAL_ATT_OUTPUT = os.path.join(OUTPUT_DIR, "save_dual_attn_output.csv")
EMBEDDINGS_OUTPUT = os.path.join(OUTPUT_DIR, "full_company_embeddings_multilabel.csv")
FIRM_PRODUCT_KEYWORDS = os.path.join(DATA_DIR, "DualAtt_Firm_Product_Keywords_Table_with_Vector.csv")
COMPANY_EMBEDDINGS = os.path.join(OUTPUT_DIR, "company_embeddings.pkl")
# === END ===

# === NEW PATHS ===

# Sentence Transformer Paths
SENTENCE_TRANSFORMER_MODEL_DIR = os.path.join(MODEL_DIR, "sentence_transformers")
SENTENCE_TRANSFORMER_EMBEDDINGS = os.path.join(OUTPUT_DIR, "sentence_transformer_embeddings.pkl")

# RAG Related Paths
RAG_VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "vector_db")
RAG_COMPANY_DOCUMENTS = os.path.join(OUTPUT_DIR, "rag_company_documents.csv")
RAG_EXTERNAL_SUMMARIES = os.path.join(DATA_DIR, "company_external_summaries.csv")

# Enhanced Model Outputs
ENHANCED_PRODUCT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "enhanced_product_embeddings.csv")
ENHANCED_PATENT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "enhanced_patent_embeddings.csv")

# Streamlit App Data
STREAMLIT_DATA_DIR = os.path.join(BASE_DIR, "data", "streamlit_data")
STREAMLIT_CONFIG = os.path.join(BASE_DIR, "streamlit_config.json")

# === CLUSTERING PATHS ===

# Clustering Results Directory
CLUSTERING_DIR = os.path.join(BASE_DIR, "data", "clustering")

# Clustering Models and Results
CLUSTERING_MODELS_DIR = os.path.join(CLUSTERING_DIR, "models")
CLUSTERING_RESULTS_DIR = os.path.join(CLUSTERING_DIR, "results")

# Clustering evaluation results
CLUSTERING_EVALUATION_RESULTS = os.path.join(CLUSTERING_RESULTS_DIR, "evaluation_results.json")

# Best clustering model and assignments
CLUSTERING_BEST_MODEL = os.path.join(CLUSTERING_MODELS_DIR, "best_clustering_model.pkl")
CLUSTERING_ASSIGNMENTS = os.path.join(CLUSTERING_RESULTS_DIR, "cluster_assignments.csv")

# Cluster centers and metadata
CLUSTERING_CENTERS = os.path.join(CLUSTERING_RESULTS_DIR, "cluster_centers.npy")
CLUSTERING_METADATA = os.path.join(CLUSTERING_RESULTS_DIR, "clustering_metadata.json")

# Embedding-specific clustering files
def get_clustering_file_paths(embedding_type, country='US'):
    """Generate clustering file paths based on embedding type and country"""
    base_name = f"{country}_{embedding_type}"
    return {
        'evaluation_results': os.path.join(CLUSTERING_RESULTS_DIR, f"{base_name}_evaluation_results.json"),
        'best_model': os.path.join(CLUSTERING_MODELS_DIR, f"{base_name}_best_clustering_model.pkl"),
        'assignments': os.path.join(CLUSTERING_RESULTS_DIR, f"{base_name}_cluster_assignments.csv"),
        'centers': os.path.join(CLUSTERING_RESULTS_DIR, f"{base_name}_cluster_centers.npy"),
        'metadata': os.path.join(CLUSTERING_RESULTS_DIR, f"{base_name}_clustering_metadata.json"),
        'visualization': os.path.join(CLUSTERING_RESULTS_DIR, f"{base_name}_clustering_visualization.png"),
        'performance_plot': os.path.join(BASE_DIR, "data", "clustering", "img", f"{base_name}_clustering_performance.png"),
        'ranking_results': os.path.join(CLUSTERING_RESULTS_DIR, f"{base_name}_ranking_results.json")
    }

# === END NEW PATHS ===



