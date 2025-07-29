"""
Enhanced Streamlit Web Interface for FullFlow Patent-Product Matching System
with Advanced Clustering Analysis and Pipeline Management
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import logging
import time
import subprocess
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import our configurations and models
from configs.hyperparams import *
from configs.paths import *
from utils.seed_everything import set_seed
from utils.vector_utils import create_embedder
from utils.rag_utils import create_rag_processor
from pipelines.patent_product_pipeline import (
    initialize_embedder, get_embedding_file_paths, 
    load_representations_from_json,
    rag_only_pipeline
)
from pipelines.clustering_pipeline import load_clustering_analyzer, clustering_pipeline

# Import product suggestion pipeline
from pipelines.product_suggestion_pipeline import PatentProductSuggester, convert_pipeline_results_to_suggestions_format

# Configure logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="FullFlow: Patent-Product Matching System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FullFlowApp:
    """Enhanced Streamlit app with comprehensive clustering and pipeline management"""
    
    def __init__(self):
        self.embedder = None
        self.rag_processor = None
        self.ml_models = {}
        self.clustering_analyzer = None
        self.demo_data = self._create_demo_data()
        
        # Initialize session state
        if 'initialization_status' not in st.session_state:
            st.session_state.initialization_status = {}
        if 'embedder' not in st.session_state:
            st.session_state.embedder = None
        if 'rag_processor' not in st.session_state:
            st.session_state.rag_processor = None
        if 'ml_models' not in st.session_state:
            st.session_state.ml_models = {}
        if 'clustering_analyzer' not in st.session_state:
            st.session_state.clustering_analyzer = None
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

    def _create_demo_data(self):
        """Create static demo data for examples tab"""
        return {
            'sample_queries': [
                {
                    'query': 'machine learning algorithms for medical diagnosis',
                    'method': 'RAG (ChromaDB)',
                    'results': [
                        {
                            'company_name': 'MedTech AI Solutions',
                            'company_id': '123456',
                            'relevance_score': 0.8934,
                            'cluster': 15,
                            'keywords': ['artificial intelligence', 'medical devices', 'diagnostic systems', 'machine learning', 'deep learning', 'neural networks', 'computer vision', 'medical imaging', 'healthcare technology', 'patient monitoring', 'clinical decision support', 'medical diagnosis', 'predictive analytics', 'health informatics', 'telemedicine', 'electronic health records', 'medical software', 'biomedical engineering', 'digital health', 'precision medicine', 'medical algorithms', 'health data analysis', 'clinical workflows', 'medical automation', 'healthcare AI', 'diagnostic algorithms', 'medical data mining', 'clinical intelligence', 'health monitoring', 'medical prediction', 'healthcare analytics', 'clinical support systems', 'medical technology', 'health tech innovation', 'smart healthcare', 'connected health', 'medical IoT', 'healthcare platforms', 'clinical research', 'medical informatics', 'health data science', 'medical machine learning', 'clinical AI', 'healthcare automation', 'medical decision support', 'health analytics', 'clinical data analysis', 'medical innovation', 'healthcare solutions', 'digital medicine'],
                            'total_patents': 247,
                            'sample_patents': [
                                {
                                    'patent_id': 'US10234567',
                                    'abstract': 'An apparatus and method for automated medical diagnosis using deep learning neural networks to analyze medical imaging data...'
                                },
                                {
                                    'patent_id': 'US10345678',
                                    'abstract': 'A machine learning system for predicting patient outcomes based on clinical data and medical history...'
                                }
                            ]
                        },
                        {
                            'company_name': 'AI Diagnostics Corp',
                            'company_id': '234567',
                            'relevance_score': 0.8721,
                            'cluster': 15,
                            'keywords': ['computer vision', 'healthcare technology', 'diagnostic algorithms', 'image processing', 'pattern recognition', 'automated analysis', 'medical imaging', 'AI diagnostics', 'machine learning models', 'deep learning', 'convolutional neural networks', 'image classification', 'medical AI', 'diagnostic accuracy', 'clinical imaging', 'radiology AI', 'medical computer vision', 'automated diagnosis', 'healthcare automation', 'medical technology', 'digital pathology', 'image analysis', 'diagnostic tools', 'clinical decision support', 'medical software', 'healthcare AI solutions', 'diagnostic imaging', 'medical data analysis', 'clinical workflows', 'automated screening'],
                            'total_patents': 189,
                            'sample_patents': [
                                {
                                    'patent_id': 'US10456789',
                                    'abstract': 'Computer vision system for automated analysis of medical images using convolutional neural networks...'
                                }
                            ]
                        }
                    ],
                    'cluster_info': {
                        'nearest_cluster': 15,
                        'cluster_distance': 0.234,
                        'cluster_size': 87,
                        'cluster_companies': [
                            {'name': 'MedTech AI Solutions', 'keywords': ['AI', 'medical', 'diagnostics']},
                            {'name': 'Healthcare Robotics Inc', 'keywords': ['robotics', 'surgery', 'automation']},
                            {'name': 'BioAI Technologies', 'keywords': ['bioinformatics', 'genomics', 'AI']}
                        ]
                    }
                }
            ],
            'clustering_analysis': {
                'embedding_type': 'sentence_transformer',
                'best_clusters': 23,
                'total_rank_score': 4.0,
                'metrics': {
                    'silhouette': {'score': 0.2341, 'rank': 1},
                    'calinski': {'score': 450.23, 'rank': 2},
                    'davies': {'score': 1.234, 'rank': 1}
                },
                'cluster_distribution': [
                    {'cluster_id': 0, 'companies': 425, 'percentage': 6.3},
                    {'cluster_id': 1, 'companies': 312, 'percentage': 4.6},
                    {'cluster_id': 2, 'companies': 298, 'percentage': 4.4},
                    {'cluster_id': 3, 'companies': 276, 'percentage': 4.1},
                    {'cluster_id': 4, 'companies': 245, 'percentage': 3.6}
                ]
            }
        }
    
    def render_app(self):
        """Main app rendering function with tabs"""
        st.set_page_config(
            page_title="FullFlow Patent-Product Matching System",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🔬 FullFlow Patent-Product Matching System")
        st.markdown("**Advanced AI-powered system for patent-product matching with clustering analysis**")
        
        # Sidebar
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Query Interface", 
            "🎯 Clustering Analysis", 
            "⚙️ Pipeline Manager",
            "📊 Demo Examples", 
            "🔧 System Status",
            "📚 Documentation"
        ])
        
        with tab1:
            self.render_query_interface()
        
        with tab2:
            self.render_clustering_analysis()
            
        with tab3:
            self.render_pipeline_manager()
            
        with tab4:
            self.render_demo_examples()
            
        with tab5:
            self.render_system_status()
            
        with tab6:
            self.render_documentation()

    def render_sidebar(self):
        """Enhanced sidebar with better styling and configuration feedback"""
        st.sidebar.header("🔧 Configuration")
        
        # Configuration status indicator
        init_status = st.session_state.get('initialization_status', {})
        if any(init_status.values()):
            st.sidebar.success("✅ Components Initialized")
        else:
            st.sidebar.warning("⚠️ Components Not Initialized")
        
        # Flow type selection with clear description
        st.sidebar.subheader("🔀 Flow Selection")
        flow_type = st.sidebar.selectbox(
            "Select Flow Type",
            ["ML (Machine Learning)", "RAG (Retrieval-Augmented Generation)"],
            help="Choose between ML approach (transformation matrix) or RAG approach (semantic search)"
        )
        
        # Visual indicator of selected flow
        if "ML" in flow_type:
            st.sidebar.info("🧠 **ML Flow**: Uses transformation matrices for structured queries")
        else:
            st.sidebar.info("🔍 **RAG Flow**: Uses semantic search for natural language queries")
        
        # ML configuration (if ML flow selected)
        if "ML" in flow_type:
            st.sidebar.subheader("🧠 ML Configuration")
            ml_mode = st.sidebar.selectbox(
                "ML Mode",
                ["Matrix (Linear)", "Model (Neural Network)"],
                help="Matrix mode uses linear transformation, Model mode uses neural networks"
            )
            
            # New 23/07/2025: Choose model type (linear or mlp)
            model_type = st.sidebar.selectbox(
                "Transformation Model Type",
                ["linear", "mlp"],
                help="Select the type of transformation model"
            )

            # Show approx_method only if model_type is not 'linear'
            if model_type != "linear":
                approx_method = st.sidebar.selectbox(
                    "Approximation Method for Matrix Extraction",
                    ["sampling", "polynomial", "kernel"],
                    help="Select the approximation method to estimate the transformation matrix"
                )
            else:
                approx_method = None  # Not needed for linear
            
            # Visual feedback for ML mode choice
            if "Matrix" in ml_mode:
                st.sidebar.info("🔢 **Matrix Mode**: Linear transformation (faster)")
            else:
                st.sidebar.info("🧠 **Model Mode**: Neural networks (more complex)")
        else:
            ml_mode = "Matrix (Linear)"
            model_type = "linear"  # Default for RAG-only mode
            approx_method = None   # Default for RAG-only mode
        
        # Embedding configuration with clear feedback
        st.sidebar.subheader("📊 Embedding Configuration")
        embedding_type = st.sidebar.selectbox(
            "Embedding Type",
            ["fasttext", "sentence_transformer"],
            help="Choose embedding method: FastText (faster) or Sentence Transformers (better semantic understanding)"
        )
        
        # Visual feedback for embedding choice
        if embedding_type == "fasttext":
            st.sidebar.info("⚡ **FastText**: Faster, lower memory usage")
        else:
            st.sidebar.info("🎯 **Sentence Transformers**: Better semantic understanding")
        
        # Sentence transformer model (if applicable)
        if embedding_type == "sentence_transformer":
            sentence_model = st.sidebar.selectbox(
                "Sentence Transformer Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                help="Choose sentence transformer model"
            )
        else:
            sentence_model = "all-MiniLM-L6-v2"
        
        # Country selection
        st.sidebar.subheader("🌍 Data Selection")
        countries = st.sidebar.multiselect(
            "Select Countries",
            ["US", "CN", "JP"],
            default=["US"],
            help="Choose countries for analysis"
        )
        
        # RAG configuration (if RAG flow selected)
        if "RAG" in flow_type:
            st.sidebar.subheader("🔍 RAG Configuration")
            rag_top_k = st.sidebar.slider("RAG Top K Results", 1, 20, 5)
            rag_use_external = st.sidebar.checkbox(
                "Use External Summaries",
                help="Use curated company summaries instead of dual attention keywords"
            )
        else:
            rag_top_k = 5
            rag_use_external = False
        
        # Clustering configuration with visual feedback
        st.sidebar.subheader("🎯 Clustering Configuration")
        enable_clustering = st.sidebar.checkbox(
            "Enable Clustering Analysis",
            value=True,
            help="Show cluster information in query results"
        )
        
        if enable_clustering:
            clustering_analyzer = st.session_state.get('clustering_analyzer')
            if clustering_analyzer:
                st.sidebar.success(f"✅ Clustering loaded ({clustering_analyzer.best_n_clusters} clusters)")
            else:
                st.sidebar.warning("⚠️ Clustering not loaded")
        
        clustering_algorithm = st.sidebar.selectbox(
            "Clustering Algorithm",
            ["kmeans", "hierarchical", "dbscan"],
            help="Choose clustering algorithm for analysis"
        )
        
        # Product suggestion configuration with visual feedback
        st.sidebar.subheader("🛍️ Product Suggestion Configuration")
        enable_product_suggestions = st.sidebar.checkbox(
            "Enable Product Suggestions",
            value=True,
            help="Generate AI-powered product suggestions for each company based on their patents and keywords"
        )
        
        if enable_product_suggestions:
            st.sidebar.info("🎯 **Product Suggestions**: AI-powered recommendations based on patents and keywords")
            
            with st.sidebar.expander("🔧 Product Suggestion Settings", expanded=False):
                suggestion_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.01,
                    max_value=0.5,
                    value=PRODUCT_SUGGESTION_CONFIG['similarity_threshold'],
                    step=0.01,
                    help="Minimum similarity score for companies to get product suggestions"
                )
                
                max_suggestions_per_company = st.slider(
                    "Max Suggestions per Company",
                    min_value=1,
                    max_value=10,
                    value=PRODUCT_SUGGESTION_CONFIG['top_k_suggestions'],
                    help="Maximum number of product suggestions per company"
                )
                
                enable_openai_enhance = st.checkbox(
                    "Use OpenAI Enhancement",
                    value=PRODUCT_SUGGESTION_CONFIG['enable_openai_enhance'],
                    help="Use OpenAI to enhance product names (requires API key in .env file)"
                )
                
                max_combinations_setting = st.slider(
                    "Max Keyword Combinations",
                    min_value=2,
                    max_value=10,
                    value=PRODUCT_SUGGESTION_CONFIG['max_combinations'],
                    help="Maximum number of keywords to combine (2=pairs only, 5=up to 5-word combinations)"
                )
        else:
            suggestion_threshold = PRODUCT_SUGGESTION_CONFIG['similarity_threshold']
            max_suggestions_per_company = PRODUCT_SUGGESTION_CONFIG['top_k_suggestions']
            enable_openai_enhance = PRODUCT_SUGGESTION_CONFIG['enable_openai_enhance']
            max_combinations_setting = PRODUCT_SUGGESTION_CONFIG['max_combinations']
        
        # Advanced options in an expander
        with st.sidebar.expander("🔧 Advanced Options"):
            force_rebuild_rag = st.checkbox(
                "Force Rebuild RAG Database",
                help="Force rebuild the RAG vector database"
            )
            
            force_rebuild_clustering = st.checkbox(
                "Force Rebuild Clustering",
                help="Force rebuild clustering models"
            )
            
            max_keywords_display = st.slider(
                "Max Keywords Display",
                min_value=10,
                max_value=100,
                value=MAX_KEYWORDS_DISPLAY,
                step=10,
                help="Maximum number of keywords to show in results (default: 50)"
            )
        
        # Store config in session state
        config = {
            'flow_type': 'rag' if 'RAG' in flow_type else 'ml',
            'ml_mode': 'matrix' if 'Matrix' in ml_mode else 'model',
            'embedding_type': embedding_type,
            'sentence_transformer_model': sentence_model,
            'countries': countries,
            'rag_top_k': rag_top_k,
            'rag_use_external_summaries': rag_use_external,
            'enable_clustering': enable_clustering,
            'clustering_algorithm': clustering_algorithm,
            'force_rebuild_rag': force_rebuild_rag,
            'force_rebuild_clustering': force_rebuild_clustering,
            'max_keywords_display': max_keywords_display,
            'model_type': model_type,
            'approx_method': approx_method,
            # Add display configuration constants from hyperparams
            'keywords_per_company_cluster': KEYWORDS_PER_COMPANY_CLUSTER,
            'companies_per_cluster_display': COMPANIES_PER_CLUSTER_DISPLAY, 
            'top_k_companies_in_cluster': TOP_K_COMPANIES_IN_CLUSTER,
            # Product suggestion configuration
            'enable_product_suggestions': enable_product_suggestions,
            'product_similarity_threshold': suggestion_threshold,
            'max_suggestions_per_company': max_suggestions_per_company,
            'enable_openai_enhance': enable_openai_enhance,
            'max_combinations_setting': max_combinations_setting,
            'max_patents_per_company': MAX_PATENTS_PER_COMPANY,
            'max_abstract_length': 300
        }
        
        st.session_state.config = config
        
        # System control with better visual feedback
        st.sidebar.subheader("🚀 System Control")
        
        # Show current configuration summary
        with st.sidebar.expander("📋 Current Config Summary", expanded=False):
            st.write(f"**Flow:** {flow_type.split(' ')[0]}")
            if "ML" in flow_type:
                st.write(f"**ML Mode:** {ml_mode.split(' ')[0]}")
            st.write(f"**Embedding:** {embedding_type}")
            st.write(f"**Countries:** {', '.join(countries)}")
            st.write(f"**Clustering:** {'✅' if enable_clustering else '❌'}")
            st.write(f"**Keywords Display:** {max_keywords_display}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Initialize", use_container_width=True, type="primary"):
                self.initialize_components(config)
        
        with col2:
            if st.button("🗑️ Reset", use_container_width=True):
                self.reset_components()

    def initialize_components(self, config):
        """Enhanced initialization with better progress feedback"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize embedder
            status_text.text("🔄 Initializing embedder...")
            progress_bar.progress(20)
            
            config['country'] = config['countries'][0] if config['countries'] else 'US'
            embedder, use_enhanced = initialize_embedder(config)
            st.session_state.embedder = embedder
            st.session_state.initialization_status['embedder'] = True
            
            status_text.text("✅ Embedder initialized")
            progress_bar.progress(40)
            
            # Step 2: Initialize RAG processor if needed
            if config['flow_type'] == 'rag':
                status_text.text("🔄 Initializing RAG processor...")
                progress_bar.progress(60)
                
                try:
                    # Note: RAG processor will be created fresh during query execution
                    # This is to ensure it uses the correct embedder and configuration
                    # Just mark as available for now
                    st.session_state.initialization_status['rag_processor'] = True
                    
                    # Show RAG configuration info
                    rag_mode = "External Summaries" if config.get('rag_use_external_summaries', False) else "Dual Attention Keywords"
                    status_text.text(f"✅ RAG processor ready (Mode: {rag_mode})")
                    
                except Exception as e:
                    st.warning(f"⚠️ RAG processor initialization failed: {str(e)}")
                    st.info("💡 You may need to build the RAG database first using the Pipeline Manager")
                    st.session_state.initialization_status['rag_processor'] = False
            
            progress_bar.progress(70)
            
            # Step 3: Initialize clustering analyzer if enabled
            if config['enable_clustering']:
                status_text.text("🔄 Loading clustering analyzer...")
                progress_bar.progress(80)
                
                clustering_analyzer = load_clustering_analyzer(
                    config['embedding_type'], 
                    config['countries'][0] if config['countries'] else 'US'
                )
                st.session_state.clustering_analyzer = clustering_analyzer
                st.session_state.initialization_status['clustering_analyzer'] = clustering_analyzer is not None
                
                if clustering_analyzer:
                    status_text.text(f"✅ Clustering analyzer loaded ({clustering_analyzer.best_n_clusters} clusters)")
                else:
                    st.warning("⚠️ Clustering analyzer not available. You can run clustering from Pipeline Manager.")
            
            progress_bar.progress(90)
            
            # Step 4: Load ML models if needed
            if config['flow_type'] == 'ml':
                status_text.text("🔄 Checking ML models...")
                # For now, just mark as loaded - actual ML implementation pending
                st.session_state.ml_models = {'loaded': True}
                st.session_state.initialization_status['ml_models'] = True
                status_text.text("✅ ML models checked")
            
            progress_bar.progress(100)
            
            # Step 5: Save configuration
            config_path = os.path.join(os.getcwd(), "streamlit_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            st.session_state.initialization_status['config_saved'] = True
            
            status_text.text("🎉 All components initialized successfully!")
            
            # Show summary of what was initialized
            with st.expander("📋 Initialization Summary", expanded=True):
                st.success(f"✅ **Embedder**: {config['embedding_type']} initialized")
                
                if config['flow_type'] == 'rag':
                    if st.session_state.initialization_status.get('rag_processor'):
                        rag_mode = "External Summaries" if config.get('rag_use_external_summaries', False) else "Dual Attention Keywords"
                        st.success(f"✅ **RAG Processor**: Ready for queries (Mode: {rag_mode})")
                    else:
                        st.warning("⚠️ **RAG Processor**: Failed to initialize")
                
                if config['enable_clustering']:
                    if st.session_state.get('clustering_analyzer'):
                        st.success(f"✅ **Clustering**: {st.session_state.clustering_analyzer.best_n_clusters} clusters loaded")
                    else:
                        st.warning("⚠️ **Clustering**: Not available")
                
                st.info(f"📊 **Configuration**: {config['flow_type'].upper()} flow with {config['embedding_type']} embeddings")
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            logger.error(f"Initialization error: {e}")
            progress_bar.empty()
            status_text.empty()

    def reset_components(self):
        """Reset all components"""
        st.session_state.embedder = None
        st.session_state.rag_processor = None
        st.session_state.ml_models = {}
        st.session_state.clustering_analyzer = None
        st.session_state.initialization_status = {}
        st.session_state.pipeline_results = {}
        st.success("🗑️ All components reset")
        st.rerun()

    def render_query_interface(self):
        """Enhanced query interface with better validation and feedback"""
        st.header("🔍 Query Interface")
        
        # Check if components are initialized with detailed feedback
        init_status = st.session_state.get('initialization_status', {})
        config = st.session_state.get('config', {})
        
        if not init_status.get('embedder'):
            st.error("❌ **Components not initialized!**")
            st.info("👉 Please initialize components using the sidebar first.")
            
            # Show what needs to be done
            st.markdown("**Required steps:**")
            st.markdown("1. Configure your settings in the sidebar")
            st.markdown("2. Click '🔄 Initialize' button")
            st.markdown("3. Wait for initialization to complete")
            st.markdown("4. Return here to run queries")
            return
        
        # Show current configuration
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                flow_type = config.get('flow_type', 'ml').upper()
                st.metric("🔀 Flow Type", flow_type)
            
            with col2:
                embedding_type = config.get('embedding_type', 'fasttext').title()
                st.metric("📊 Embedding", embedding_type)
            
            with col3:
                clustering_status = "✅ Enabled" if config.get('enable_clustering') else "❌ Disabled"
                st.metric("🎯 Clustering", clustering_status)
            
            with col4:
                rag_status = "✅ Ready" if init_status.get('rag_processor') else "❌ Not Ready"
                if config.get('flow_type') == 'rag':
                    # Show RAG mode configuration
                    if init_status.get('rag_processor'):
                        rag_mode = "Ext. Sum." if config.get('rag_use_external_summaries') else "Keywords"
                        st.metric("🔍 RAG Status", f"Ready ({rag_mode})")
                    else:
                        st.metric("🔍 RAG Status", "Not Ready")
                else:
                    st.metric("🧠 ML Status", "Ready" if init_status.get('embedder') else "Not Ready")
        
        st.divider()
        
        # Query input with better styling
        st.subheader("💬 Enter your query")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Product Query or Patent Abstract:",
                placeholder="Enter your product query or patent abstract here...\n\nExample: 'machine learning algorithms for medical diagnosis'",
                height=120,
                help="Enter a detailed description of the technology or product you're looking for"
            )
        
        with col2:
            st.markdown("**🚀 Quick Examples:**")
            
            examples = [
                ("🤖 AI Medical", "machine learning algorithms for medical diagnosis"),
                ("🔋 Energy Tech", "renewable energy storage systems"),
                ("🚗 Autonomous Vehicles", "autonomous vehicle navigation systems"),
                ("🧬 Biotech", "gene therapy delivery mechanisms")
            ]
            
            for label, example_query in examples:
                if st.button(label, use_container_width=True):
                    st.session_state.example_query = example_query
                    st.rerun()
            
            # Use example query if set
            if hasattr(st.session_state, 'example_query'):
                query = st.session_state.example_query
                delattr(st.session_state, 'example_query')
        
        # Query validation and execution
        query_valid = query.strip() != ""
        
        if query_valid:
            st.success(f"✅ Query ready: {len(query.split())} words")
        else:
            st.warning("⚠️ Please enter a query to proceed")
        
        # Enhanced run button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_button = st.button(
                "🔍 Run Query", 
                type="primary", 
                use_container_width=True,
                disabled=not query_valid
            )
        
        if run_button and query_valid:
            self.execute_query(query)

    def execute_query(self, query):
        """Execute query with clustering information using actual initialized components"""
        config = st.session_state.get('config', {})
        
        # Check if components are properly initialized
        if not st.session_state.get('embedder'):
            st.error("❌ Embedder not initialized. Please initialize components first.")
            return
        
        with st.spinner("🔍 Processing query..."):
            try:
                # Get the actual method being used
                flow_type = config.get('flow_type', 'ml')
                embedding_type = config.get('embedding_type', 'fasttext')
                
                if flow_type == 'rag':
                    # RAG approach - implement using complete chat_pipeline logic
                    try:
                        # Import necessary functions
                        from utils.rag_utils import create_rag_processor
                        import pandas as pd
                        import numpy as np
                        
                        st.info("🔍 **RAG Mode**: Processing query using semantic search with ChromaDB...")
                        
                        # Setup status indicators
                        status_placeholder = st.empty()
                        status_placeholder.text("🔄 Initializing RAG processor...")
                        
                        # Get configuration
                        country = config.get('countries', ['US'])[0]
                        embedder = st.session_state.get('embedder')
                        
                        # Create RAG processor with current embedder
                        rag_processor = create_rag_processor(embedder, force_rebuild=config.get('force_rebuild_rag', False))
                        
                        status_placeholder.text("📊 Loading company data...")
                        
                        # Load company data for displaying keywords (same as chat_pipeline)
                        company_df = None
                        if os.path.exists(EMBEDDINGS_OUTPUT):
                            company_df = pd.read_csv(EMBEDDINGS_OUTPUT)
                            
                            # Load company names like in chat_pipeline
                            us_web_data = pd.read_csv(US_WEB_DATA)
                            company_name_map = us_web_data[['hojin_id', 'company_name']].drop_duplicates()
                            company_df = company_df.merge(company_name_map, on='hojin_id', how='left')
                        
                        status_placeholder.text("🎯 Processing query...")
                        
                        # Execute RAG query
                        rag_top_k = config.get('rag_top_k', 5)
                        rag_results = rag_processor.rag_query_companies(query, rag_top_k)
                        
                        # Get query embedding for cluster analysis (same as chat_pipeline)
                        query_embedding = None
                        clustering_analyzer = st.session_state.get('clustering_analyzer')
                        
                        if clustering_analyzer is not None:
                            try:
                                # Always use encode_text since we now use EnhancedEmbedder consistently
                                query_embedding = embedder.encode_text(query)
                            except Exception as e:
                                logger.warning(f"Could not compute query embedding: {e}")
                        
                        status_placeholder.text("🔍 Formatting results...")
                        
                        # Format results for Streamlit display (enhanced version)
                        results = {
                            'query': query,
                            'method': f'RAG (ChromaDB) - {embedding_type.title()}',
                            'results': []
                        }
                        
                        # Process RAG results with enhanced company information
                        if rag_results:
                            # Load patent data for patent counts and samples
                            patent_df = None
                            firm_patent_ids = {}
                            patent_text_map = {}
                            
                            try:
                                status_placeholder.text("📜 Loading patent data for counts...")
                                patent_file = US_PATENT_DATA_CLEANED  # US patents
                                patent_df = pd.read_csv(patent_file)
                                
                                # Process patent IDs and text mapping
                                for firm_id, group in patent_df.groupby('hojin_id'):
                                    firm_id = str(firm_id)
                                    firm_patent_ids[firm_id] = group['appln_id'].tolist()
                                    for app_id, abs_text in zip(group['appln_id'], group['clean_abstract'].dropna()):
                                        patent_text_map[app_id] = abs_text
                            except Exception as e:
                                logger.warning(f"Could not load patent data: {e}")
                            
                            for result in rag_results:
                                company_id = str(result.get('company_id', result.get('hojin_id', 'N/A')))
                                company_name = result.get('company_name', 'Unknown')
                                
                                # Fix score field mapping - RAG returns 'relevance_score', not 'similarity_score'
                                relevance_score = result.get('relevance_score', result.get('score', 0.0))
                                
                                # Get enhanced company information from company_df
                                keywords = result.get('keywords', [])
                                
                                # Try to get additional info from company_df if available
                                if company_df is not None and company_id != 'N/A':
                                    try:
                                        company_row = company_df[company_df['hojin_id'].astype(str) == company_id]
                                        if not company_row.empty:
                                            # Get keywords from company data if not in RAG result
                                            if not keywords and 'company_keywords' in company_row.columns:
                                                keywords_str = company_row.iloc[0]['company_keywords']
                                                if pd.notna(keywords_str):
                                                    keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                                    keywords = [k.strip() for k in keywords if k.strip()]
                                            
                                            # Get company name if not available
                                            if company_name == 'Unknown' and 'company_name' in company_row.columns:
                                                company_name = company_row.iloc[0]['company_name']
                                    except Exception as e:
                                        logger.warning(f"Could not enhance company info for {company_id}: {e}")
                                
                                # Get patent information from loaded patent data
                                total_patents = 0
                                sample_patents = []
                                
                                if company_id in firm_patent_ids:
                                    total_patents = len(firm_patent_ids[company_id])
                                    # Get sample patents (top 3)
                                    patent_ids = firm_patent_ids[company_id][:3]
                                    for patent_id in patent_ids:
                                        if patent_id in patent_text_map:
                                            abstract_text = patent_text_map[patent_id]
                                            sample_patents.append({
                                                'patent_id': patent_id,
                                                'abstract': abstract_text[:300] + "..." if len(abstract_text) > 300 else abstract_text
                                            })
                                
                                # Get cluster information if available
                                cluster_assignment = "N/A"
                                if clustering_analyzer and clustering_analyzer.cluster_assignments is not None:
                                    try:
                                        # Try exact company_id match first
                                        if company_id in clustering_analyzer.company_ids:
                                            company_idx = clustering_analyzer.company_ids.index(company_id)
                                            cluster_assignment = clustering_analyzer.cluster_assignments[company_idx]
                                        else:
                                            # Try converting to int if company_id is numeric string
                                            try:
                                                company_id_int = str(int(company_id))
                                                if company_id_int in clustering_analyzer.company_ids:
                                                    company_idx = clustering_analyzer.company_ids.index(company_id_int)
                                                    cluster_assignment = clustering_analyzer.cluster_assignments[company_idx]
                                                else:
                                                    # Try with original hojin_id from result if available
                                                    original_id = str(result.get('hojin_id', ''))
                                                    if original_id and original_id in clustering_analyzer.company_ids:
                                                        company_idx = clustering_analyzer.company_ids.index(original_id)
                                                        cluster_assignment = clustering_analyzer.cluster_assignments[company_idx]
                                            except ValueError:
                                                pass
                                    except Exception as e:
                                        logger.warning(f"Could not get cluster assignment for {company_id}: {e}")
                                
                                company_result = {
                                    'company_name': company_name,
                                    'company_id': company_id,
                                    'relevance_score': relevance_score,
                                    'keywords': keywords,
                                    'total_patents': total_patents,
                                    'sample_patents': sample_patents,
                                    'cluster': cluster_assignment
                                }
                                results['results'].append(company_result)
                        
                        # Add clustering info if available (same as chat_pipeline)
                        if clustering_analyzer and config.get('enable_clustering') and query_embedding is not None:
                            try:
                                cluster_id, distance = clustering_analyzer.find_nearest_cluster(query_embedding)
                                cluster_info = clustering_analyzer.get_cluster_info(cluster_id)
                                
                                results['cluster_info'] = {
                                    'nearest_cluster': cluster_id,
                                    'cluster_distance': distance,
                                    'cluster_size': cluster_info.get('n_companies', 0),
                                    'cluster_companies': []
                                }
                                
                                # Get top-k most relevant companies from cluster instead of samples
                                try:
                                    if hasattr(clustering_analyzer, 'embeddings_matrix') and clustering_analyzer.embeddings_matrix is not None:
                                        # Use config value for number of companies to show
                                        top_k_companies_count = config.get('top_k_companies_in_cluster', TOP_K_COMPANIES_IN_CLUSTER)
                                        top_k_companies = clustering_analyzer.get_top_k_companies_in_cluster(
                                            cluster_id=cluster_id,
                                            query_embedding=query_embedding,
                                            k=top_k_companies_count,
                                            embeddings_dict=None  # Use stored embeddings matrix
                                        )
                                        
                                        if top_k_companies:
                                            for company in top_k_companies:
                                                company_name = company.get('company_name', 'Unknown')
                                                keywords_str = str(company.get('keywords', ''))
                                                similarity_score = company.get('similarity_score', 0.0)
                                                
                                                # Parse keywords
                                                if keywords_str and keywords_str != 'nan' and keywords_str != '':
                                                    keywords = keywords_str.split('|')[:3]
                                                else:
                                                    keywords = []
                                                
                                                results['cluster_info']['cluster_companies'].append({
                                                    'name': company_name,
                                                    'keywords': keywords,
                                                    'similarity_score': similarity_score,
                                                    'rank_in_cluster': company.get('rank_in_cluster', 0)
                                                })
                                        else:
                                            # Fallback to sample companies
                                            if 'companies' in cluster_info:
                                                for _, company in cluster_info['companies'].head(3).iterrows():
                                                    results['cluster_info']['cluster_companies'].append({
                                                        'name': company.get('company_name', 'Unknown'),
                                                        'keywords': str(company.get('company_keywords', '')).split('|')[:3],
                                                        'similarity_score': None,
                                                        'rank_in_cluster': None
                                                    })
                                    else:
                                        # Fallback to sample companies if embeddings matrix not available
                                        if 'companies' in cluster_info:
                                            for _, company in cluster_info['companies'].head(3).iterrows():
                                                results['cluster_info']['cluster_companies'].append({
                                                    'name': company.get('company_name', 'Unknown'),
                                                    'keywords': str(company.get('company_keywords', '')).split('|')[:3],
                                                    'similarity_score': None,
                                                    'rank_in_cluster': None
                                                })
                                
                                except Exception as e:
                                    logger.warning(f"Could not retrieve top-k companies for cluster, using samples: {e}")
                                    # Fallback to sample companies
                                    if 'companies' in cluster_info:
                                        for _, company in cluster_info['companies'].head(3).iterrows():
                                            results['cluster_info']['cluster_companies'].append({
                                                'name': company.get('company_name', 'Unknown'),
                                                'keywords': str(company.get('company_keywords', '')).split('|')[:3],
                                                'similarity_score': None,
                                                'rank_in_cluster': None
                                            })
                            except Exception as e:
                                logger.warning(f"Could not determine cluster information: {e}")
                        
                        status_placeholder.empty()
                        
                    except Exception as e:
                        st.error(f"❌ RAG query failed: {str(e)}")
                        st.info("💡 Make sure you have built the RAG database first using Pipeline Manager")
                        logger.error(f"RAG query error: {e}")
                        return
                
                else:
                    # ML approach - implement using existing chat_pipeline logic
                    try:
                        # Import necessary functions
                        from pipelines.patent_product_pipeline import get_embedding_file_paths, load_representations_from_json
                        from inference.query_opportunity_matrix import query_opportunity_product_matrix_only
                        from inference.query_opportunity_best import query_opportunity_product_best
                        from models.patent2product import Patent2Product
                        from models.product2patent import Product2Patent
                        from ast import literal_eval
                        import torch
                        import pandas as pd
                        import numpy as np
                        
                        st.info("🧠 **ML Mode**: Processing query using transformation matrices and neural networks...")
                        
                        # Get configuration
                        country = config.get('countries', ['US'])[0]
                        
                        # Get embedding-specific file paths
                        file_paths = get_embedding_file_paths(embedding_type, country)
                        
                        # Load data
                        status_placeholder = st.empty()
                        status_placeholder.text("📊 Loading company data...")
                        
                        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
                        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
                        
                        # Load company names
                        us_web_data = pd.read_csv(US_WEB_DATA)
                        company_name_map = us_web_data[['hojin_id', 'company_name']].drop_duplicates()
                        product_df = product_df.merge(company_name_map, on='hojin_id', how='left')
                        
                        status_placeholder.text("🔄 Loading embeddings...")
                        
                        # Load product representations
                        product_rep = load_representations_from_json(file_paths['product_rep'])
                        if not product_rep:
                            st.error("❌ Product representations not found. Please train the models first.")
                            st.info("💡 Use Pipeline Manager to run: patent_product --mode train")
                            return
                        
                        # Load patent data
                        status_placeholder.text("📜 Loading patent data...")
                        patent_file = US_PATENT_DATA_CLEANED  # US patents
                        patent_df = pd.read_csv(patent_file)
                        
                        # Load patent representations
                        patent_rep = load_representations_from_json(file_paths['patent_rep'])
                        if not patent_rep:
                            st.error("❌ Patent representations not found. Please train the models first.")
                            return
                        
                        # Process patent IDs and text mapping
                        firm_patent_ids, patent_text_map = {}, {}
                        for firm_id, group in patent_df.groupby('hojin_id'):
                            firm_id = str(firm_id)
                            firm_patent_ids[firm_id] = group['appln_id'].tolist()
                            for app_id, abs_text in zip(group['appln_id'], group['clean_abstract'].dropna()):
                                patent_text_map[app_id] = abs_text
                        
                        status_placeholder.text("🔧 Loading models and matrices...")
                        
                        # Get embedding dimension
                        embedding_dim = 300  # Default
                        if patent_rep:
                            sample_embedding = next(iter(patent_rep.values()))
                            embedding_dim = len(sample_embedding)
                        
                        # Check for matrices and models
                        matrices_available = (os.path.exists(file_paths['matrix_A']) and 
                                            os.path.exists(file_paths['matrix_B']))
                        models_available = (os.path.exists(file_paths['model_A']) and 
                                          os.path.exists(file_paths['model_B']))
                        
                        if not matrices_available:
                            st.error("❌ Transformation matrices not found. Please train the models first.")
                            st.info("💡 Use Pipeline Manager to run: patent_product --mode train")
                            return
                        
                        # Load matrices
                        A_matrix = np.load(file_paths['matrix_A'])
                        B_matrix = np.load(file_paths['matrix_B'])
                        
                        # Load neural network models if available
                        model_A, model_B = None, None
                        if models_available:
                            model_A = Patent2Product(dim=embedding_dim)
                            model_B = Product2Patent(dim=embedding_dim)
                            model_A.load_state_dict(torch.load(file_paths['model_A']))
                            model_B.load_state_dict(torch.load(file_paths['model_B']))
                            model_A.eval()
                            model_B.eval()
                        
                        status_placeholder.text("🎯 Processing query...")
                        
                        # Get query embedding for clustering
                        query_embedding = None
                        clustering_analyzer = st.session_state.get('clustering_analyzer')
                        embedder = st.session_state.get('embedder')
                        
                        if clustering_analyzer and embedder:
                            try:
                                # Always use encode_text since we now use EnhancedEmbedder consistently
                                query_embedding = embedder.encode_text(query)
                            except Exception as e:
                                logger.warning(f"Could not compute query embedding: {e}")
                        
                        # Choose mode based on availability
                        # Get mode from config
                        requested_mode = config.get('ml_mode', 'matrix')
                        
                        if models_available or requested_mode == "matrix":
                            mode = requested_mode
                            if requested_mode == "model" and not models_available:
                                st.warning("⚠️ Model mode requested but neural network models not found. Falling back to Matrix mode.")
                                mode = "matrix"
                        else:
                            mode = "matrix"
                            st.info("ℹ️ Using Matrix mode (Neural network models not found)")
                        
                        # Execute query based on mode
                        top_k = 5
                        
                        if mode == "matrix":
                            st.info("🔢 **Matrix Mode**: Using linear transformation matrices")
                            
                            results_matrix = query_opportunity_product_matrix_only(
                                product_query_text=query.lower(),
                                ft_model=embedder,
                                mat_B=B_matrix,
                                mat_A=A_matrix,
                                patent_rep_dict=patent_rep,
                                product_rep_dict=product_rep,
                                firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
                                firm_patent_ids=firm_patent_ids,
                                patent_text_map=patent_text_map,
                                top_k=top_k
                            )
                            
                            # Format results for Streamlit
                            results = {
                                'query': query,
                                'method': f'ML-Matrix ({embedding_type.title()})',
                                'results': []
                            }
                            
                            if results_matrix:
                                for result in results_matrix:
                                    # Add keywords to result
                                    firm_id = str(result.get('firm_id', ''))
                                    company_row = product_df[product_df['Firm ID'] == firm_id]
                                    keywords = []
                                    if not company_row.empty and 'company_keywords' in company_row.columns:
                                        keywords_str = company_row.iloc[0]['company_keywords']
                                        if pd.notna(keywords_str):
                                            keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                            keywords = [k.strip() for k in keywords if k.strip()]
                                    
                                    # Get sample patents
                                    sample_patents = []
                                    if firm_id in firm_patent_ids:
                                        patent_ids = firm_patent_ids[firm_id][:3]  # Top 3 patents
                                        for patent_id in patent_ids:
                                            if patent_id in patent_text_map:
                                                sample_patents.append({
                                                    'patent_id': patent_id,
                                                    'abstract': patent_text_map[patent_id][:300] + "..." if len(patent_text_map[patent_id]) > 300 else patent_text_map[patent_id]
                                                })
                                    
                                    # Get cluster assignment
                                    cluster_assignment = "N/A"
                                    if clustering_analyzer and clustering_analyzer.cluster_assignments is not None:
                                        try:
                                            # Try exact company_id match first
                                            if firm_id in clustering_analyzer.company_ids:
                                                company_idx = clustering_analyzer.company_ids.index(firm_id)
                                                cluster_assignment = clustering_analyzer.cluster_assignments[company_idx]
                                            else:
                                                # Try converting to int if company_id is numeric string
                                                try:
                                                    company_id_int = str(int(firm_id))
                                                    if company_id_int in clustering_analyzer.company_ids:
                                                        company_idx = clustering_analyzer.company_ids.index(company_id_int)
                                                        cluster_assignment = clustering_analyzer.cluster_assignments[company_idx]
                                                except ValueError:
                                                    pass
                                        except Exception as e:
                                            logger.warning(f"Could not get cluster assignment for {firm_id}: {e}")
                                    
                                    company_result = {
                                        'company_name': result.get('firm_name', 'Unknown'),
                                        'company_id': firm_id,
                                        'relevance_score': result.get('cosine_similarity', 0.0),
                                        'keywords': keywords,
                                        'total_patents': len(firm_patent_ids.get(firm_id, [])),
                                        'sample_patents': sample_patents,
                                        'cluster': cluster_assignment
                                    }
                                    results['results'].append(company_result)
                        
                        elif mode == "model" and models_available:
                            st.info("🧠 **Model Mode**: Using neural network transformation")
                            
                            results_best = query_opportunity_product_best(
                                product_query_text=query.lower(),
                                ft_model=embedder,
                                model_B=model_B,
                                model_A=model_A,
                                patent_rep_dict=patent_rep,
                                product_rep_dict=product_rep,
                                firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
                                firm_patent_ids=firm_patent_ids,
                                patent_text_map=patent_text_map,
                                top_k=top_k
                            )
                            
                            # Format results for Streamlit (similar to matrix mode)
                            results = {
                                'query': query,
                                'method': f'ML-Model ({embedding_type.title()})',
                                'results': []
                            }
                            
                            if results_best:
                                for result in results_best:
                                    # Add keywords to result
                                    firm_id = str(result.get('firm_id', ''))
                                    company_row = product_df[product_df['Firm ID'] == firm_id]
                                    keywords = []
                                    if not company_row.empty and 'company_keywords' in company_row.columns:
                                        keywords_str = company_row.iloc[0]['company_keywords']
                                        if pd.notna(keywords_str):
                                            keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                            keywords = [k.strip() for k in keywords if k.strip()]
                                    
                                    # Get sample patents
                                    sample_patents = []
                                    if firm_id in firm_patent_ids:
                                        patent_ids = firm_patent_ids[firm_id][:3]  # Top 3 patents
                                        for patent_id in patent_ids:
                                            if patent_id in patent_text_map:
                                                sample_patents.append({
                                                    'patent_id': patent_id,
                                                    'abstract': patent_text_map[patent_id][:300] + "..." if len(patent_text_map[patent_id]) > 300 else patent_text_map[patent_id]
                                                })
                                    
                                    # Get cluster assignment
                                    cluster_assignment = "N/A"
                                    if clustering_analyzer and clustering_analyzer.cluster_assignments is not None:
                                        try:
                                            # Try exact company_id match first
                                            if firm_id in clustering_analyzer.company_ids:
                                                company_idx = clustering_analyzer.company_ids.index(firm_id)
                                                cluster_assignment = clustering_analyzer.cluster_assignments[company_idx]
                                            else:
                                                # Try converting to int if company_id is numeric string
                                                try:
                                                    company_id_int = str(int(firm_id))
                                                    if company_id_int in clustering_analyzer.company_ids:
                                                        company_idx = clustering_analyzer.company_ids.index(company_id_int)
                                                        cluster_assignment = clustering_analyzer.cluster_assignments[company_idx]
                                                except ValueError:
                                                    pass
                                        except Exception as e:
                                            logger.warning(f"Could not get cluster assignment for {firm_id}: {e}")
                                    
                                    company_result = {
                                        'company_name': result.get('firm_name', 'Unknown'),
                                        'company_id': firm_id,
                                        'relevance_score': result.get('cosine_similarity', 0.0),
                                        'keywords': keywords,
                                        'total_patents': len(firm_patent_ids.get(firm_id, [])),
                                        'sample_patents': sample_patents,
                                        'cluster': cluster_assignment
                                    }
                                    results['results'].append(company_result)
                        
                        # Add clustering info if available
                        if clustering_analyzer and config.get('enable_clustering') and query_embedding is not None:
                            cluster_id, distance = clustering_analyzer.find_nearest_cluster(query_embedding)
                            cluster_info = clustering_analyzer.get_cluster_info(cluster_id)
                            
                            results['cluster_info'] = {
                                'nearest_cluster': cluster_id,
                                'cluster_distance': distance,
                                'cluster_size': cluster_info.get('n_companies', 0),
                                'cluster_companies': []
                            }
                            
                            # Get top-k most relevant companies from cluster instead of samples
                            try:
                                if hasattr(clustering_analyzer, 'embeddings_matrix') and clustering_analyzer.embeddings_matrix is not None:
                                    # Use config value for number of companies to show
                                    top_k_companies_count = config.get('top_k_companies_in_cluster', TOP_K_COMPANIES_IN_CLUSTER)
                                    top_k_companies = clustering_analyzer.get_top_k_companies_in_cluster(
                                        cluster_id=cluster_id,
                                        query_embedding=query_embedding,
                                        k=top_k_companies_count,
                                        embeddings_dict=None  # Use stored embeddings matrix
                                    )
                                    
                                    if top_k_companies:
                                        for company in top_k_companies:
                                            company_name = company.get('company_name', 'Unknown')
                                            keywords_str = str(company.get('keywords', ''))
                                            similarity_score = company.get('similarity_score', 0.0)
                                            
                                            # Parse keywords
                                            if keywords_str and keywords_str != 'nan' and keywords_str != '':
                                                keywords = keywords_str.split('|')[:10]
                                            else:
                                                keywords = []
                                            
                                            results['cluster_info']['cluster_companies'].append({
                                                'name': company_name,
                                                'keywords': keywords,
                                                'similarity_score': similarity_score,
                                                'rank_in_cluster': company.get('rank_in_cluster', 0)
                                            })
                                    else:
                                        # Fallback to sample companies
                                        if 'companies' in cluster_info:
                                            for _, company in cluster_info['companies'].head(3).iterrows():
                                                results['cluster_info']['cluster_companies'].append({
                                                    'name': company.get('company_name', 'Unknown'),
                                                    'keywords': str(company.get('company_keywords', '')).split('|')[:3],
                                                    'similarity_score': None,
                                                    'rank_in_cluster': None
                                                })
                                else:
                                    # Fallback to sample companies if embeddings matrix not available
                                    if 'companies' in cluster_info:
                                        for _, company in cluster_info['companies'].head(3).iterrows():
                                            results['cluster_info']['cluster_companies'].append({
                                                'name': company.get('company_name', 'Unknown'),
                                                'keywords': str(company.get('company_keywords', '')).split('|')[:3],
                                                'similarity_score': None,
                                                'rank_in_cluster': None
                                            })
                            
                            except Exception as e:
                                logger.warning(f"Could not retrieve top-k companies for cluster, using samples: {e}")
                                # Fallback to sample companies
                                if 'companies' in cluster_info:
                                    for _, company in cluster_info['companies'].head(3).iterrows():
                                        results['cluster_info']['cluster_companies'].append({
                                            'name': company.get('company_name', 'Unknown'),
                                            'keywords': str(company.get('company_keywords', '')).split('|')[:3],
                                            'similarity_score': None,
                                            'rank_in_cluster': None
                                        })
                        
                        status_placeholder.empty()
                        
                    except Exception as e:
                        st.error(f"❌ ML query failed: {str(e)}")
                        st.info("💡 Make sure you have trained the ML models first using Pipeline Manager")
                        logger.error(f"ML query error: {e}")
                        return
                
                st.success("✅ Query completed successfully!")
                
                # Display results with actual data
                self.display_query_results(query, results, config)
                
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
                logger.error(f"Query error: {e}")

    def display_query_results(self, query, results, config):
        """Display query results with better styling and real data"""
        st.subheader(f"🔍 Results for: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        
        # Enhanced results summary with better styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_results = len(results.get('results', []))
            st.metric(
                label="📊 Total Results",
                value=total_results,
                help="Number of companies returned"
            )
        
        with col2:
            method = results.get('method', 'Unknown')
            st.metric(
                label="🔬 Method Used",
                value=method.split(' - ')[0] if ' - ' in method else method,
                delta=method.split(' - ')[1] if ' - ' in method else None,
                help="Query method and embedding type"
            )
        
        with col3:
            if config.get('enable_clustering') and 'cluster_info' in results:
                cluster_id = results['cluster_info']['nearest_cluster']
                st.metric(
                    label="🎯 Nearest Cluster",
                    value=cluster_id,
                    help="Cluster ID closest to your query"
                )
            else:
                st.metric(
                    label="🎯 Clustering",
                    value="Disabled",
                    help="Enable clustering in sidebar for cluster analysis"
                )
        
        with col4:
            if 'cluster_info' in results:
                cluster_size = results['cluster_info']['cluster_size']
                st.metric(
                    label="🏢 Cluster Size",
                    value=f"{cluster_size} companies",
                    help="Number of companies in the nearest cluster"
                )
            else:
                embedding_type = config.get('embedding_type', 'fasttext').title()
                st.metric(
                    label="📊 Embedding",
                    value=embedding_type,
                    help="Embedding type used for query"
                )
        
        # Show note if ML not implemented
        if 'note' in results:
            st.info(f"ℹ️ {results['note']}")
            return
        
        # Cluster information (if available) with better styling
        if config.get('enable_clustering') and 'cluster_info' in results:
            with st.container():
                st.markdown("### 🎯 Cluster Analysis")
                
                cluster_info = results['cluster_info']
                
                # Cluster info in a nice box
                st.info(f"""
                **🎯 Nearest Cluster:** {cluster_info['nearest_cluster']}  
                **📏 Distance:** {cluster_info['cluster_distance']:.4f}  
                **🏢 Companies in Cluster:** {cluster_info['cluster_size']}
                """)
                
                # Top-k most relevant companies from cluster with enhanced display
                if cluster_info.get('cluster_companies'):
                    # Check if we have similarity scores to determine display style
                    has_similarity_scores = any(company.get('similarity_score') is not None for company in cluster_info['cluster_companies'])
                    
                    if has_similarity_scores:
                        st.markdown("**🎯 Top Most Relevant Companies in this Cluster:**")
                    else:
                        st.markdown("**🏢 Sample Companies in this Cluster:**")
                    
                    # Use config values instead of hardcoded constants
                    keywords_per_company = config.get('max_keywords_display', MAX_KEYWORDS_DISPLAY)
                    # For cluster display, use fewer keywords to avoid clutter
                    cluster_keywords_limit = min(keywords_per_company // 2, 15)
                    
                    # Display cluster companies with enhanced formatting similar to main results
                    for i, company in enumerate(cluster_info['cluster_companies']):
                        company_name = company.get('name', 'Unknown Company')
                        keywords = company.get('keywords', [])
                        
                        # Determine display elements based on available data
                        if company.get('similarity_score') is not None:
                            rank = company.get('rank_in_cluster', i+1) 
                            similarity = company.get('similarity_score', 0.0)
                            
                            # Color-coded expansion based on similarity (similar to main results)
                            score_color = "🟢" if similarity > 0.8 else "🟡" if similarity > 0.6 else "🔴"
                            expander_title = f"{score_color} **#{rank}**: {company_name} (Similarity: {similarity:.3f})"
                        else:
                            score_color = "🔵"
                            expander_title = f"{score_color} **{i+1}**: {company_name}"
                        
                        # Create expandable section (expanded for first 2 companies)
                        with st.expander(expander_title, expanded=i < 2):
                            
                            # Company details in organized columns (similar to main results)
                            detail_col1, detail_col2 = st.columns([3, 2])
                            
                            with detail_col1:
                                st.markdown("**🏢 Cluster Company Information:**")
                                st.write(f"• **Company Name:** {company_name}")
                                if company.get('similarity_score') is not None:
                                    st.write(f"• **Similarity to Query:** {similarity:.4f}")
                                    st.write(f"• **Rank in Cluster:** #{rank}")
                                st.write(f"• **Position:** Company in nearest cluster")
                            
                            with detail_col2:
                                st.markdown("**🏷️ Company Keywords:**")
                                if keywords:
                                    # Display keywords as tags (similar to main results)
                                    keyword_tags = []
                                    displayed_keywords = keywords[:cluster_keywords_limit]
                                    
                                    for keyword in displayed_keywords:
                                        keyword_tags.append(f"`{keyword}`")
                                    
                                    # Show keywords in rows for better readability
                                    keywords_text = " ".join(keyword_tags)
                                    st.markdown(keywords_text)
                                    
                                    # Show count if more keywords exist
                                    if len(keywords) > cluster_keywords_limit:
                                        st.caption(f"... and {len(keywords) - cluster_keywords_limit} more keywords")
                                        
                                        # Show additional keywords in an expander
                                        with st.expander(f"Show all {len(keywords)} keywords", expanded=False):
                                            all_keyword_tags = []
                                            for keyword in keywords:
                                                all_keyword_tags.append(f"`{keyword}`")
                                            st.markdown(" ".join(all_keyword_tags))
                                else:
                                    st.caption("No keywords available")
                            
                            # Add a subtle divider between companies
                            if i < len(cluster_info['cluster_companies']) - 1:
                                st.divider()
        
        # Company results with enhanced display
        if results.get('results'):
            st.markdown("### 🏢 Company Results")
            
            for i, result in enumerate(results['results']):
                company_name = result.get('company_name', 'Unknown Company')
                relevance_score = result.get('relevance_score', 0.0)
                
                # Color-coded expansion based on relevance
                score_color = "🟢" if relevance_score > 0.8 else "🟡" if relevance_score > 0.6 else "🔴"
                
                with st.expander(
                    f"{score_color} **Rank {i+1}**: {company_name} (Score: {relevance_score:.3f})", 
                    expanded=i < 2
                ):
                    
                    # Company details in organized columns
                    detail_col1, detail_col2, detail_col3 = st.columns([2, 1, 1])
                    
                    with detail_col1:
                        st.markdown("**📋 Company Information:**")
                        st.write(f"• **Company ID:** {result.get('company_id', 'N/A')}")
                        st.write(f"• **Relevance Score:** {relevance_score:.4f}")
                        
                        # Show cluster assignment if available
                        cluster_assignment = result.get('cluster', 'N/A')
                        st.write(f"• **Cluster:** {cluster_assignment}")
                    
                    with detail_col2:
                        st.markdown("**📊 Patent Portfolio:**")
                        total_patents = result.get('total_patents', 0)
                        st.metric("Total Patents", total_patents)
                    
                    with detail_col3:
                        st.markdown("**🏷️ Keywords:**")
                        keywords = result.get('keywords', [])
                        if keywords:
                            keyword_tags = []
                            for keyword in keywords[:config.get('max_keywords_display', MAX_KEYWORDS_DISPLAY)]:
                                keyword_tags.append(f"`{keyword}`")
                            st.markdown(" ".join(keyword_tags))
                            
                            if len(keywords) > config.get('max_keywords_display', MAX_KEYWORDS_DISPLAY):
                                st.caption(f"... and {len(keywords) - config.get('max_keywords_display', MAX_KEYWORDS_DISPLAY)} more")
                        else:
                            st.caption("No keywords available")
                    
                    # Sample patents (if available)
                    sample_patents = result.get('sample_patents', [])
                    if sample_patents:
                        st.markdown("**📄 Sample Patents:**")
                        
                        for j, patent in enumerate(sample_patents[:3]):  # Show max 3 patents
                            patent_id = patent.get('patent_id', f'Patent {j+1}')
                            abstract = patent.get('abstract', 'No abstract available')
                            
                            st.markdown(f"**{j+1}. {patent_id}**")
                            
                            # Truncate long abstracts
                            if len(abstract) > 200:
                                abstract = abstract[:200] + "..."
                            
                            st.caption(abstract)
                            st.divider()
            
            # Generate and display product suggestions if enabled
            if config.get('enable_product_suggestions', False):
                self.display_product_suggestions(query, results, config)
        else:
            st.warning("🔍 No companies found matching your query. Try different keywords or check your configuration.")
            
            # Helpful suggestions
            st.markdown("**💡 Suggestions:**")
            st.markdown("• Try broader or more specific keywords")
            st.markdown("• Check if your embedding type matches trained models")
            st.markdown("• Ensure RAG database is built for your embedding type")
            st.markdown("• Try different flow types (ML vs RAG)")

    def display_product_suggestions(self, query, results, config):
        """Generate and display AI-powered product suggestions for each company"""
        st.markdown("---")
        st.markdown("### 🛍️ AI-Powered Product Suggestions")
        # st.info("🎯 **Product suggestions are generated using AI analysis of patents and company keywords**")
        
        # Get configuration
        similarity_threshold = config.get('product_similarity_threshold', PRODUCT_SUGGESTION_CONFIG['similarity_threshold'])
        max_suggestions = config.get('max_suggestions_per_company', PRODUCT_SUGGESTION_CONFIG['top_k_suggestions'])
        enable_openai = config.get('enable_openai_enhance', PRODUCT_SUGGESTION_CONFIG['enable_openai_enhance'])
        
        try:
            with st.spinner("🔄 Generating AI product suggestions..."):
                # Convert query results to format expected by PatentProductSuggester
                companies_data = []
                
                # Process main results (transformation matrix companies)
                # st.info("📊 **Processing Transformation Matrix Companies**")
                for result in results.get('results', []):
                    company_data = {
                        'hojinid': result.get('company_id', 'unknown'),
                        'name': result.get('company_name', 'Unknown Company'),
                        'keywords': result.get('keywords', []),
                        'patents': [],
                        'source': 'transformation_matrix'
                    }
                    
                    # Convert sample patents to the expected format
                    sample_patents = result.get('sample_patents', [])
                    for patent in sample_patents:
                        company_data['patents'].append({
                            'id': patent.get('patent_id', ''),
                            'patent_id': patent.get('patent_id', ''),
                            'abstract': patent.get('abstract', '')
                        })
                    
                    companies_data.append(company_data)
                
                # Process cluster companies if available
                cluster_info = results.get('cluster_info', {})
                cluster_companies = cluster_info.get('cluster_companies', [])
                
                # Debug cluster companies structure
                if cluster_companies:
                    logger.info(f"Found {len(cluster_companies)} cluster companies")
                    for i, cc in enumerate(cluster_companies[:2]):  # Log first 2 for debugging
                        logger.info(f"Cluster company {i+1}: {cc}")
                        logger.info(f"  Available keys: {list(cc.keys())}")
                        if 'keywords' in cc:
                            logger.info(f"  Keywords count: {len(cc['keywords'])}")
                else:
                    logger.info("No cluster companies found in results")
                    if 'cluster_info' in results:
                        logger.info(f"Cluster info keys: {list(cluster_info.keys())}")
                    else:
                        logger.info("No cluster_info in results")
                
                if cluster_companies and config.get('enable_clustering', False):
                    # st.info(f"🎯 **Processing Nearest Cluster Companies** ({len(cluster_companies)} found)")
                    
                    # Load patent data for cluster companies
                    try:
                        import pandas as pd
                        patent_df = None
                        firm_patent_ids = {}
                        patent_text_map = {}
                        
                        # Load patent data
                        patent_file = US_PATENT_DATA_CLEANED  # US patents
                        patent_df = pd.read_csv(patent_file)
                        
                        # Process patent IDs and text mapping
                        for firm_id, group in patent_df.groupby('hojin_id'):
                            firm_id = str(firm_id)
                            firm_patent_ids[firm_id] = group['appln_id'].tolist()
                            for app_id, abs_text in zip(group['appln_id'], group['clean_abstract'].dropna()):
                                patent_text_map[app_id] = abs_text
                    except Exception as e:
                        logger.warning(f"Could not load patent data for cluster companies: {e}")
                        firm_patent_ids = {}
                        patent_text_map = {}
                    
                    # Try to get company names and IDs from clustering analyzer
                    clustering_analyzer = st.session_state.get('clustering_analyzer')
                    
                    for cluster_company in cluster_companies:
                        company_name = cluster_company.get('name', 'Unknown Company')
                        
                        # FIXED: Get full company data from clustering analyzer instead of limited display data
                        company_id = 'unknown'
                        full_keywords = []
                        
                        # Try to get company_id from cluster_company data first
                        if 'company_id' in cluster_company:
                            company_id = str(cluster_company['company_id'])
                            logger.info(f"Using company_id from cluster data: {company_id}")
                        
                        if clustering_analyzer and hasattr(clustering_analyzer, 'company_data') and clustering_analyzer.company_data is not None:
                            try:
                                company_row = None
                                
                                # Try lookup by company_id first (more reliable)
                                if company_id != 'unknown':
                                    company_row = clustering_analyzer.company_data[
                                        clustering_analyzer.company_data['hojin_id'].astype(str) == company_id
                                    ]
                                    if not company_row.empty:
                                        logger.info(f"Found company by ID: {company_id}")
                                
                                # Fallback to lookup by name if ID lookup failed
                                if company_row is None or company_row.empty:
                                    company_row = clustering_analyzer.company_data[
                                        clustering_analyzer.company_data['company_name'] == company_name
                                    ]
                                    if not company_row.empty:
                                        company_id = str(company_row.iloc[0]['hojin_id'])
                                        logger.info(f"Found company by name: {company_name} -> ID: {company_id}")
                                
                                if not company_row.empty:
                                    # FIXED: Get FULL keywords, not truncated ones
                                    full_keywords_str = company_row.iloc[0].get('company_keywords', '')
                                    if full_keywords_str and str(full_keywords_str) != 'nan':
                                        full_keywords = str(full_keywords_str).split('|')
                                        full_keywords = [kw.strip() for kw in full_keywords if kw.strip()]
                                        logger.info(f"Found full company data for {company_name}: ID={company_id}, {len(full_keywords)} keywords")
                                    else:
                                        logger.warning(f"No keywords found for {company_name} in full company data")
                                else:
                                    logger.warning(f"Company {company_name} (ID: {company_id}) not found in clustering analyzer company_data")
                            except Exception as e:
                                logger.warning(f"Could not find company data for {company_name}: {e}")
                        
                        # Fallback to display keywords if full keywords not found
                        if not full_keywords:
                            full_keywords = cluster_company.get('keywords', [])
                            logger.info(f"Using fallback keywords for {company_name}: {len(full_keywords)} keywords")
                        
                        company_data = {
                            'hojinid': company_id,
                            'name': company_name,
                            'keywords': full_keywords,  # FIXED: Use full keywords
                            'patents': [],
                            'source': 'nearest_cluster'
                        }
                        
                        # Add patent data if company ID was found (use config values)
                        max_patents_per_company = config.get('max_patents_per_company', MAX_PATENTS_PER_COMPANY)
                        max_abstract_length = config.get('max_abstract_length', 300)
                        
                        if company_id != 'unknown' and company_id in firm_patent_ids:
                            patent_ids = firm_patent_ids[company_id][:max_patents_per_company]
                            for patent_id in patent_ids:
                                if patent_id in patent_text_map:
                                    abstract_text = patent_text_map[patent_id]
                                    company_data['patents'].append({
                                        'id': patent_id,
                                        'patent_id': patent_id,
                                        'abstract': abstract_text[:max_abstract_length] + "..." if len(abstract_text) > max_abstract_length else abstract_text
                                    })
                        
                        # FIXED: Add company data outside the patent loop
                        companies_data.append(company_data)
                        
                        # Debug logging for cluster companies
                        if company_id != 'unknown':
                            logger.info(f"Added cluster company: {company_name} (ID: {company_id}) with {len(company_data['patents'])} patents and {len(full_keywords)} keywords")
                        else:
                            logger.info(f"Added cluster company: {company_name} (ID: unknown) with {len(full_keywords)} keywords only")
                    
                    # st.info(f"✅ **Processed {len([c for c in companies_data if c.get('source') == 'nearest_cluster'])} cluster companies**")
                
                if not companies_data:
                    st.warning("⚠️ No companies available for product suggestions")
                    return
                
                # Display processing summary
                total_matrix_companies = len([c for c in companies_data if c.get('source') == 'transformation_matrix'])
                total_cluster_companies = len([c for c in companies_data if c.get('source') == 'nearest_cluster'])
                
                if total_matrix_companies > 0 and total_cluster_companies > 0:
                    st.success(f"✅ **Ready to process**: {total_matrix_companies} Matrix companies + {total_cluster_companies} Cluster companies = {len(companies_data)} total")
                elif total_matrix_companies > 0:
                    st.success(f"✅ **Ready to process**: {total_matrix_companies} Matrix companies")
                elif total_cluster_companies > 0:
                    st.success(f"✅ **Ready to process**: {total_cluster_companies} Cluster companies")
                else:
                    st.warning("⚠️ No valid companies found for processing")
                
                # Initialize PatentProductSuggester with configuration
                suggestion_config = {
                    'similarity_threshold': similarity_threshold,
                    'top_k_suggestions': max_suggestions,
                    'enable_openai_enhance': enable_openai,
                    'alpha': PRODUCT_SUGGESTION_CONFIG['alpha'],
                    'beta': PRODUCT_SUGGESTION_CONFIG['beta'],
                    'max_keywords': PRODUCT_SUGGESTION_CONFIG['max_keywords'],
                    'max_combinations': config.get('max_combinations_setting', PRODUCT_SUGGESTION_CONFIG['max_combinations']),
                    'use_patent_data': PRODUCT_SUGGESTION_CONFIG['use_patent_data'],
                    'debug_enabled': False  # Disable debug for cleaner UI
                }
                
                # Debug: Show configuration being used
                st.info(f"🔧 **Config**: max_combinations={suggestion_config['max_combinations']}, max_keywords={suggestion_config['max_keywords']}, threshold={suggestion_config['similarity_threshold']:.3f}")
                
                suggester = PatentProductSuggester(suggestion_config)
                
                # Generate suggestions
                suggestion_results = suggester.suggest_products(query, companies_data)
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_companies = len(suggestion_results.get('results', []))
                companies_with_suggestions = len([r for r in suggestion_results.get('results', []) if r.get('products', [])])
                total_products = sum(len(r.get('products', [])) for r in suggestion_results.get('results', []))
                avg_similarity = suggestion_results.get('summary', {}).get('average_similarity', 0.0)
                
                with col1:
                    st.metric("🏢 Companies Analyzed", total_companies)
                with col2:
                    st.metric("✅ Companies with Suggestions", companies_with_suggestions)
                with col3:
                    st.metric("🛍️ Total Products Suggested", total_products)
                with col4:
                    st.metric("📊 Avg. Similarity", f"{avg_similarity:.3f}")
                
                # Display individual company suggestions
                if suggestion_results.get('results'):
                    st.markdown("#### 🏢 Product Suggestions by Company")
                    
                    # Separate results by source for better organization
                    matrix_results = [r for r in suggestion_results['results'] if r.get('source') == 'transformation_matrix']
                    cluster_results = [r for r in suggestion_results['results'] if r.get('source') == 'nearest_cluster']
                    
                    # Display matrix companies first
                    if matrix_results:
                        st.markdown("##### 🔬 Transformation Matrix Companies")
                        for i, company_result in enumerate(matrix_results):
                            self._display_company_suggestion(company_result, i, similarity_threshold, enable_openai)
                    
                    # Display cluster companies second
                    if cluster_results:
                        st.markdown("##### 🎯 Nearest Cluster Companies")
                        for i, company_result in enumerate(cluster_results):
                            self._display_company_suggestion(company_result, i + len(matrix_results), similarity_threshold, enable_openai)
                    
                    # If no source separation, display all together
                    if not matrix_results and not cluster_results:
                        for i, company_result in enumerate(suggestion_results['results']):
                            self._display_company_suggestion(company_result, i, similarity_threshold, enable_openai)
                
                # Automatic export (like in main.py)
                if suggestion_results.get('results'):
                    st.markdown("---")
                    st.markdown("#### 📥 Automatic Export")
                    
                    with st.spinner("💾 Automatically exporting results..."):
                        try:
                            # Export to JSON
                            json_path = suggester.export_results(suggestion_results)
                            
                            # Export to Text  
                            text_path = suggester.export_results_as_text(suggestion_results)
                            
                            st.success("✅ **Results automatically exported!**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"📄 **JSON**: `{json_path}`")
                            with col2:
                                st.info(f"📝 **Text**: `{text_path}`")
                            
                            # Display summary like in main.py
                            total_companies = len(suggestion_results['results'])
                            companies_with_suggestions = len([r for r in suggestion_results['results'] if r['products']])
                            total_products = sum(len(r['products']) for r in suggestion_results['results'])
                            
                            st.markdown("#### 📊 Export Summary")
                            st.write(f"**Query**: '{query[:60]}{'...' if len(query) > 60 else ''}'")
                            st.write(f"**Companies processed**: {total_companies}")
                            st.write(f"**Companies with suggestions**: {companies_with_suggestions}")
                            st.write(f"**Total products suggested**: {total_products}")
                            
                        except Exception as e:
                            st.error(f"❌ Automatic export failed: {str(e)}")
                            logger.error(f"Export error: {e}")
                    
                    # Manual export option as backup
                    with st.expander("🔧 Manual Export Options", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📄 Re-export to JSON", type="secondary"):
                                try:
                                    json_path = suggester.export_results(suggestion_results)
                                    st.success(f"✅ Results re-exported to: `{json_path}`")
                                except Exception as e:
                                    st.error(f"❌ Export failed: {str(e)}")
                        
                        with col2:
                            if st.button("📝 Re-export to Text", type="secondary"):
                                try:
                                    text_path = suggester.export_results_as_text(suggestion_results)
                                    st.success(f"✅ Results re-exported to: `{text_path}`")
                                except Exception as e:
                                    st.error(f"❌ Export failed: {str(e)}")
                
                else:
                    st.warning("⚠️ No product suggestions generated. Try lowering the similarity threshold or using different keywords.")
                    
        except Exception as e:
            st.error(f"❌ Product suggestion generation failed: {str(e)}")
            st.info("💡 Make sure the product suggestion models are properly initialized")
            logger.error(f"Product suggestion error: {e}")

    def _display_company_suggestion(self, company_result, index, similarity_threshold, enable_openai):
        """Helper method to display individual company suggestion"""
        company_name = company_result.get('company_name', 'Unknown Company')
        company_similarity = company_result.get('company_similarity', 0.0)
        products = company_result.get('products', [])
        source = company_result.get('source', 'unknown')
        
        # Color coding based on similarity
        if company_similarity > 0.3:
            score_color = "🟢"
        elif company_similarity > 0.15:
            score_color = "🟡"
        else:
            score_color = "🔴"
        
        # Source emoji
        source_emoji = "🔬" if source == 'transformation_matrix' else "🎯" if source == 'nearest_cluster' else "❓"
        source_text = "Matrix" if source == 'transformation_matrix' else "Cluster" if source == 'nearest_cluster' else "Unknown"
        
        with st.expander(
            f"{score_color} **{company_name}** ({source_emoji} {source_text}) - {len(products)} products (Similarity: {company_similarity:.3f})",
            expanded=index < 3 and len(products) > 0  # Expand first 3 companies with suggestions
        ):
            if products:
                # Company similarity details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Overall Similarity", f"{company_similarity:.3f}")
                with col2:
                    keyword_sim = company_result.get('keyword_similarity', 0.0)
                    st.metric("🏷️ Keyword Similarity", f"{keyword_sim:.3f}")
                with col3:
                    patent_sim = company_result.get('patent_similarity', 0.0)
                    st.metric("📄 Patent Similarity", f"{patent_sim:.3f}")
                
                st.markdown("**🛍️ Suggested Products:**")
                
                # Display products in a structured way
                for j, product in enumerate(products, 1):
                    product_name = product.get('product_name', 'Unknown Product')
                    product_score = product.get('score', 0.0)
                    lexical_sim = product.get('lexical_similarity', 0.0)
                    semantic_sim = product.get('semantic_similarity', 0.0)
                    
                    # Product box with score
                    with st.container():
                        st.markdown(f"**{j}. {product_name}**")
                        
                        # Product metrics in smaller columns
                        p_col1, p_col2, p_col3 = st.columns(3)
                        with p_col1:
                            st.caption(f"🎯 Score: {product_score:.3f}")
                        with p_col2:
                            st.caption(f"📝 Lexical: {lexical_sim:.3f}")
                        with p_col3:
                            st.caption(f"🧠 Semantic: {semantic_sim:.3f}")
                        
                        # Show enhanced vs original if available
                        if 'original_name' in product and enable_openai:
                            st.caption(f"💡 *Enhanced by AI from: {product['original_name']}*")
                        
                        if j < len(products):
                            st.divider()
            else:
                st.info(f"💡 No product suggestions generated for {company_name} (similarity below threshold: {similarity_threshold:.3f})")
                
                # Show why no suggestions were generated
                st.caption(f"Company similarity: {company_similarity:.3f} < Threshold: {similarity_threshold:.3f}")

    def display_cluster_info(self, cluster_info):
        """Display cluster information"""
        st.subheader("🎯 Cluster Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **🎯 Nearest Cluster:** {cluster_info['nearest_cluster']}
            **📏 Distance:** {cluster_info['cluster_distance']:.4f}
            **🏢 Cluster Size:** {cluster_info['cluster_size']} companies
            """)
        
        with col2:
            # Check if we have similarity scores to determine display style
            has_similarity_scores = any(company.get('similarity_score') is not None for company in cluster_info['cluster_companies'][:3])
            
            if has_similarity_scores:
                st.write("**🎯 Top Most Relevant Companies in this Cluster:**")
                for company in cluster_info['cluster_companies'][:3]:
                    rank = company.get('rank_in_cluster', 1)
                    similarity = company.get('similarity_score', 0.0)
                    st.write(f"• **#{rank}. {company['name']}** (sim: {similarity:.3f})")
                    st.caption(f"Keywords: {', '.join(company['keywords'])}")
            else:
                st.write("**🏢 Sample Companies in this Cluster:**")
                for company in cluster_info['cluster_companies'][:3]:
                    st.write(f"• **{company['name']}**")
                    st.caption(f"Keywords: {', '.join(company['keywords'])}")

    def render_clustering_analysis(self):
        """Enhanced clustering analysis tab with real data support"""
        st.header("🎯 Clustering Analysis")
        
        # Check if clustering is available
        clustering_analyzer = st.session_state.get('clustering_analyzer')
        
        if clustering_analyzer is None:
            st.warning("⚠️ Clustering analyzer not loaded. Please initialize components or run clustering pipeline first.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Try Load Clustering", type="primary"):
                    config = st.session_state.get('config', {})
                    embedding_type = config.get('embedding_type', 'fasttext')
                    country = config.get('countries', ['US'])[0]
                    
                    clustering_analyzer = load_clustering_analyzer(embedding_type, country)
                    if clustering_analyzer:
                        st.session_state.clustering_analyzer = clustering_analyzer
                        st.success("✅ Clustering analyzer loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ No clustering results found. Please run clustering pipeline first.")
            
            with col2:
                st.info("💡 Run clustering from Pipeline Manager tab or use main.py")
            
            # Show demo clustering data
            st.subheader("📊 Demo Clustering Analysis")
            st.info("Here's what clustering analysis would look like:")
            self._show_demo_clustering_analysis()
            
        else:
            # Show real clustering analysis
            self._show_real_clustering_analysis(clustering_analyzer)

    def _show_demo_clustering_analysis(self):
        """Show demo clustering analysis"""
        demo_clustering = self.demo_data['clustering_analysis']
        
        # Metrics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔢 Best Clusters", demo_clustering['best_clusters'])
        with col2:
            st.metric("🏆 Total Rank Score", f"{demo_clustering['total_rank_score']:.1f}")
        with col3:
            st.metric("📊 Embedding Type", demo_clustering['embedding_type'].title())
        with col4:
            st.metric("🏢 Total Companies", "6,764")
        
        # Individual metrics
        st.subheader("📈 Individual Metric Performance")
        metrics_data = demo_clustering['metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔵 Silhouette Score", 
                f"{metrics_data['silhouette']['score']:.4f}",
                f"Rank: {metrics_data['silhouette']['rank']}"
            )
        
        with col2:
            st.metric(
                "🟢 Calinski-Harabasz", 
                f"{metrics_data['calinski']['score']:.2f}",
                f"Rank: {metrics_data['calinski']['rank']}"
            )
        
        with col3:
            st.metric(
                "🟡 Davies-Bouldin", 
                f"{metrics_data['davies']['score']:.4f}",
                f"Rank: {metrics_data['davies']['rank']}"
            )
        
        # Cluster distribution
        st.subheader("📋 Cluster Size Distribution")
        
        cluster_df = pd.DataFrame(demo_clustering['cluster_distribution'])
        
        # Create pie chart
        fig = px.pie(
            cluster_df, 
            values='companies', 
            names='cluster_id',
            title="Company Distribution Across Top 5 Clusters",
            labels={'cluster_id': 'Cluster ID', 'companies': 'Number of Companies'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def _show_real_clustering_analysis(self, clustering_analyzer):
        """Show real clustering analysis with loaded data"""
        st.success("✅ Clustering analyzer loaded successfully!")
        
        # Real clustering metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            best_clusters = clustering_analyzer.best_n_clusters if clustering_analyzer.best_n_clusters else "N/A"
            st.metric("🔢 Best Clusters", best_clusters)
        with col2:
            if clustering_analyzer.best_score is not None:
                st.metric("🏆 Best Score", f"{clustering_analyzer.best_score:.4f}")
            else:
                st.metric("🏆 Best Score", "N/A")
        with col3:
            if clustering_analyzer.company_ids:
                st.metric("🏢 Total Companies", len(clustering_analyzer.company_ids))
            else:
                st.metric("🏢 Total Companies", "N/A")
        with col4:
            if clustering_analyzer.embeddings_matrix is not None:
                st.metric("📏 Embedding Dim", clustering_analyzer.embeddings_matrix.shape[1])
            else:
                st.metric("📏 Embedding Dim", "N/A")
        
        # Debug information
        with st.expander("🔧 Debug Information", expanded=False):
            st.write("**Clustering Analyzer Attributes:**")
            st.write(f"- best_n_clusters: {clustering_analyzer.best_n_clusters}")
            st.write(f"- best_score: {clustering_analyzer.best_score}")
            st.write(f"- best_model: {clustering_analyzer.best_model is not None}")
            st.write(f"- cluster_assignments: {clustering_analyzer.cluster_assignments is not None}")
            st.write(f"- company_ids: {len(clustering_analyzer.company_ids) if clustering_analyzer.company_ids else 0}")
            
            # Enhanced file paths display with existence check
            if hasattr(clustering_analyzer, 'file_paths') and clustering_analyzer.file_paths:
                # st.write("**📁 File Status:**")
                for file_type, file_path in clustering_analyzer.file_paths.items():
                    exists = "-" if os.path.exists(file_path) else "❌"
                    # Show relative path for cleaner display
                    rel_path = os.path.relpath(file_path, start=os.getcwd()).replace('\\', '/')
                    st.write(f"  {exists} {file_type}: `{rel_path}`")
            else:
                st.write("- file_paths: N/A")
        
        # Load and display real ranking results
        ranking_file = clustering_analyzer.file_paths.get('ranking_results')
        if ranking_file and os.path.exists(ranking_file):
            try:
                with open(ranking_file, 'r') as f:
                    ranking_results = json.load(f)
                
                best_config = ranking_results['best_configuration']
                
                # Individual metrics with real data
                st.subheader("📈 Multi-Metric Performance (Real Data)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🔵 Silhouette Score", 
                        f"{best_config['silhouette_score']:.4f}",
                        f"Rank: {int(best_config['silhouette_rank'])}"
                    )
                
                with col2:
                    st.metric(
                        "🟢 Calinski-Harabasz", 
                        f"{best_config['calinski_score']:.2f}",
                        f"Rank: {int(best_config['calinski_rank'])}"
                    )
                
                with col3:
                    st.metric(
                        "🟡 Davies-Bouldin", 
                        f"{best_config['davies_score']:.4f}",
                        f"Rank: {int(best_config['davies_rank'])}"
                    )
                
                # Ranking details
                st.subheader("🏆 Ranking Summary")
                summary = ranking_results['summary']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **📊 Configurations Tested:** {summary['total_configurations_tested']}
                    **🏅 Best Total Rank Score:** {summary['best_total_rank_score']:.0f}
                    **🎯 Tied Configurations:** {summary['num_tied_configurations']}
                    """)
                
                with col2:
                    st.info(f"""
                    **🔬 Ranking Method:** Multi-metric total rank
                    **⚖️ Tie-breaking Rule:** {summary['tie_breaking_rule'].replace('_', ' ').title()}
                    **📏 Metrics Used:** Silhouette, Calinski-Harabasz, Davies-Bouldin
                    """)
                
            except Exception as e:
                logger.warning(f"Could not load ranking results: {e}")
        
        # Load and display clustering visualizations
        viz_path = clustering_analyzer.file_paths.get('performance_plot')
        if viz_path and os.path.exists(viz_path):
            st.subheader("📊 Performance Plots")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(viz_path, caption="Multi-Metric Performance Analysis", use_container_width=True)
            
            with col2:
                ranking_table_path = viz_path.replace('.png', '_ranking_table.png')
                if os.path.exists(ranking_table_path):
                    st.image(ranking_table_path, caption="Top Configurations Ranking Table", use_container_width=True)
        
        # Cluster distribution analysis
        if clustering_analyzer.cluster_assignments is not None and len(clustering_analyzer.cluster_assignments) > 0:
            st.subheader("📋 Cluster Distribution Analysis")
            
            unique_clusters, cluster_counts = np.unique(clustering_analyzer.cluster_assignments, return_counts=True)
            
            # Create distribution dataframe
            distribution_data = []
            total_companies = len(clustering_analyzer.cluster_assignments)
            
            for cluster_id, count in zip(unique_clusters, cluster_counts):
                if cluster_id != -1:  # Exclude noise points
                    percentage = (count / total_companies) * 100
                    distribution_data.append({
                        'cluster_id': cluster_id,
                        'companies': count,
                        'percentage': percentage
                    })
            
            distribution_df = pd.DataFrame(distribution_data)
            distribution_df = distribution_df.sort_values('companies', ascending=False)
            
            # Display top clusters
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of top clusters
                top_clusters = distribution_df.head(10)
                fig = px.bar(
                    top_clusters,
                    x='cluster_id',
                    y='companies',
                    title="Top 10 Clusters by Size",
                    labels={'cluster_id': 'Cluster ID', 'companies': 'Number of Companies'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart of cluster distribution
                top_5_clusters = distribution_df.head(5)
                others_count = distribution_df.iloc[5:]['companies'].sum()
                
                if others_count > 0:
                    pie_data = pd.concat([
                        top_5_clusters,
                        pd.DataFrame([{'cluster_id': 'Others', 'companies': others_count, 'percentage': (others_count/total_companies)*100}])
                    ])
                else:
                    pie_data = top_5_clusters
                
                fig = px.pie(
                    pie_data,
                    values='companies',
                    names='cluster_id',
                    title="Cluster Size Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("📋 Cluster Distribution Analysis")
            st.warning("⚠️ No cluster assignments available. Please run the clustering pipeline to generate cluster analysis.")
            st.info("💡 Go to the Pipeline Manager tab and run the clustering pipeline with your preferred embedding type.")
        
        # Interactive Cluster Viewer
        if clustering_analyzer.cluster_assignments is not None and len(clustering_analyzer.cluster_assignments) > 0:
            st.subheader("🔍 Interactive Cluster Explorer")
            st.markdown("**Select a cluster to view all companies and their keywords**")
            
            # Get available clusters
            unique_clusters = np.unique(clustering_analyzer.cluster_assignments)
            available_clusters = [c for c in unique_clusters if c != -1]  # Exclude noise points (-1)
            available_clusters = sorted(available_clusters)
            
            if available_clusters:
                # Cluster selection
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    selected_cluster = st.selectbox(
                        "🎯 Select Cluster to Explore:",
                        available_clusters,
                        help="Choose a cluster to view all companies and their keywords"
                    )
                
                with col2:
                    # Show cluster size
                    cluster_size = np.sum(clustering_analyzer.cluster_assignments == selected_cluster)
                    st.metric("🏢 Companies in Cluster", cluster_size)
                
                # Get cluster information
                try:
                    cluster_info = clustering_analyzer.get_cluster_info(selected_cluster)
                    
                    if cluster_info and 'companies' in cluster_info:
                        companies_df = cluster_info['companies']
                        
                        st.markdown(f"### 📊 All Companies in Cluster {selected_cluster}")
                        
                        # Pagination for large clusters
                        companies_per_page = 10
                        total_companies = len(companies_df)
                        total_pages = (total_companies + companies_per_page - 1) // companies_per_page
                        
                        if total_pages > 1:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                current_page = st.selectbox(
                                    f"📄 Page (Total: {total_pages})",
                                    range(1, total_pages + 1),
                                    help=f"Navigate through {total_companies} companies"
                                )
                        else:
                            current_page = 1
                        
                        # Calculate start and end indices
                        start_idx = (current_page - 1) * companies_per_page
                        end_idx = min(start_idx + companies_per_page, total_companies)
                        
                        # Display companies for current page
                        page_companies = companies_df.iloc[start_idx:end_idx]
                        
                        for idx, (_, company) in enumerate(page_companies.iterrows()):
                            company_number = start_idx + idx + 1
                            
                            with st.expander(f"🏢 {company_number}. {company.get('company_name', 'Unknown Company')}", expanded=False):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.markdown("**📋 Company Details:**")
                                    st.write(f"**Company ID:** {company.get('hojin_id', 'N/A')}")
                                    st.write(f"**Company Name:** {company.get('company_name', 'N/A')}")
                                
                                with col2:
                                    st.markdown("**🏷️ Keywords:**")
                                    keywords_str = company.get('company_keywords', '')
                                    if pd.notna(keywords_str) and keywords_str:
                                        keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                        keywords = [k.strip() for k in keywords if k.strip()]
                                        
                                        if keywords:
                                            # Display keywords as tags
                                            config = st.session_state.get('config', {})
                                            max_display = config.get('max_keywords_display', 20)  # Use fewer for cluster view
                                            displayed_keywords = keywords[:max_display]
                                            
                                            keyword_tags = []
                                            for keyword in displayed_keywords:
                                                keyword_tags.append(f"`{keyword}`")
                                            st.markdown(" ".join(keyword_tags))
                                            
                                            if len(keywords) > max_display:
                                                st.caption(f"... and {len(keywords) - max_display} more keywords")
                                        else:
                                            st.caption("No keywords available")
                                    else:
                                        st.caption("No keywords available")
                        
                        # Summary at bottom
                        if total_pages > 1:
                            st.info(f"📄 Showing companies {start_idx + 1}-{end_idx} of {total_companies} in Cluster {selected_cluster}")
                    
                    else:
                        st.warning(f"⚠️ Could not load detailed information for Cluster {selected_cluster}")
                        
                except Exception as e:
                    st.error(f"❌ Error loading cluster information: {str(e)}")
                    logger.error(f"Cluster viewer error: {e}")
            
            else:
                st.warning("⚠️ No valid clusters found for exploration.")

    def render_pipeline_manager(self):
        """New pipeline manager tab for running clustering and other pipelines"""
        st.header("⚙️ Pipeline Manager")
        st.markdown("**Run FullFlow pipelines directly from the web interface**")
        
        # Get current config - Fixed indentation issues
        config = st.session_state.get('config', {})
        
        # Pipeline selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pipeline_type = st.selectbox(
                "Select Pipeline",
                ["clustering", "dual_attn", "patent_product", "rag_only", "product_suggestion"],
                help="Choose which pipeline to run"
            )
        
        with col2:
            if pipeline_type == "patent_product":
                mode = st.selectbox("Mode", ["train", "test", "chat"], index=0)
            else:
                mode = None
        
        # Pipeline-specific options
        st.subheader("Pipeline Configuration")
        
        if pipeline_type == "clustering":
            st.info("🎯 **Clustering Pipeline**: Analyze company embeddings to find optimal cluster numbers using multi-metric ranking")
            
            col1, col2 = st.columns(2)
            with col1:
                enable_clustering = st.checkbox("Enable Clustering", value=True)
                force_rebuild = st.checkbox("Force Rebuild", value=config.get('force_rebuild_clustering', False))
            
            with col2:
                embedding_type = st.selectbox("Embedding Type", ["fasttext", "sentence_transformer"], 
                                            index=0 if config.get('embedding_type') == 'fasttext' else 1)
                if embedding_type == "sentence_transformer":
                    model_name = st.selectbox("Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
                else:
                    model_name = "all-MiniLM-L6-v2"
        
        elif pipeline_type == "dual_attn":
            st.info("🧠 **Dual Attention Pipeline**: Train the dual attention model for keyword extraction")
            enable_clustering = False
            force_rebuild = False
        
        elif pipeline_type == "patent_product":
            st.info(f"🔬 **Patent-Product Pipeline ({mode})**: {'Train models' if mode == 'train' else 'Run queries'}")
            
            col1, col2 = st.columns(2)
            with col1:
                use_rag = st.checkbox("Use RAG", value=config.get('flow_type') == 'rag')
                enable_clustering = st.checkbox("Enable Clustering", value=config.get('enable_clustering', True))
            
            with col2:
                embedding_type = st.selectbox("Embedding Type", ["fasttext", "sentence_transformer"], 
                                            index=0 if config.get('embedding_type') == 'fasttext' else 1)
                if embedding_type == "sentence_transformer":
                    model_name = st.selectbox("Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
                else:
                    model_name = "all-MiniLM-L6-v2"
            
            force_rebuild = False
        
        elif pipeline_type == "rag_only":
            st.info("🔍 **RAG-Only Pipeline**: Direct RAG queries without full pipeline")
            
            query = st.text_area("Query", placeholder="Enter your query here...", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                embedding_type = st.selectbox("Embedding Type", ["fasttext", "sentence_transformer"], 
                                            index=0 if config.get('embedding_type') == 'fasttext' else 1)
                if embedding_type == "sentence_transformer":
                    model_name = st.selectbox("Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
                else:
                    model_name = "all-MiniLM-L6-v2"
            
            with col2:
                enable_clustering = st.checkbox("Enable Clustering", value=True)
                rag_top_k = st.slider("Top K Results", 1, 20, 5)
            
            force_rebuild = False
        
        elif pipeline_type == "product_suggestion":
            st.info("🛍️ **Product Suggestion Pipeline**: Generate AI-powered product suggestions based on patent queries")
            
            query = st.text_area("Patent Query", placeholder="Enter patent abstract or product query here...", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold = st.slider("Similarity Threshold", 0.01, 0.5, 
                                               PRODUCT_SUGGESTION_CONFIG['similarity_threshold'], 0.01)
                max_suggestions = st.slider("Max Suggestions per Company", 1, 10, 
                                          PRODUCT_SUGGESTION_CONFIG['top_k_suggestions'])
            
            with col2:
                enable_openai = st.checkbox("Enable OpenAI Enhancement", 
                                          value=PRODUCT_SUGGESTION_CONFIG['enable_openai_enhance'])
                use_demo_data = st.checkbox("Use Demo Data", value=True, 
                                          help="Use built-in demo companies or real data")
            
            enable_clustering = False
            force_rebuild = False
        
        # Run pipeline button
        st.subheader("Execute Pipeline")
        
        if st.button(f"🚀 Run {pipeline_type.title()} Pipeline", type="primary", use_container_width=True):
            self.run_pipeline(pipeline_type, mode, locals())

    def run_pipeline(self, pipeline_type, mode, params):
        """Execute a pipeline with given parameters"""
        
        # Build command
        cmd = ["python", "main.py", "--pipeline", pipeline_type]
        
        # Add mode if applicable
        if mode:
            cmd.extend(["--mode", mode])
        
        # Add embedding configuration
        if 'embedding_type' in params:
            cmd.extend(["--embedding_type", params['embedding_type']])
            
            if params['embedding_type'] == "sentence_transformer" and 'model_name' in params:
                cmd.extend(["--sentence_transformer_model", params['model_name']])
        
        # Add clustering options
        if params.get('enable_clustering'):
            cmd.append("--enable_clustering")
        
        if params.get('force_rebuild_clustering'):
            cmd.append("--force_rebuild_clustering")
        
        # Add RAG options
        if params.get('use_rag'):
            cmd.append("--use_rag")
        
        if 'rag_top_k' in params:
            cmd.extend(["--rag_top_k", str(params['rag_top_k'])])
        
        # Add query for RAG-only
        if pipeline_type == "rag_only" and 'query' in params and params['query'].strip():
            cmd.extend(["--query", params['query'].strip()])
        
        # Add product suggestion options
        if pipeline_type == "product_suggestion":
            if 'query' in params and params['query'].strip():
                cmd.extend(["--query", params['query'].strip()])
            
            if 'similarity_threshold' in params:
                cmd.extend(["--product_similarity_threshold", str(params['similarity_threshold'])])
            
            if 'max_suggestions' in params:
                cmd.extend(["--product_suggestions_only"])  # Use suggestions only mode
            
            if params.get('enable_openai'):
                cmd.append("--enable_openai_enhance")
        
        # Display command
        st.subheader("🔧 Command Being Executed")
        st.code(" ".join(cmd))
        
        # Execute pipeline
        with st.spinner(f"🔄 Running {pipeline_type} pipeline..."):
            try:
                # Run the command
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                # Display results
                if result.returncode == 0:
                    st.success(f"✅ {pipeline_type.title()} pipeline completed successfully!")
                    
                    # Show output
                    if result.stdout:
                        st.subheader("📄 Pipeline Output")
                        st.text(result.stdout)
                    
                    # Store results
                    st.session_state.pipeline_results[pipeline_type] = {
                        'command': " ".join(cmd),
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'timestamp': datetime.now().isoformat(),
                        'success': True
                    }
                    
                    # If clustering pipeline completed, try to reload clustering analyzer
                    if pipeline_type == "clustering":
                        st.info("🔄 Reloading clustering analyzer...")
                        config = st.session_state.get('config', {})
                        embedding_type = params.get('embedding_type', config.get('embedding_type', 'fasttext'))
                        country = config.get('countries', ['US'])[0]
                        
                        clustering_analyzer = load_clustering_analyzer(embedding_type, country)
                        if clustering_analyzer:
                            st.session_state.clustering_analyzer = clustering_analyzer
                            st.success("✅ Clustering analyzer reloaded successfully!")
                        
                else:
                    st.error(f"❌ {pipeline_type.title()} pipeline failed!")
                    
                    if result.stderr:
                        st.subheader("❌ Error Output")
                        st.text(result.stderr)
                    
                    if result.stdout:
                        st.subheader("📄 Standard Output")
                        st.text(result.stdout)
                    
                    # Store error results
                    st.session_state.pipeline_results[pipeline_type] = {
                        'command': " ".join(cmd),
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'timestamp': datetime.now().isoformat(),
                        'success': False,
                        'return_code': result.returncode
                    }
                
            except subprocess.TimeoutExpired:
                st.error("⏰ Pipeline execution timed out (5 minutes)")
            except Exception as e:
                st.error(f"❌ Error running pipeline: {str(e)}")

    def render_demo_examples(self):
        """Render demo examples tab"""
        st.header("📊 Demo Examples")
        st.markdown("**Explore sample outputs without running the full pipeline**")
        
        # Example selection
        example_type = st.selectbox(
            "Select Demo Type",
            ["Query Results", "Clustering Analysis", "Performance Comparison (not implemented)", "Pipeline Results"]
        )
        
        if example_type == "Query Results":
            self.show_demo_query_results()
        elif example_type == "Clustering Analysis":
            self.show_demo_clustering()
        elif example_type == "Performance Comparison (not implemented)":
            self.show_demo_performance()
        elif example_type == "Pipeline Results":
            self.show_demo_pipeline_results()

    def show_demo_query_results(self):
        """Show demo query results"""
        st.subheader("🔍 Sample Query Results")
        
        # Sample queries
        sample_queries = [
            "machine learning algorithms for medical diagnosis",
            "renewable energy storage systems",
            "autonomous vehicle navigation systems",
            "quantum computing applications"
        ]
        
        selected_query = st.selectbox("Select a sample query:", sample_queries)
        
        if st.button("👀 View Results", type="primary"):
            # Show demo results
            demo_result = self.demo_data['sample_queries'][0]
            demo_result['query'] = selected_query  # Update query
            
            self.display_query_results(selected_query, demo_result, {'enable_clustering': True})

    def show_demo_clustering(self):
        """Show demo clustering analysis"""
        st.subheader("🎯 Sample Clustering Analysis")
        
        demo_clustering = self.demo_data['clustering_analysis']
        
        # Performance metrics comparison
        st.subheader("📈 Multi-Metric Performance")
        
        metrics_df = pd.DataFrame([
            {"Metric": "Silhouette Score", "Score": 0.2341, "Rank": 1, "Higher_Better": True},
            {"Metric": "Calinski-Harabasz", "Score": 450.23, "Rank": 2, "Higher_Better": True},
            {"Metric": "Davies-Bouldin", "Score": 1.234, "Rank": 1, "Higher_Better": False}
        ])
        
        fig = px.bar(
            metrics_df, 
            x="Metric", 
            y="Score",
            color="Rank",
            title="Clustering Performance Metrics",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster size distribution
        st.subheader("📋 Cluster Distribution")
        cluster_df = pd.DataFrame(demo_clustering['cluster_distribution'])
        
        fig2 = px.bar(
            cluster_df,
            x="cluster_id",
            y="companies",
            title="Companies per Cluster",
            labels={"cluster_id": "Cluster ID", "companies": "Number of Companies"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    def show_demo_performance(self):
        """Show demo performance comparison"""
        st.subheader("⚡ Performance Comparison (not implemented)")
        
        # Create sample performance data
        comparison_data = {
            "Method": ["ML-Matrix (FastText)", "ML-Model (FastText)", "RAG (FastText)", 
                      "ML-Matrix (Sentence)", "ML-Model (Sentence)", "RAG (Sentence)"],
            "Accuracy": [0.85, 0.88, 0.82, 0.89, 0.92, 0.87],
            "Speed (ms)": [120, 450, 200, 180, 600, 300],
            "Memory (MB)": [50, 200, 150, 80, 300, 250]
        }
        
        perf_df = pd.DataFrame(comparison_data)
        
        # Accuracy comparison
        fig1 = px.bar(
            perf_df,
            x="Method",
            y="Accuracy",
            title="Accuracy Comparison Across Methods",
            color="Accuracy",
            color_continuous_scale="Viridis"
        )
        fig1.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Speed vs Accuracy scatter
        fig2 = px.scatter(
            perf_df,
            x="Speed (ms)",
            y="Accuracy",
            size="Memory (MB)",
            color="Method",
            title="Speed vs Accuracy Trade-off",
            hover_data=["Memory (MB)"]
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary table
        st.subheader("📊 Performance Summary")
        st.dataframe(
            perf_df,
            column_config={
                "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1),
                "Speed (ms)": st.column_config.NumberColumn("Speed (ms)", format="%d ms"),
                "Memory (MB)": st.column_config.NumberColumn("Memory (MB)", format="%d MB")
            },
            use_container_width=True
        )

    def show_demo_pipeline_results(self):
        """Show demo pipeline execution results"""
        st.subheader("⚙️ Sample Pipeline Results")
        
        # Sample pipeline results
        demo_pipeline_results = {
            "clustering": {
                "command": "python main.py --pipeline clustering --embedding_type fasttext --enable_clustering",
                "success": True,
                "output": """
🎯 CLUSTERING ANALYSIS SUMMARY (US - fasttext)
================================================================================
📊 **Multi-Metric Ranking Results:**
   🔢 Best Number of Clusters: 23
   🏆 Total Rank Score: 4 (lower is better)
   🏢 Total Companies: 6,764
   📏 Embedding Dimension: 300

📈 **Individual Metric Performance:**
   🔵 Silhouette Score: 0.2341 (Rank: 1)
   🟢 Calinski-Harabasz: 450.23 (Rank: 2)
   🟡 Davies-Bouldin: 1.234 (Rank: 1)

✅ Multi-metric clustering analysis completed successfully!
🎯 Best configuration selected using combined ranking of 3 metrics!
                """,
                "timestamp": "2024-01-15T10:30:00"
            },
            "dual_attn": {
                "command": "python main.py --pipeline dual_attn",
                "success": True,
                "output": """
[Pipeline] Start Dual Attention training...
[Data] Loading web data for training...
[Model] Building dual attention model...
[Training] Epoch 1/50 - Loss: 0.234
[Training] Epoch 10/50 - Loss: 0.123
[Training] Epoch 20/50 - Loss: 0.089
[Training] Training completed successfully!
[Save] Model saved to models/trained_models/
                """,
                "timestamp": "2024-01-15T09:15:00"
            }
        }
        
        # Pipeline selection
        selected_pipeline = st.selectbox("Select Pipeline Result", list(demo_pipeline_results.keys()))
        
        result = demo_pipeline_results[selected_pipeline]
        
        # Display result
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔧 Command")
            st.code(result['command'])
        
        with col2:
            status = "✅ Success" if result['success'] else "❌ Failed"
            st.metric("Status", status)
            st.write(f"**Timestamp:** {result['timestamp']}")
        
        st.subheader("📄 Output")
        st.text(result['output'])

    def render_system_status(self):
        """Enhanced system status with clustering info"""
        st.header("🔧 System Status")
        
        # Component status
        st.subheader("🔄 Component Status")
        
        status = st.session_state.get('initialization_status', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core Components:**")
            st.write("🤖 Embedder:", "✅ Loaded" if status.get('embedder') else "❌ Not loaded")
            st.write("🔍 RAG Processor:", "✅ Loaded" if status.get('rag_processor') else "❌ Not loaded")
            st.write("🧠 ML Models:", "✅ Loaded" if status.get('ml_models') else "❌ Not loaded")
        
        with col2:
            st.write("**Enhanced Features:**")
            clustering_status = "✅ Loaded" if status.get('clustering_analyzer') else "❌ Not loaded"
            if st.session_state.get('clustering_analyzer'):
                clustering_status += f" ({st.session_state.clustering_analyzer.best_n_clusters} clusters)"
            st.write("🎯 Clustering:", clustering_status)
            st.write("⚙️ Config Saved:", "✅ Saved" if status.get('config_saved') else "❌ Not saved")
        
        # Pipeline execution history
        st.subheader("📋 Pipeline Execution History")
        
        pipeline_results = st.session_state.get('pipeline_results', {})
        
        if pipeline_results:
            for pipeline_name, result in pipeline_results.items():
                with st.expander(f"{pipeline_name.title()} Pipeline - {'✅' if result['success'] else '❌'}"):
                    st.write(f"**Command:** `{result['command']}`")
                    st.write(f"**Timestamp:** {result['timestamp']}")
                    st.write(f"**Success:** {result['success']}")
                    
                    if result.get('stdout'):
                        st.text_area("Standard Output", result['stdout'], height=100)
                    
                    if result.get('stderr'):
                        st.text_area("Error Output", result['stderr'], height=100)
        else:
            st.info("No pipelines have been executed yet.")
        
        # Configuration display
        st.subheader("⚙️ Current Configuration")
        config = st.session_state.get('config', {})
        if config:
            st.json(config)
        else:
            st.info("No configuration set. Please use the sidebar to configure the system.")
        
        # File system info
        st.subheader("📁 File System Status")
        
        # Check for important directories and files
        important_paths = {
            "Data Directory": "data/CN_JP_US_data",  # FIXED: Check actual data directory
            "Models Directory": "models", 
            "Clustering Results": "data/clustering/results",
            "Clustering Models": "data/clustering/models",
            "Output Images": "data/outputs/img",
            "Vector Database": "data/vector_db",
            "Config File": "streamlit_config.json"
        }
        
        # Also check critical data files
        critical_files = {
            "US Web Data": "data/CN_JP_US_data/us_web_with_company.csv",
            "US Patent Data": "data/CN_JP_US_data/us_patent202506.csv", 
            "Cleaned Patents": "data/CN_JP_US_data/cleaned_patents.csv",
            "FastText Vectors": "data/CN_JP_US_data/fasttext_web_patent.vec"
        }
        
        for name, path in important_paths.items():
            exists = os.path.exists(path)
            status_icon = "✅" if exists else "❌"
            size_info = ""
            
            if exists:
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    size_info = f" ({size:,} bytes)"
                elif os.path.isdir(path):
                    try:
                        file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                        size_info = f" ({file_count} files)"
                    except:
                        size_info = " (access denied)"
            
            st.write(f"{status_icon} **{name}:** {path}{size_info}")
        
        # Check critical data files
        st.write("---")
        st.write("**📄 Critical Data Files:**")
        
        for name, path in critical_files.items():
            exists = os.path.exists(path)
            status_icon = "✅" if exists else "❌"
            size_info = ""
            
            if exists and os.path.isfile(path):
                size = os.path.getsize(path)
                if size > 1024**3:  # > 1GB
                    size_info = f" ({size / 1024**3:.1f} GB)"
                elif size > 1024**2:  # > 1MB
                    size_info = f" ({size / 1024**2:.1f} MB)"
                elif size > 1024:  # > 1KB
                    size_info = f" ({size / 1024:.1f} KB)"
                else:
                    size_info = f" ({size} bytes)"
            
            st.write(f"{status_icon} **{name}:** `{path}`{size_info}")

    def render_documentation(self):
        """Enhanced documentation tab"""
        st.header("📚 Documentation")
        
        doc_section = st.selectbox(
            "Select Documentation Section",
            ["Quick Start", "Pipeline Guide", "Product Suggestions Guide", "Clustering Guide", "Export & File Management", "API Reference", "Configuration Guide", "Troubleshooting"]
        )
        
        if doc_section == "Quick Start":
            self.show_quick_start()
        elif doc_section == "Pipeline Guide":
            self.show_pipeline_guide()
        elif doc_section == "Product Suggestions Guide":
            self.show_product_suggestions_guide()
        elif doc_section == "Clustering Guide":
            self.show_clustering_guide()
        elif doc_section == "Export & File Management":
            self.show_export_file_guide()
        elif doc_section == "API Reference":
            self.show_api_reference()
        elif doc_section == "Configuration Guide":
            self.show_config_guide()
        elif doc_section == "Troubleshooting":
            self.show_troubleshooting()

    def show_quick_start(self):
        """Show quick start documentation"""
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        ### 1. Initialize Components
        1. Configure settings in the sidebar
        2. Click "🔄 Initialize Components"
        3. Wait for all components to load
        
        ### 2. Run Training Pipelines
        1. Go to "⚙️ Pipeline Manager" tab
        2. Run **dual_attn** pipeline first (required for all flows)
        3. Run **clustering** pipeline to build clustering analysis
        4. Run **patent_product** pipeline with mode=train to build models/databases
        
        ### 3. Configure Features (Optional)
        1. **🛍️ Product Suggestions**: Enable AI-powered product name generation
        2. **🎯 Clustering**: Enable company clustering analysis  
        3. **🤖 OpenAI**: Enable AI enhancement for product suggestions
        
        ### 4. Run Queries
        1. Go to "🔍 Query Interface" tab
        2. Enter your query or use quick examples
        3. Click "🔍 Run Query" to get results with clustering info
        4. **New**: Product suggestions automatically appear at bottom (if enabled)
        5. **New**: Results auto-exported to JSON and text files
        
        ### 5. Explore Advanced Features
        1. **🎯 Clustering Analysis**: View cluster distributions and performance
        2. **📊 Product Suggestions**: AI-generated product names with similarity scores
        3. **🔧 System Status**: Monitor files, components, and debug information
        4. **📁 Export Management**: Access exported suggestion files
        """)

    def show_pipeline_guide(self):
        """Show pipeline documentation"""
        st.markdown("""
        ## ⚙️ Pipeline Guide
        
        ### Available Pipelines
        
        **1. dual_attn**
        - Trains the dual attention model for keyword extraction
        - Required first step for all other pipelines
        - No additional parameters needed
        
        **2. clustering**
        - Analyzes company embeddings to find optimal cluster numbers
        - Uses multi-metric ranking (Silhouette, Calinski-Harabasz, Davies-Bouldin)
        - Generates performance plots and ranking tables
        - Options: `--enable_clustering`, `--force_rebuild_clustering`
        
        **3. patent_product**
        - Main pipeline with three modes:
          - `train`: Build models/databases
          - `test`: Test with predefined queries
          - `chat`: Interactive querying
        - Options: `--use_rag`, `--enable_clustering`, `--embedding_type`
        
        **4. rag_only**
        - Direct RAG queries without full pipeline
        - Requires: `--query "your query here"`
        - Options: `--embedding_type`, `--rag_top_k`
        
        **5. product_suggestion** ⭐ NEW
        - AI-powered product suggestions for companies
        - Generates product names using patent analysis and keyword combinations
        - Automatic export to JSON and text files
        - Options: `--query`, `--product_similarity_threshold`, `--enable_openai_enhance`
        
        ### Execution Order
        1. **dual_attn** (always first)
        2. **clustering** (for clustering analysis)
        3. **patent_product --mode train** (build models)
        4. **patent_product --mode test/chat** (run queries)
        5. **product_suggestion** (generate AI product suggestions)
        """)

    def show_clustering_guide(self):
        """Show clustering documentation"""
        st.markdown("""
        ## 🎯 Clustering Guide
        
        ### Multi-Metric Ranking System
        
        **Metrics Used:**
        - **Silhouette Score**: Measures cluster separation (higher is better)
        - **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance (higher is better)
        - **Davies-Bouldin Index**: Average similarity between clusters (lower is better)
        
        **Ranking Process:**
        1. Each metric ranks all cluster configurations (1 = best rank)
        2. Total rank score = sum of individual ranks
        3. Best configuration = lowest total rank score
        4. Tie-breaking = largest number of clusters
        
        ### Configuration Options
        
        **Clustering Algorithms:**
        - **K-means**: Default, good for spherical clusters
        - **Hierarchical**: Good for nested structures
        - **DBSCAN**: Handles noise and outliers
        
        **Key Parameters:**
        - `CLUSTER_NUMBERS_RANGE`: Range of cluster numbers to test
        - `CLUSTERING_METRICS`: Metrics for evaluation
        - `CLUSTERING_RANDOM_STATE`: For reproducible results
        
        ### Output Files
        - Performance plots: `data/output/img/clustering_performance_*.png`
        - Ranking tables: `*_ranking_table.png`
        - Best model: `data/clustering/models/best_model_*.pkl`
        - Results: `data/clustering/results/`
        """)

    def show_api_reference(self):
        """Show API reference"""
        st.markdown("""
        ## 🔧 API Reference
        
        ### Query Methods
        
        **ML Approach:**
        - `query_opportunity_product_matrix_only()` - Matrix-based querying
        - `query_opportunity_product_best()` - Neural network-based querying
        
        **RAG Approach:**
        - `rag_processor.rag_query_companies()` - Semantic search with ChromaDB
        
        **Clustering Analysis:**
        - `load_clustering_analyzer()` - Load pre-computed clustering
        - `find_nearest_cluster()` - Find cluster for query embedding
        - `get_cluster_info()` - Get cluster details
        
        **Product Suggestion Pipeline:** ⭐ NEW
        - `PatentProductSuggester()` - Main suggestion class
        - `suggest_products()` - Generate product suggestions for companies
        - `suggest_products_for_company()` - Analyze single company
        - `generate_product_names()` - Create product names from keywords
        - `export_results()` - Export to JSON format
        - `export_results_as_text()` - Export to human-readable text
        
        ### Configuration Options
        
        ```python
        config = {
            'embedding_type': 'fasttext' | 'sentence_transformer',
            'flow_type': 'ml' | 'rag',
            'enable_clustering': True | False,
            'force_rebuild_clustering': True | False,
            'rag_top_k': int,
            'countries': ['US', 'CN', 'JP'],
            # ⭐ NEW: Product Suggestion Options
            'enable_product_suggestions': True | False,
            'product_similarity_threshold': float,  # 0.0-1.0
            'max_suggestions_per_company': int,     # 1-10
            'enable_openai_enhance': True | False,
            'max_combinations_setting': int,       # 2-5
            'max_patents_per_company': int,        # Performance limit
            'max_abstract_length': int             # Text truncation
        }
        ```
        
        ### Pipeline Functions
        
        ```python
        # Run clustering analysis
        clustering_pipeline(config)
        
        # Load existing clustering
        analyzer = load_clustering_analyzer(embedding_type, country)
        
        # Find nearest cluster for query
        cluster_id, distance = analyzer.find_nearest_cluster(query_embedding)
        
        # ⭐ NEW: Product Suggestion Examples
        from pipelines.product_suggestion_pipeline import PatentProductSuggester
        
        # Initialize product suggester
        suggestion_config = {
            'similarity_threshold': 0.075,
            'top_k_suggestions': 5,
            'enable_openai_enhance': False,
            'max_combinations': 5
        }
        suggester = PatentProductSuggester(suggestion_config)
        
        # Generate suggestions for companies
        companies_data = [
            {
                'hojinid': '12345',
                'name': 'TechCorp Inc.',
                'keywords': ['ai', 'machine', 'learning', 'software'],
                'patents': [{'id': 'P001', 'abstract': 'AI system for...'}],
                'source': 'transformation_matrix'
            }
        ]
        
        results = suggester.suggest_products('AI technology', companies_data)
        
        # Export results
        json_path = suggester.export_results(results)
        text_path = suggester.export_results_as_text(results)
        ```
        """)

    def show_config_guide(self):
        """Show configuration guide"""
        st.markdown("""
        ## ⚙️ Configuration Guide
        
        ### Embedding Types
        
        **FastText:**
        - ✅ Faster processing
        - ✅ Lower memory usage
        - ✅ Suitable for large datasets
        - ❌ Less semantic understanding
        
        **Sentence Transformers:**
        - ✅ Better semantic understanding
        - ✅ More accurate results
        - ✅ Context-aware embeddings
        - ❌ Slower processing
        - ❌ Higher memory usage
        
        ### Flow Types
        
        **ML Approach:**
        - Uses transformation matrices
        - Good for structured queries
        - Requires pre-trained models
        - Two variants: matrix-only and neural network
        
        **RAG Approach:**
        - Uses semantic search with ChromaDB
        - Good for natural language queries
        - Requires vector database
        - Supports external company summaries
        
        ### Clustering Configuration
        
        **Algorithm Selection:**
        - **K-means**: Best for spherical clusters, scalable
        - **Hierarchical**: Good for nested structures, slower
        - **DBSCAN**: Handles noise and outliers, requires tuning
        
        **Performance Tuning:**
        - Adjust `CLUSTER_NUMBERS_RANGE` for different ranges
        - Modify `CLUSTERING_MAX_ITER` for convergence
        - Set `CLUSTERING_RANDOM_STATE` for reproducibility
        
        ### Product Suggestion Configuration ⭐ NEW
        
        **Core Settings:**
        - **Enable Product Suggestions**: Turn feature on/off
        - **Similarity Threshold**: Minimum score for suggestions (0.0-1.0)
        - **Max Suggestions**: Number of products per company (1-10)
        - **Max Combinations**: Keywords per product name (2-5)
        
        **AI Enhancement:**
        - **Enable OpenAI**: Use GPT models for name improvement
        - **Model Selection**: Choose GPT model (gpt-4o-mini default)
        - **API Key**: Required for OpenAI features (.env file)
        
        **Advanced Parameters:**
        ```python
        PRODUCT_SUGGESTION_CONFIG = {
            'alpha': 0.3,              # Keywords vs patents weight
            'beta': 0.3,               # Lexical vs semantic weight  
            'similarity_threshold': 0.075,  # Minimum similarity
            'max_keywords': 20,        # Keywords per analysis
            'max_combinations': 5,     # Max keyword combinations
            'use_patent_data': True    # Include patent analysis
        }
        ```
        
        ### Export Configuration ⭐ NEW
        
        **Automatic Export:**
        - **JSON Directory**: `data/suggestions/` (structured data)
        - **Text Directory**: `data/submissions/` (human-readable)
        - **File Naming**: `product_suggestions_YYYYMMDD_HHMMSS`
        - **Manual Re-export**: Available in UI after each query
        
        **File Management:**
        - Files created automatically after each query
        - Consistent relative path display
        - Size and timestamp information included
        - Export status shown in UI with file paths
        """)

    def show_troubleshooting(self):
        """Show troubleshooting guide"""
        st.markdown("""
        ## 🔧 Troubleshooting
        
        ### Common Issues
        
        **"Components not initialized"**
        - Solution: Click "🔄 Initialize Components" in sidebar
        - Check that all required files exist
        - Verify embedding models are downloaded
        
        **"Clustering not available"**
        - Run clustering pipeline first:
        ```bash
        python main.py --pipeline clustering --enable_clustering --embedding_type fasttext
        ```
        - Check `data/clustering/results/` for existing results
        
        **"No query results"**
        - Ensure models are trained first (dual_attn → patent_product train)
        - Check that embeddings are generated
        - Verify RAG vector database exists
        - Try different embedding types
        
        **"Pipeline execution failed"**
        - Check the Pipeline Manager tab for detailed error messages
        - Verify all dependencies are installed
        - Ensure sufficient disk space and memory
        - Check log files for detailed error information
        
        **"Memory errors during clustering"**
        - Use FastText instead of Sentence Transformers
        - Reduce `CLUSTER_NUMBERS_RANGE` in config
        - Close other applications
        - Consider using a smaller dataset for testing
        
        **"No product suggestions generated"** ⭐ NEW
        - Lower similarity threshold (try 0.05 or 0.03)
        - Check if companies have sufficient keywords/patents
        - Verify clustering data loaded properly
        - Enable debug mode for detailed logging
        
        **"Product suggestions showing 'unknown' company IDs"** ⭐ NEW
        - Fixed: Enhanced company ID resolution implemented
        - Ensure clustering analyzer is properly loaded
        - Check that company data is complete
        - Verify hojin_id mapping exists
        
        **"Only single-word products generated"** ⭐ NEW
        - Fixed: System now enforces 2+ keyword combinations
        - Check `max_combinations_setting` configuration
        - Verify keyword filtering is working properly
        - Use debug mode to see keyword processing
        
        **"Export files not created"** ⭐ NEW
        - Check write permissions for `data/suggestions/` and `data/submissions/`
        - Verify sufficient disk space
        - Check console logs for export error messages
        - Ensure output directories exist
        
        **"OpenAI enhancement not working"** ⭐ NEW
        - Verify `.env` file contains valid `OPENAI_API_KEY`
        - Check OpenAI API quota and billing
        - Try disabling OpenAI to test basic functionality
        - Check console for OpenAI API error messages
        
        ### Performance Tips
        
        1. **Use FastText for speed** - Especially for large datasets
        2. **Enable clustering for insights** - Provides valuable market segmentation
        3. **Start with smaller datasets** - Test functionality before full runs
        4. **Monitor memory usage** - Watch system resources during clustering
        5. **Use force rebuild sparingly** - Only when necessary to save time
        6. **Optimize product suggestions** ⭐ NEW:
           - Start with higher similarity threshold (0.1) and adjust down
           - Limit max suggestions per company (3-5) for faster processing
           - Disable OpenAI enhancement for faster results during testing
           - Use `max_combinations_setting=3` for quicker keyword processing
        7. **Manage export files** ⭐ NEW:
           - Clean old suggestion files regularly to save disk space
           - Monitor `data/suggestions/` and `data/submissions/` directories
           - Use System Status tab to check file sizes and counts
        
        ### File Locations
        
        - **Logs**: `pdzttb.log` (main log file)
        - **Clustering Results**: `data/clustering/results/`
        - **Performance Plots**: `data/output/img/`
        - **Models**: `data/clustering/models/`
        - **Config**: `streamlit_config.json`
        - **Product Suggestions (JSON)**: `data/suggestions/` ⭐ NEW
        - **Product Suggestions (Text)**: `data/submissions/` ⭐ NEW
        - **Core Data**: `data/CN_JP_US_data/` (patent & company data)
        - **Vector Database**: `data/vector_db/` (RAG embeddings)
        """)

    def show_product_suggestions_guide(self):
        """Show product suggestions documentation"""
        st.markdown("""
        ## 🛍️ Product Suggestions Guide
        
        ### Overview
        
        The **Product Suggestion Pipeline** is an AI-powered feature that generates relevant product names for companies based on:
        - 📊 **Company Keywords**: Extracted from business profiles  
        - 🔬 **Patent Analysis**: Technical abstracts and innovations
        - 🤖 **AI Enhancement**: Optional OpenAI integration for product name refinement
        
        ### How It Works
        
        **1. Data Collection**
        - Analyzes companies from both **Transformation Matrix** results and **Nearest Cluster** matches
        - Loads full patent abstracts (not truncated versions)
        - Extracts comprehensive keyword sets (not just display keywords)
        
        **2. Product Generation**
        - Creates combinations of 2-5 keywords (configurable)
        - Filters out single-word and invalid combinations
        - Uses semantic similarity for patent-product matching
        - Applies lexical and semantic weighting (alpha/beta parameters)
        
        **3. AI Enhancement** (Optional)
        - Uses OpenAI GPT models to improve product names
        - Makes names more market-ready and professional
        - Maintains technical accuracy while improving readability
        
        ### Configuration Options
        
        **Sidebar Settings:**
        ```
        🛍️ Product Suggestion Configuration
        ├── ✅ Enable Product Suggestions
        ├── 🎯 Similarity Threshold (0.075 default)
        ├── 📊 Max Suggestions per Company (5 default)  
        ├── 🤖 Enable OpenAI Enhancement
        └── 🔢 Max Keyword Combinations (2-5)
        ```
        
        **Key Parameters:**
        - **Similarity Threshold**: Minimum similarity score for suggestions (0.0-1.0)
        - **Alpha (α)**: Weight between keywords vs patents (0.3 default)
        - **Beta (β)**: Weight between lexical vs semantic similarity (0.3 default)
        - **Max Combinations**: Maximum keywords per product name (2-5)
        
        ### Usage Steps
        
        **1. Enable Feature**
        ```
        ☑️ Enable Product Suggestions (in sidebar)
        ```
        
        **2. Configure Settings**
        ```
        🎯 Similarity Threshold: 0.075
        📊 Max Suggestions: 5
        🤖 OpenAI Enhancement: Optional
        🔢 Max Combinations: 5
        ```
        
        **3. Run Query**
        - Execute any query in the Query Interface
        - Product suggestions automatically appear at the bottom
        - Results are auto-exported to files
        
        ### Output Files
        
        **Automatic Export** (after each query):
        - 📄 **JSON**: `data/suggestions/product_suggestions_YYYYMMDD_HHMMSS.json`
        - 📝 **Text**: `data/submissions/product_suggestions_YYYYMMDD_HHMMSS.txt`
        
        **JSON Structure:**
        ```json
        {
          "metadata": {
            "query": "your search query",
            "timestamp": "2025-01-XX XX:XX:XX",
            "total_companies": 10,
            "companies_with_suggestions": 8
          },
          "results": [
            {
              "company_name": "TechCorp Inc.",
              "company_id": "12345",
              "source": "transformation_matrix",
              "products": [
                {
                  "name": "Smart IoT Sensor",
                  "similarity_score": 0.85,
                  "keywords_used": ["smart", "iot", "sensor"],
                  "enhanced_name": "Smart IoT Environmental Sensor" 
                }
              ]
            }
          ]
        }
        ```
        
        ### Integration with Clustering
        
        **Dual Source Processing:**
        - **🔬 Transformation Matrix Companies**: Direct query results
        - **🎯 Nearest Cluster Companies**: Companies from similar market segments
        
        **Enhanced Data Access:**
        - Full company keyword sets (not display-limited)
        - Complete patent abstracts for better analysis
        - Dynamic company ID resolution from clustering data
        
        ### Performance Considerations
        
        **Optimization Settings:**
        ```python
        MAX_PATENTS_PER_COMPANY = 20  # Patent processing limit
        MAX_KEYWORDS = 20            # Keyword extraction limit  
        MAX_COMBINATIONS = 5         # Keyword combination limit
        ```
        
        **Processing Time:**
        - ~2-5 seconds per company (without OpenAI)
        - ~5-15 seconds per company (with OpenAI)
        - Parallel processing for better performance
        
        ### Troubleshooting
        
        **"No product suggestions generated"**
        - Lower similarity threshold (try 0.05)
        - Check if companies have sufficient keywords/patents
        - Verify clustering data is loaded properly
        
        **"Only getting single-word products"**
        - Fixed: System now only generates 2+ keyword combinations
        - Check `max_combinations_setting` in configuration
        
        **"Company IDs showing as 'unknown'"**
        - Fixed: Enhanced company ID resolution from clustering data
        - Verify clustering analyzer is properly loaded
        
        **"Export files not created"**
        - Check write permissions for `data/suggestions/` and `data/submissions/`
        - Verify sufficient disk space
        - Check console for export error messages
        """)

    def show_export_file_guide(self):
        """Show export and file management documentation"""
        st.markdown("""
        ## 📁 Export & File Management Guide
        
        ### Automatic Export Features
        
        **Product Suggestions Export** ⭐ NEW
        - **When**: Automatically after each query with product suggestions enabled
        - **Formats**: JSON (structured data) + Text (human-readable)
        - **Location**: `data/suggestions/` and `data/submissions/`
        - **Naming**: `product_suggestions_YYYYMMDD_HHMMSS.json/txt`
        
        **Export Status Display:**
        ```
        📄 JSON: data/suggestions/product_suggestions_20250130_143022.json
        📝 Text: data/submissions/product_suggestions_20250130_143022.txt
        
        📊 Export Summary:
        • Companies Analyzed: 12
        • Companies with Suggestions: 8  
        • Total Products: 34
        • Average Similarity: 0.124
        ```
        
        ### File System Structure
        
        ```
        FullFlow/
        ├── data/
        │   ├── CN_JP_US_data/          # Core dataset files
        │   │   ├── us_web_with_company.csv
        │   │   ├── us_patent202506.csv
        │   │   ├── cleaned_patents.csv
        │   │   └── fasttext_web_patent.vec
        │   ├── clustering/             # Clustering analysis results
        │   │   ├── models/            # Trained clustering models
        │   │   └── results/           # Performance data & assignments
        │   ├── suggestions/           # Product suggestion JSON exports ⭐ NEW
        │   ├── submissions/           # Product suggestion text exports ⭐ NEW
        │   ├── outputs/              # General pipeline outputs
        │   └── vector_db/            # RAG vector database
        ├── models/                   # ML model checkpoints
        └── streamlit_config.json     # UI configuration persistence
        ```
        
        ### Enhanced File Status Monitoring
        
        **System Status Tab** shows comprehensive file checking:
        
        **📁 Directory Status:**
        ```
        ✅ Data Directory: data/CN_JP_US_data (X files)
        ✅ Models Directory: models (5 files)  
        ✅ Clustering Results: data/clustering/results (12 files)
        ✅ Vector Database: data/vector_db (1 files)
        ```
        
        **📄 Critical Data Files:**
        ```
        ✅ US Web Data: data/CN_JP_US_data/us_web_with_company.csv (15.2 MB)
        ✅ US Patent Data: data/CN_JP_US_data/us_patent202506.csv (2.1 GB)
        ✅ FastText Vectors: data/CN_JP_US_data/fasttext_web_patent.vec (1.8 GB)
        ```
        
        ### Enhanced Debug Information
        
        **Clustering Debug** (improved display):
        ```
        📁 File Status:
        ✅ evaluation_results: data/clustering/results/US_fasttext_evaluation_results.json
        ✅ best_model: data/clustering/models/US_fasttext_best_clustering_model.pkl
        ✅ assignments: data/clustering/results/US_fasttext_cluster_assignments.csv
        ❌ visualization: data/clustering/results/US_fasttext_clustering_visualization.png
        ```
        
        ### Manual Export Options
        
        **Product Suggestions Re-export:**
        - 📄 **Re-export to JSON**: Regenerate JSON file with current timestamp
        - 📝 **Re-export to Text**: Regenerate human-readable text file
        - **Location**: Same directories as automatic export
        
        ### File Management Best Practices
        
        **1. Regular Cleanup**
        ```bash
        # Clean old suggestion files (keep last 10)
        ls -t data/suggestions/*.json | tail -n +11 | xargs rm -f
        ls -t data/submissions/*.txt | tail -n +11 | xargs rm -f
        ```
        
        **2. Backup Important Results**
        ```bash
        # Backup clustering results
        cp -r data/clustering/ backup/clustering_$(date +%Y%m%d)/
        
        # Backup recent suggestions  
        cp data/suggestions/product_suggestions_*.json backup/suggestions/
        ```
        
        **3. Monitor Disk Space**
        - **Large Files**: Patent data (~2GB), FastText vectors (~2GB)
        - **Growing**: Product suggestions, clustering results
        - **Check**: Use System Status tab for monitoring
        
        ### File Formats & Compatibility
        
        **JSON Export** (machine-readable):
        - Full structured data with metadata
        - Easy import into other tools/databases
        - Preserves all similarity scores and parameters
        
        **Text Export** (human-readable):
        - Formatted for easy reading and sharing
        - Includes summary statistics
        - Suitable for reports and presentations
        
        **Clustering Files**:
        - **Models**: `.pkl` (Python pickle format)
        - **Assignments**: `.csv` (Excel/database compatible)
        - **Plots**: `.png` (standard image format)
        
        ### Troubleshooting File Issues
        
        **"Permission denied" errors**
        - Check write permissions for output directories
        - Run with appropriate user privileges
        - Ensure directories exist and are writable
        
        **"Disk space" errors**
        - Clean old export files regularly
        - Monitor large dataset files
        - Consider using external storage for backups
        
        **"File not found" errors**
        - Use System Status tab to verify file existence
        - Check file paths in configuration
        - Ensure required pipelines have been run
        
        **"Corrupted export files"**
        - Use manual re-export options
        - Check console logs for detailed error messages
        - Verify sufficient disk space during export
        """)

def main():
    """Main function to run the Streamlit app"""
    app = FullFlowApp()
    app.render_app()

if __name__ == "__main__":
    main() 