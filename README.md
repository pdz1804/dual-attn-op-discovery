# 🔬 FullFlow Patent-Product Matching System

**Advanced AI-powered system for patent-product matching with dual attention models, transformation matrices, RAG, clustering analysis, and intelligent product suggestions**

## 🌟 Overview

FullFlow is a comprehensive system that provides multiple approaches for patent-product matching and intelligent product suggestion:

1. **🧠 ML Flow**: Uses Dual Attention Model + Transformation Matrix
2. **🔍 RAG Flow**: Uses Retrieval-Augmented Generation with ChromaDB
3. **🎯 Product Suggestion**: AI-powered domain-aware product generation (NEW)

All flows support **FastText** and **Sentence Transformer** embeddings, with advanced clustering analysis for market insights.

## 🔍 Innovation Discovery System Overview

<details>
<summary><strong>Toggle to view Web Scraping Process</strong></summary>
<img src="imgs/IDS_webscraping.png" alt="Web Scraping Process" style="max-width: 100%;">
<p><em>This diagram illustrates how HTML pages are extracted from the company's website and Wayback Machine snapshots for further analysis.</em></p>
</details>

<details>
<summary><strong>Toggle to view Dual Attention Model</strong></summary>
<img src="imgs/IDS_Dual Attention Model.png" alt="Dual Attention Model" style="max-width: 100%;">
<p><em>This figure demonstrates the dual attention mechanism that captures both word-level and page-level context to generate firm representations.</em></p>
</details>

<details>
<summary><strong>Toggle to view Transformation Matrix Pipeline</strong></summary>
<img src="imgs/IDS_Approach01_Transformation Matrix.png" alt="Transformation Matrix Pipeline" style="max-width: 100%;">
<p><em>This shows the pipeline for transforming patents and firm keywords into vector spaces and matching them for discovery.</em></p>
</details>

## 🛠️ Installation

<details>
<summary><strong>Click to view installation commands</strong></summary>

```bash
# Clone the repository
git clone https://github.com/pdz1804/dual-attn-op-discovery.git FullFlow
cd FullFlow

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models and data
python -c "import spacy; spacy.download('en_core_web_sm')"

# Optional: Install OpenAI for product name enhancement
pip install openai python-dotenv
```

</details>

## 🏗️ System Architecture & Flows

<details>
<summary><strong>View System Flow Diagrams</strong></summary>

### Flow 1: ML Approach (Dual Attention + Transformation Matrix)
```
Training: Data → Dual Attention Model → Keywords → Transformation Matrix → Clustering
Testing:  Input Query  → Keywords → Transformation Matrix → Results + Product Suggestions
```

### Flow 2: RAG Approach (Retrieval-Augmented Generation)
```
Training: Data → Dual Attention Model → Company Documents → ChromaDB → Clustering  
Testing:  Input Query → Query Embedding → ChromaDB Search → Semantic Matching → Results + Product Suggestions
```

### Flow 3: Product Suggestion (AI-Enhanced)
```
Input: User Query + Company Results → Domain Detection → Multi-Keyword Combinations → 
AI Enhancement (Optional) → Domain-Specific Products → JSON + Text Export
```

</details>

## 🚀 Complete Workflows

## 🧠 Flow 1: ML Approach

### 📚 Training Phase

<details>
<summary><strong>View ML Training Commands</strong></summary>

#### Step 1: Train Dual Attention Model (Required for Both Flows)
```bash
# Train dual attention model (always uses FastText)
python main.py --pipeline dual_attn
```

#### Step 2: Train Transformation Matrix

**With FastText embeddings:**
```bash
# Train transformation matrix with FastText embeddings
python main.py --pipeline patent_product --mode train --embedding_type fasttext
```

**With Sentence Transformer embeddings:**
```bash
# Train transformation matrix with Sentence Transformer embeddings
python main.py --pipeline patent_product --mode train --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2
```

**New Fixes on 23/07/2025:**

- Now we have create 2 new configs `model_type` and `approx_method`
- For `model_type`, we can now choose between `linear` and `mlp`, which would be use to train model to map between patents and products (firms)
- For `approx_method`, now we can choose between `'sampling', 'polynomial', 'kernel'`. **Sampling** here we mean that we would take a samples of data for the trained model to train on, then use that predicted values of the model to train the mapped model. **Polynomial** here we mean the same as **Sampling** but we would use **Polynomial** function to train as the mapped model. For the **kernel**, we mean the same thing but we would use a **Kernel**. 
- Example usage would be 
  ```bash
  python main.py --pipeline patent_product --mode train --embedding_type fasttext --model_type linear --approx_method sampling
  ```

#### Step 3: Build Clustering Analysis

**For FastText embeddings:**
```bash
# Build clustering analysis with FastText
python main.py --pipeline clustering --embedding_type fasttext --enable_clustering
```

**For Sentence Transformer embeddings:**
```bash
# Build clustering analysis with Sentence Transformers
python main.py --pipeline clustering --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering
```

**Note**: Should add `--force_rebuild_clustering` if you want to build the second clustering things

</details>

### 🔍 Testing/Inference Phase (With Queries)

<details>
<summary><strong>View ML Testing Commands</strong></summary>

**Test and Chat with FastText:**
```bash
# Test ML approach with FastText
python main.py --pipeline patent_product --mode test --embedding_type fasttext --enable_clustering

# Interactive chat with FastText
python main.py --pipeline patent_product --mode chat --embedding_type fasttext --enable_clustering

# With product suggestions enabled
python main.py --pipeline patent_product --mode test --embedding_type fasttext --enable_clustering --enable_product_suggestions
```

**Test and Chat with Sentence Transformers:**
```bash
# Test ML approach with Sentence Transformers
python main.py --pipeline patent_product --mode test --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering

# Interactive chat with Sentence Transformers
python main.py --pipeline patent_product --mode chat --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering

# With AI-enhanced product suggestions
python main.py --pipeline patent_product --mode chat --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering --enable_product_suggestions --enable_openai_enhance
```

</details>

---

## 🔍 Flow 2: RAG Approach

### 📚 Training Phase 

<details>
<summary><strong>View RAG Training Commands</strong></summary>

#### Step 1: Train Dual Attention Model (Same as ML Flow)
```bash
# Train dual attention model (required for RAG document creation)
python main.py --pipeline dual_attn
```

#### Step 2: Build RAG Vector Database

**With FastText embeddings:**
```bash
# Build RAG system with FastText embeddings
python main.py --pipeline patent_product --mode train --use_rag --embedding_type fasttext
```

**With Sentence Transformer embeddings:**
```bash
# Build RAG system with Sentence Transformer embeddings
python main.py --pipeline patent_product --mode train --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2
```

**Note**: should add `--force_rebuild_rag` if want to rebuild the RAG.

#### Step 3: Build Clustering Analysis (Same as ML Flow)

**For FastText embeddings:**
```bash
# Build clustering for RAG with FastText
python main.py --pipeline clustering --embedding_type fasttext --enable_clustering
```

**For Sentence Transformer embeddings:**
```bash
# Build clustering for RAG with Sentence Transformers
python main.py --pipeline clustering --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering
```

</details>

### 🔍 Testing/Inference Phase (With Queries)

<details>
<summary><strong>View RAG Testing Commands</strong></summary>

**Test and Chat with FastText:**
```bash
# Test RAG approach with FastText
python main.py --pipeline patent_product --mode test --use_rag --embedding_type fasttext --rag_top_k 5 --enable_clustering

# Interactive chat with RAG (FastText)
python main.py --pipeline patent_product --mode chat --use_rag --embedding_type fasttext --rag_top_k 5 --enable_clustering

# Direct RAG query with FastText
python main.py --pipeline rag_only --embedding_type fasttext --query "machine learning algorithms for medical diagnosis"

# RAG with product suggestions
python main.py --pipeline patent_product --mode test --use_rag --embedding_type fasttext --rag_top_k 5 --enable_clustering --enable_product_suggestions
```

**Test and Chat with Sentence Transformers:**
```bash
# Test RAG approach with Sentence Transformers
python main.py --pipeline patent_product --mode test --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering

# Interactive chat with RAG (Sentence Transformers)
python main.py --pipeline patent_product --mode chat --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering

# Direct RAG query with Sentence Transformers
python main.py --pipeline rag_only --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --query "renewable energy storage systems"

# RAG with AI-enhanced product suggestions
python main.py --pipeline patent_product --mode chat --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering --enable_product_suggestions --enable_openai_enhance
```

</details>

---

## 🎯 Product Suggestion Pipeline 

**Next-generation AI-powered product suggestion system with domain-aware generation, multi-source integration, and professional name enhancement**

### 🌟 Overview

The Product Suggestion Pipeline is a comprehensive AI system that analyzes patent abstractions and company data to generate highly relevant commercial product suggestions. It combines traditional ML techniques, deep learning, and optional AI enhancement to provide intelligent, domain-specific product recommendations with complete transparency and traceability.

### 🔬 Advanced Methods and Techniques

#### **1. Hybrid Multi-Layer Similarity Scoring**
- **Lexical Similarity**: TF-IDF + Jaccard similarity for precise keyword matching
- **Semantic Similarity**: Sentence Transformers (all-MiniLM-L6-v2) for deep contextual understanding
- **Patent-Company Fusion**: Combines keyword-based and patent-based similarities
- **Configurable Weighting**: Fine-tunable α (keyword vs patent) and β (lexical vs semantic) parameters

#### **2. Domain-Aware Intelligence (10+ Domains)**
- **Pharmaceutical**: Inhibitors, therapies, compounds, treatments
- **Medical**: Devices, scanners, diagnostics, imaging equipment
- **Technology**: Algorithms, platforms, frameworks, APIs
- **Manufacturing**: Machines, processes, automation, production lines
- **Energy**: Batteries, generators, solar panels, renewable systems
- **Materials**: Composites, polymers, coatings, nanoparticles
- **Automotive**: Engines, sensors, control systems, assemblies
- **Electronics**: Chips, circuits, processors, displays
- **Agricultural**: Fertilizers, growth enhancers, crop protection
- **Financial**: Trading platforms, analytics, risk management

#### **3. Enhanced Keyword Extraction & Processing**
- **Full Abstract Processing**: Utilizes original patent abstracts instead of tokenized versions
- **Normalization Pipeline**: Lowercase normalization, special character filtering
- **Multi-Context Limits**: Configurable extraction limits for different processing contexts

#### **4. Multi-Source Data Integration**
- **Company Keywords**: From dual attention training pipeline
- **Original Patent Abstracts**: Full patent text analysis with fallback mechanisms
- **Clustering Integration**: Companies from both transformation matrix top-k AND nearest clusters
- **Source Tracking**: Complete traceability of company origins (transformation matrix vs clusters)

#### **5. Professional AI Enhancement**
- **OpenAI Integration**: GPT-4o-mini for professional product name refinement
- **Dual Name System**: Maintains both original and enhanced names for transparency
- **Context-Aware Enhancement**: Company profile and query-specific improvements
- **Selective Enhancement**: Only processes top products to optimize performance and costs

### 🚀 Usage Examples

<details>
<summary><strong>View Product Suggestion Usage Commands</strong></summary>

#### **Standalone Product Suggestion Pipeline**
```bash
# Interactive product suggestion with domain detection
python main.py --pipeline product_suggestion

# With specific pharmaceutical query
python main.py --pipeline product_suggestion --query "SGK-1 inhibitor compounds for diabetes treatment"

# With technology domain query
python main.py --pipeline product_suggestion --query "machine learning algorithms for predictive analytics"
```

#### **Integrated with Existing Pipelines**
```bash
# Enable product suggestions in test mode (processes both matrix + cluster companies)
python main.py --pipeline patent_product --mode test --enable_product_suggestions

# Enable with OpenAI enhancement for professional names
python main.py --pipeline patent_product --mode chat --enable_product_suggestions --enable_openai_enhance

# Configure similarity threshold for broader/narrower suggestions
python main.py --pipeline patent_product --mode test --enable_product_suggestions --product_similarity_threshold 0.03
```

#### **Advanced Configuration Examples**
```bash
# Full configuration with all options
python main.py --pipeline patent_product --mode test \
  --enable_product_suggestions \
  --product_similarity_threshold 0.05 \
  --enable_openai_enhance \
  --embedding_type sentence_transformer \
  --enable_clustering

# Disable debug logging for clean output
python main.py --pipeline product_suggestion --query "renewable energy storage"
```

</details>

#### **Configuration Parameters**

<details>
<summary><strong>View Configuration Parameters (configs/hyperparams.py)</strong></summary>

```python
# Core product suggestion configuration
PRODUCT_SUGGESTION_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',        # Sentence transformer model
    'alpha': 0.6,                           # Weight: keyword vs patent similarity (0-1)
    'beta': 0.2,                            # Weight: lexical vs semantic (lower = more semantic)
    'similarity_threshold': 0.05,           # Minimum similarity threshold (lowered for real data)
    'top_k_suggestions': 5,                 # Number of product suggestions per company
    'max_keywords': 20,                     # Maximum keywords to extract per text
    'max_combinations': 5,                  # Maximum keyword combinations for product generation
    'use_patent_data': True,                # Whether to use company patent data
    'enable_openai_enhance': False,         # Use OpenAI to enhance product names
    'openai_model': 'gpt-4o-mini',         # OpenAI model for enhancement
    'output_directory': 'data/suggestions', # JSON output directory
    'submissions_directory': 'data/submissions', # Text output directory
    'debug_enabled': False                  # Enable debug logging
}

# Patent processing limits
MAX_PATENTS_PER_COMPANY = 20        # Patents for similarity computation
MAX_PATENTS_FOR_ANALYSIS = 20       # Patents for keyword extraction

# Keyword extraction limits for different contexts
KEYWORD_EXTRACTION_LIMITS = {
    'similarity_computation': 15,    # Keywords for similarity comparison
    'patent_analysis': 10,          # Keywords per patent analysis
    'theme_extraction': 5,          # Keywords for theme generation
    'top_keywords_limit': 15,       # Top keywords from frequency analysis
    'theme_combinations': 5         # Max theme combinations to generate
}
```

</details>

### 🔧 OpenAI Integration Setup

<details>
<summary><strong>View OpenAI Setup Instructions</strong></summary>

1. **Create Environment File**:
   ```bash
   cp .env.template .env
   ```

2. **Add Your OpenAI API Key**:
   ```bash
   # In .env file
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

3. **Install Dependencies**:
   ```bash
   pip install openai python-dotenv
   ```

</details>

### 📊 Comprehensive Output Format

#### **Dual Output System**
- **JSON Format**: `data/suggestions/product_suggestions_YYYYMMDD_HHMMSS.json` (machine-readable)
- **Text Format**: `data/submissions/product_suggestions_YYYYMMDD_HHMMSS.txt` (human-readable)

#### **Enhanced JSON Structure**

<details>
<summary><strong>View JSON Output Example</strong></summary>

```json
{
  "summary": {
    "query": "SGK-1 inhibitor compounds for diabetes treatment",
    "timestamp": "2025-01-29T14:55:39",
    "total_companies_processed": 10,
    "companies_with_suggestions": 8,
    "total_products_suggested": 25
  },
  "results": [
    {
      "company_id": "704524",
      "company_name": "Delcath Systems",
      "source": "transformation_matrix",          // Source tracking
      "company_similarity": 0.1202,
      "keyword_similarity": 0.0156,
      "patent_similarity": 0.2771,
      "patent_stats": {
        "total_patents": 20,
        "full_abstracts_used": 20,
        "fallback_abstracts_used": 0
      },
      "products": [
        {
          "product_name": "Transdermal Therapeutic Delivery Device",  // Enhanced name
          "original_name": "Device Melphalan Bodys",                  // Original name (if enhanced)
          "score": 0.4708,
          "lexical_similarity": 0.0556,
          "semantic_similarity": 0.5746
        },
        {
          "product_name": "Advanced Device Melphalan Delivery",       // No original_name = unchanged
          "score": 0.3824,
          "lexical_similarity": 0.0556,
          "semantic_similarity": 0.4642
        }
      ],
      "metadata": {
        "keywords_count": 640,
        "patent_keywords_count": 45,
        "total_candidates_generated": 120
      }
    }
  ]
}
```

</details>

#### **Human-Readable Text Format**

<details>
<summary><strong>View Text Output Example</strong></summary>

```
================================================================================
PRODUCT SUGGESTIONS REPORT
================================================================================
Query: SGK-1 inhibitor compounds for diabetes treatment
Generated: 2025-01-29 14:55:39
Total Companies: 10
Companies with Suggestions: 8

RANK 1: Delcath Systems
------------------------------------------------------------
Company ID: 704524
Source: Transformation Matrix Top-K
Overall Similarity Score: 0.1202
Keyword Similarity: 0.0156
Patent Similarity: 0.2771
Patent Processing: 20/20 full abstracts used

SUGGESTED PRODUCTS (5 products):
  1. Transdermal Therapeutic Delivery Device
     (Original: Device Melphalan Bodys)
     Overall Score: 0.4708
     Lexical Similarity: 0.0556
     Semantic Similarity: 0.5746
  2. Advanced Device Melphalan Delivery
     Overall Score: 0.3824
     Lexical Similarity: 0.0556
     Semantic Similarity: 0.4642
```

</details>

### 🧪 Comprehensive Testing

<details>
<summary><strong>View Testing Commands</strong></summary>

```bash
# Run full test suite (10+ test cases)
python test/test_product_suggestions.py

# Test specific functionality
python -c "from test.test_product_suggestions import test_basic_functionality; test_basic_functionality()"

# Test edge cases and export functionality
python -c "from test.test_product_suggestions import test_edge_cases, test_export_functionality; test_edge_cases(); test_export_functionality()"
```

</details>

### 🔧 Key Features Summary

✅ **10+ Domain Intelligence**: Pharmaceutical, Medical, Technology, Manufacturing, etc.  
✅ **Dual Name System**: Original + AI-enhanced names with full transparency  
✅ **Source Tracking**: Know if companies come from transformation matrix or clusters  
✅ **Full Patent Processing**: Uses original abstracts (not tokenized)  
✅ **Configurable Limits**: All processing limits centralized in `configs/hyperparams.py`  
✅ **Comprehensive Testing**: 10+ test cases covering all functionality

---

## 🎛️ Advanced Features

### Streamlit Web Interface

<details>
<summary><strong>View Streamlit Configuration</strong></summary>

Launch the interactive web interface:

```bash
# Start Streamlit app
streamlit run streamlit_app.py
```

The web interface provides:
- **🔍 Query Interface**: Interactive query processing
- **🎯 Clustering Analysis**: Visual cluster exploration
- **📊 Demo Examples**: Sample results without real data
- **🔧 System Status**: Component monitoring
- **📚 Documentation**: In-app help and guides

</details>

### Advanced Configuration Options

<details>
<summary><strong>View Advanced Configuration Commands</strong></summary>

#### Clustering Configuration
```bash
# Force rebuild clustering models
--force_rebuild_clustering

# Enable clustering analysis
--enable_clustering
```

#### RAG Configuration
```bash
# Use external company summaries instead of dual attention keywords
--rag_use_external_summaries

# Adjust number of retrieved documents
--rag_top_k 10

# Force rebuild RAG vector database
--force_rebuild_rag
```

#### Display Configuration
```bash
# Customize keyword display in results  
python main.py --pipeline patent_product --mode test --max_keywords_display 50

# Example: Show only top 20 keywords
python main.py --pipeline patent_product --mode test --max_keywords_display 20
```

**Display Options:**
- `--max_keywords_display`: Number of keywords shown in results (default: 50)
- `KEYWORDS_PER_COMPANY_CLUSTER`: Keywords shown per company in cluster view (default: 5)  
- `COMPANIES_PER_CLUSTER_DISPLAY`: Sample companies shown per cluster (default: 3)

#### UI Flow Configuration
```bash
# Set UI flow type
--ui_flow_type ml    # or rag

# Enable/disable components
--enable_dual_attention
--enable_transformation_matrix
```

</details>

## Enhanced Output Structure

<details>
<summary><strong>View Complete File Structure</strong></summary>

```
FullFlow/
├── data/
│   ├── clustering/
│   │   ├── models/
│   │   ├── results/
│   │   └── analysis/
│   ├── suggestions/                # NEW: Product suggestion JSON outputs
│   │   ├── product_suggestions_20250129_145539.json
│   │   └── product_suggestions_*.json
│   ├── submissions/                # NEW: Human-readable text outputs
│   │   ├── product_suggestions_20250129_145539.txt
│   │   └── product_suggestions_*.txt
│   ├── outputs/
│   │   ├── img/                    # Clustering plots
│   │   └── enhanced/               # Enhanced model outputs
│   └── vector_db/                  # ChromaDB storage
├── models/
│   ├── trained_models/
│   └── transformation_matrices/
├── pipelines/                      # Core pipeline modules
│   ├── dual_attention_pipeline.py
│   ├── patent_product_pipeline.py
│   ├── clustering_pipeline.py
│   └── product_suggestion_pipeline.py  # NEW: Product suggestion system
├── configs/
│   ├── hyperparams.py             # ENHANCED: All product configs centralized
│   └── paths.py
├── test/
│   ├── test_product_suggestions.py  # NEW: Comprehensive test suite
│   └── test.py
├── streamlit_data/
├── .env                           # NEW: OpenAI API configuration
└── pdzttb.log                     # Main log file
```

</details>

## 📊 Testing & Performance

### Example Queries for Testing

<details>
<summary><strong>View Example Testing Queries</strong></summary>

#### AI and Medical Technology
```bash
python main.py --pipeline rag_only --query "machine learning algorithms for medical diagnosis and patient monitoring systems"
```

#### Energy and Sustainability
```bash
python main.py --pipeline rag_only --query "renewable energy storage systems with lithium-ion battery technology"
```

#### Autonomous Vehicles
```bash
python main.py --pipeline rag_only --query "autonomous vehicle navigation systems using computer vision"
```

#### Quantum Computing
```bash
python main.py --pipeline rag_only --query "quantum computing applications for cryptographic security"
```

</details>

### Performance Comparison

| Method | Embedding Type | Speed | Memory | Use Case |
|--------|---------------|-------|--------|----------|
| ML-Matrix | FastText | Fast | Low | Structured queries |
| ML-Matrix | Sentence | Medium | Medium | Semantic matching |
| RAG | FastText | Medium | Medium | Natural language |
| RAG | Sentence | Slow | High | Complex queries |
| Product Suggestions | Sentence | Medium | Medium | Commercial products |

### Clustering Insights

The clustering analysis provides:

- **📈 Multi-Metric Evaluation**: Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- **🏆 Optimal Cluster Selection**: Automated ranking and selection
- **📊 Visual Analysis**: Performance plots and cluster distributions
- **🏢 Market Segmentation**: Company grouping by technology domains
- **🔍 Query-Cluster Matching**: Find nearest clusters for queries

## 📋 Reference Guide

### Quick Start Examples

#### Complete ML Pipeline (FastText)

<details>
<summary><strong>View Complete ML Pipeline Commands</strong></summary>

```bash
# === TRAINING PHASE ===
# 1. Train dual attention
python main.py --pipeline dual_attn

# 2. Train transformation matrix
python main.py --pipeline patent_product --mode train --embedding_type fasttext

# 3. Build clustering
python main.py --pipeline clustering --embedding_type fasttext --enable_clustering

# === TESTING PHASE ===
# 4. Test with clustering and product suggestions
python main.py --pipeline patent_product --mode test --embedding_type fasttext --enable_clustering --enable_product_suggestions

# 5. Interactive chat with AI-enhanced suggestions
python main.py --pipeline patent_product --mode chat --embedding_type fasttext --enable_clustering --enable_product_suggestions --enable_openai_enhance
```

</details>

#### Complete RAG Pipeline (Sentence Transformers)

<details>
<summary><strong>View Complete RAG Pipeline Commands</strong></summary>

```bash
# === TRAINING PHASE ===
# 1. Train dual attention (for document creation)
python main.py --pipeline dual_attn

# 2. Build RAG database
python main.py --pipeline patent_product --mode train --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2

# 3. Build clustering
python main.py --pipeline clustering --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering

# === TESTING PHASE ===
# 4. Test RAG with clustering and product suggestions
python main.py --pipeline patent_product --mode test --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering --enable_product_suggestions

# 5. Interactive RAG chat with AI enhancement
python main.py --pipeline patent_product --mode chat --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering --enable_product_suggestions --enable_openai_enhance

# 6. Direct RAG queries
python main.py --pipeline rag_only --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --query "your query here"
```

</details>

### Pipeline Reference

#### Available Pipelines
- **`dual_attn`**: Train dual attention model (always first step)
- **`patent_product`**: Main pipeline (requires --mode)
  - `--mode train`: Build models/databases (no queries)
  - `--mode test`: Test with predefined queries + optional product suggestions
  - `--mode chat`: Interactive querying + optional product suggestions
- **`rag_only`**: Direct RAG querying (requires --query)
- **`clustering`**: Build clustering analysis
- **`product_suggestion`**: AI-powered product suggestion system (ENHANCED)

#### Command Line Arguments Reference

<details>
<summary><strong>View All Command Line Arguments</strong></summary>

```bash
# Core pipeline arguments
--pipeline                    # Pipeline type (dual_attn, patent_product, etc.)
--mode                       # Mode for patent_product (train, test, chat)
--embedding_type             # fasttext or sentence_transformer
--sentence_transformer_model # Model name (e.g., all-MiniLM-L6-v2)

# Product suggestion arguments
--enable_product_suggestions     # Enable product suggestions
--product_similarity_threshold   # Similarity threshold (0-1, default: 0.05)
--enable_openai_enhance         # Use OpenAI for name enhancement

# Clustering arguments
--enable_clustering          # Enable clustering analysis
--force_rebuild_clustering   # Force rebuild clustering models

# RAG arguments
--use_rag                   # Use RAG approach
--rag_top_k                 # Number of RAG documents to retrieve
--force_rebuild_rag         # Force rebuild RAG database

# Display arguments
--max_keywords_display      # Number of keywords to show in results
```

</details>

#### Training vs Testing vs Product Suggestions
- **Training Phase**: Build models, databases, clustering (no queries needed)
- **Testing Phase**: Use trained components with actual queries
- **Product Suggestions**: 
  - **Standalone**: `--pipeline product_suggestion`
  - **Integrated**: Add `--enable_product_suggestions` to test/chat modes
  - **Enhanced**: Add `--enable_openai_enhance` for AI-improved names

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

## 👥 Authors

This project was developed by:
- **[Nguyen Quang Phu (pdz1804)](https://github.com/pdz1804)** and Tieu Tri Bang.

## 📞 Support

For questions, issues, or contributions:
- 🐛 **Issues**: [GitHub Issues](https://github.com/pdz1804/dual-attn-op-discovery/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/pdz1804/dual-attn-op-discovery/discussions)
- 📧 **Email**: [quangphunguyen1804@gmail.com](mailto:quangphunguyen1804@gmail.com)


