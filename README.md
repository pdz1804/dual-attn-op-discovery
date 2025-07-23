# 🔬 FullFlow Patent-Product Matching System

**Advanced AI-powered system for patent-product matching with dual attention models, transformation matrices, RAG, and clustering analysis**

## 🌟 Overview

FullFlow is a comprehensive system that provides two main approaches for patent-product matching:

1. **🧠 ML Flow**: Uses Dual Attention Model + Transformation Matrix
2. **🔍 RAG Flow**: Uses Retrieval-Augmented Generation with ChromaDB

Both flows support **FastText** and **Sentence Transformer** embeddings, with advanced clustering analysis for market insights.

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd FullFlow

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models and data
python -c "import spacy; spacy.download('en_core_web_sm')"
```

## 📊 System Flows

### Flow 1: ML Approach (Dual Attention + Transformation Matrix)
```
Training: Data → Dual Attention Model → Keywords → Transformation Matrix → Clustering
Testing:  Input Query  → Keywords → Transformation Matrix → Results
```

### Flow 2: RAG Approach (Retrieval-Augmented Generation)
```
Training: Data → Dual Attention Model → Company Documents → ChromaDB → Clustering  
Testing:  Input Query → Query Embedding → ChromaDB Search → Semantic Matching → Results
```

## 🚀 Complete Workflows

## 🧠 Flow 1: ML Approach

### 📚 Training Phase

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

### 🔍 Testing/Inference Phase (With Queries)

**Test and Chat with FastText:**
```bash
# Test ML approach with FastText
python main.py --pipeline patent_product --mode test --embedding_type fasttext --enable_clustering

# Interactive chat with FastText
python main.py --pipeline patent_product --mode chat --embedding_type fasttext --enable_clustering
```

**Test and Chat with Sentence Transformers:**
```bash
# Test ML approach with Sentence Transformers
python main.py --pipeline patent_product --mode test --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering

# Interactive chat with Sentence Transformers
python main.py --pipeline patent_product --mode chat --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering
```

---

## 🔍 Flow 2: RAG Approach

### 📚 Training Phase 

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

### 🔍 Testing/Inference Phase (With Queries)

**Test and Chat with FastText:**
```bash
# Test RAG approach with FastText
python main.py --pipeline patent_product --mode test --use_rag --embedding_type fasttext --rag_top_k 5 --enable_clustering

# Interactive chat with RAG (FastText)
python main.py --pipeline patent_product --mode chat --use_rag --embedding_type fasttext --rag_top_k 5 --enable_clustering

# Direct RAG query with FastText
python main.py --pipeline rag_only --embedding_type fasttext --query "machine learning algorithms for medical diagnosis"
```

**Test and Chat with Sentence Transformers:**
```bash
# Test RAG approach with Sentence Transformers
python main.py --pipeline patent_product --mode test --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering

# Interactive chat with RAG (Sentence Transformers)
python main.py --pipeline patent_product --mode chat --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering

# Direct RAG query with Sentence Transformers
python main.py --pipeline rag_only --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --query "renewable energy storage systems"
```

## 🎛️ Streamlit Web Interface

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

## 🔧 Advanced Configuration

### Clustering Options
```bash
# Force rebuild clustering models
--force_rebuild_clustering

# Enable clustering analysis
--enable_clustering
```

### RAG Configuration
```bash
# Use external company summaries instead of dual attention keywords
--rag_use_external_summaries

# Adjust number of retrieved documents
--rag_top_k 10

# Force rebuild RAG vector database
--force_rebuild_rag
```

### Display Configuration
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

### UI Flow Configuration
```bash
# Set UI flow type
--ui_flow_type ml    # or rag

# Enable/disable components
--enable_dual_attention
--enable_transformation_matrix
```

## 📁 Output Structure

```
FullFlow/
├── data/
│   ├── embeddings/
│   │   ├── fasttext/
│   │   └── sentence_transformer/
│   ├── clustering/
│   │   ├── models/
│   │   ├── results/
│   │   └── analysis/
│   ├── output/
│   │   ├── img/                    # Clustering plots
│   │   └── enhanced/               # Enhanced model outputs
│   └── vector_db/                  # ChromaDB storage
├── models/
│   ├── trained_models/
│   └── transformation_matrices/
├── streamlit_data/
└── pdzttb.log                      # Main log file
```

## 🔬 Example Queries for Testing

### AI and Medical Technology
```bash
python main.py --pipeline rag_only --query "machine learning algorithms for medical diagnosis and patient monitoring systems"
```

### Energy and Sustainability
```bash
python main.py --pipeline rag_only --query "renewable energy storage systems with lithium-ion battery technology"
```

### Autonomous Vehicles
```bash
python main.py --pipeline rag_only --query "autonomous vehicle navigation systems using computer vision"
```

### Quantum Computing
```bash
python main.py --pipeline rag_only --query "quantum computing applications for cryptographic security"
```

## 📊 Performance Comparison

| Method | Embedding Type | Speed | Memory | Use Case |
|--------|---------------|-------|--------|----------|
| ML-Matrix | FastText | Fast | Low | Structured queries |
| ML-Matrix | Sentence | Medium | Medium | Semantic matching |
| RAG | FastText | Medium | Medium | Natural language |
| RAG | Sentence | Slow | High | Complex queries |

## 🎯 Clustering Insights

The clustering analysis provides:

- **📈 Multi-Metric Evaluation**: Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- **🏆 Optimal Cluster Selection**: Automated ranking and selection
- **📊 Visual Analysis**: Performance plots and cluster distributions
- **🏢 Market Segmentation**: Company grouping by technology domains
- **🔍 Query-Cluster Matching**: Find nearest clusters for queries


## 🏗️ Architecture Overview

### ML Flow Architecture
```
Training:  Data → Dual Attention → Keywords → Transformation Matrix → Ready
Testing:   Query → Dual Attention → Keywords → Matrix Search → Results
```

### RAG Flow Architecture
```
Training:  Data → Dual Attention → Documents → ChromaDB → Ready
Testing:   Query → Query Embedding → ChromaDB Search → Results
```

### Clustering Integration
```
Training:  Company Embeddings → Multi-Metric Evaluation → Optimal Clusters
Testing:   Query Results + Cluster Info → Enhanced Results
```

## 🔄 Quick Start Examples

### Complete ML Pipeline (FastText)
```bash
# === TRAINING PHASE ===
# 1. Train dual attention
python main.py --pipeline dual_attn

# 2. Train transformation matrix
python main.py --pipeline patent_product --mode train --embedding_type fasttext

# 3. Build clustering
python main.py --pipeline clustering --embedding_type fasttext --enable_clustering

# === TESTING PHASE ===
# 4. Test with clustering
python main.py --pipeline patent_product --mode test --embedding_type fasttext --enable_clustering

# 5. Interactive chat
python main.py --pipeline patent_product --mode chat --embedding_type fasttext --enable_clustering
```

### Complete RAG Pipeline (Sentence Transformers)
```bash
# === TRAINING PHASE ===
# 1. Train dual attention (for document creation)
python main.py --pipeline dual_attn

# 2. Build RAG database
python main.py --pipeline patent_product --mode train --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2

# 3. Build clustering
python main.py --pipeline clustering --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --enable_clustering

# === TESTING PHASE ===
# 4. Test RAG with clustering
python main.py --pipeline patent_product --mode test --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering

# 5. Interactive RAG chat
python main.py --pipeline patent_product --mode chat --use_rag --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --rag_top_k 5 --enable_clustering

# 6. Direct RAG queries
python main.py --pipeline rag_only --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2 --query "your query here"
```

## 📋 Pipeline Reference

### Available Pipelines
- **`dual_attn`**: Train dual attention model (always first step)
- **`patent_product`**: Main pipeline (requires --mode)
  - `--mode train`: Build models/databases (no queries)
  - `--mode test`: Test with predefined queries
  - `--mode chat`: Interactive querying
- **`rag_only`**: Direct RAG querying (requires --query)
- **`clustering`**: Build clustering analysis

### Training vs Testing
- **Training Phase**: Build models, databases, clustering (no queries needed)
- **Testing Phase**: Use trained components with actual queries

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


