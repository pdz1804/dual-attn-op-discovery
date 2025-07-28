import logging
logger = logging.getLogger(__name__)

import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from ast import literal_eval
from sklearn.model_selection import train_test_split

from configs.paths import *
from configs.hyperparams import *
from models.patent2product import Patent2Product
from models.product2patent import Product2Patent
from utils.vector_utils import load_gensim_vec, text_to_vector, download_aligned_vec
from utils.vector_utils import create_embedder
from utils.model_utils import train_loop, extract_matrix, save_model_and_matrix
from utils.plot_utils import plot_train_history_trans_matrix
from inference.query_opportunity_best import query_opportunity_product_best
from inference.query_opportunity_matrix import query_opportunity_product_matrix_only

def display_unified_results(query, results, method, company_data=None, firm_patent_ids=None, patent_text_map=None, max_patents_per_company=3, clustering_analyzer=None, query_embedding=None, config=None):
    """
    Unified function to display results from both ML and RAG flows
    
    Args:
        query: The input query/patent abstract
        results: List of company results with scores
        method: String describing the method used (e.g., "RAG", "ML-Matrix", "ML-Model")
        company_data: DataFrame with company information and keywords
        firm_patent_ids: Dict mapping firm_id to list of patent IDs
        patent_text_map: Dict mapping patent_id to patent text/abstract
        max_patents_per_company: Maximum number of sample patents to show per company
        clustering_analyzer: CompanyClusteringAnalyzer for cluster information
        query_embedding: Query embedding for finding nearest cluster
        config: Configuration dictionary with display settings
    """
    print(f"\n{'='*80}")
    print(f"🔍 QUERY RESULTS")
    print(f"{'='*80}")
    print(f"📝 Query: \"{query}\"")
    print(f"🔬 Method: {method}")
    print(f"📊 Total Results: {len(results)}")
    
    # Show cluster information if available
    if clustering_analyzer is not None and query_embedding is not None:
        try:
            nearest_cluster, cluster_distance = clustering_analyzer.find_nearest_cluster(query_embedding)
            
            if nearest_cluster != -1:
                print(f"🎯 Nearest Cluster: {nearest_cluster} (distance: {cluster_distance:.4f})")
                
                # Get cluster information
                cluster_info = clustering_analyzer.get_cluster_info(nearest_cluster)
                
                if cluster_info and cluster_info.get('companies') is not None:
                    print(f"📁 Cluster {nearest_cluster} contains {cluster_info['n_companies']} companies")
                    
                    # Get top-k nearest companies in this cluster to the query
                    try:
                        # Load embeddings dictionary if available
                        embeddings_dict = None
                        if hasattr(clustering_analyzer, 'embeddings_matrix') and clustering_analyzer.embeddings_matrix is not None:
                            # Use the embeddings from the clustering analyzer
                            top_k_companies = clustering_analyzer.get_top_k_companies_in_cluster(
                                cluster_id=nearest_cluster,
                                query_embedding=query_embedding,
                                k=TOP_K_COMPANIES_IN_CLUSTER,
                                embeddings_dict=embeddings_dict  # Will use stored embeddings matrix
                            )
                            
                            if top_k_companies:
                                print(f"🏢 Top-{len(top_k_companies)} most relevant companies in Cluster {nearest_cluster}:")
                                
                                for company in top_k_companies:
                                    company_name = company.get('company_name', 'Unknown')
                                    similarity_score = company.get('similarity_score', 0.0)
                                    rank_in_cluster = company.get('rank_in_cluster', 0)
                                    keywords = str(company.get('keywords', ''))
                                    
                                    # Show first few keywords
                                    if keywords and keywords != 'nan' and keywords != '':
                                        keywords_list = keywords.split('|')[:KEYWORDS_PER_COMPANY_CLUSTER]
                                        keywords_display = ', '.join(keywords_list)
                                        if len(keywords.split('|')) > KEYWORDS_PER_COMPANY_CLUSTER:
                                            keywords_display += f" ... (+{len(keywords.split('|')) - KEYWORDS_PER_COMPANY_CLUSTER} more)"
                                    else:
                                        keywords_display = "No keywords available"
                                    
                                    print(f"  {rank_in_cluster}. {company_name} (similarity: {similarity_score:.4f})")
                                    print(f"     Keywords: {keywords_display}")
                            else:
                                # Fallback to sample companies if top-k retrieval fails
                                print(f"🏢 Sample companies in Cluster {nearest_cluster}:")
                                sample_companies = cluster_info['companies'].head(COMPANIES_PER_CLUSTER_DISPLAY)
                                
                                for idx, (_, company) in enumerate(sample_companies.iterrows()):
                                    company_name = company.get('company_name', 'Unknown')
                                    keywords = str(company.get('company_keywords', ''))
                                    
                                    # Show first few keywords
                                    if keywords and keywords != 'nan':
                                        keywords_list = keywords.split('|')[:KEYWORDS_PER_COMPANY_CLUSTER]
                                        keywords_display = ', '.join(keywords_list)
                                        if len(keywords.split('|')) > KEYWORDS_PER_COMPANY_CLUSTER:
                                            keywords_display += f" ... (+{len(keywords.split('|')) - KEYWORDS_PER_COMPANY_CLUSTER} more)"
                                    else:
                                        keywords_display = "No keywords available"
                                    
                                    print(f"  {idx+1}. {company_name}")
                                    print(f"     Keywords: {keywords_display}")
                        else:
                            # Fallback to sample companies if embeddings matrix not available
                            print(f"🏢 Sample companies in Cluster {nearest_cluster}:")
                            sample_companies = cluster_info['companies'].head(COMPANIES_PER_CLUSTER_DISPLAY)
                            
                            for idx, (_, company) in enumerate(sample_companies.iterrows()):
                                company_name = company.get('company_name', 'Unknown')
                                keywords = str(company.get('company_keywords', ''))
                                
                                # Show first few keywords
                                if keywords and keywords != 'nan':
                                    keywords_list = keywords.split('|')[:KEYWORDS_PER_COMPANY_CLUSTER]
                                    keywords_display = ', '.join(keywords_list)
                                    if len(keywords.split('|')) > KEYWORDS_PER_COMPANY_CLUSTER:
                                        keywords_display += f" ... (+{len(keywords.split('|')) - KEYWORDS_PER_COMPANY_CLUSTER} more)"
                                else:
                                    keywords_display = "No keywords available"
                                
                                print(f"  {idx+1}. {company_name}")
                                print(f"     Keywords: {keywords_display}")
                    
                    except Exception as e:
                        logger.warning(f"Could not retrieve top-k companies, falling back to samples: {e}")
                        # Fallback to sample companies
                        print(f"🏢 Sample companies in Cluster {nearest_cluster}:")
                        sample_companies = cluster_info['companies'].head(COMPANIES_PER_CLUSTER_DISPLAY)
                        
                        for idx, (_, company) in enumerate(sample_companies.iterrows()):
                            company_name = company.get('company_name', 'Unknown')
                            keywords = str(company.get('company_keywords', ''))
                            
                            # Show first few keywords
                            if keywords and keywords != 'nan':
                                keywords_list = keywords.split('|')[:KEYWORDS_PER_COMPANY_CLUSTER]
                                keywords_display = ', '.join(keywords_list)
                                if len(keywords.split('|')) > KEYWORDS_PER_COMPANY_CLUSTER:
                                    keywords_display += f" ... (+{len(keywords.split('|')) - KEYWORDS_PER_COMPANY_CLUSTER} more)"
                            else:
                                keywords_display = "No keywords available"
                            
                            print(f"  {idx+1}. {company_name}")
                            print(f"     Keywords: {keywords_display}")
            else:
                print(f"⚠️ Could not determine nearest cluster")
        except Exception as e:
            logger.warning(f"Could not determine cluster information: {e}")
    
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        company_id = str(result.get('company_id', result.get('firm_id', '')))
        company_name = result.get('company_name', result.get('firm_name', 'Unknown'))
        score = result.get('relevance_score', result.get('cosine_similarity', result.get('score', 0)))
        
        print(f"\n🏢 RANK {i+1}: {company_name}")
        print(f"   📍 Company ID: {company_id}")
        print(f"   ⭐ Relevance Score: {score:.4f}")
        
        # Show cluster membership if available
        if clustering_analyzer is not None and clustering_analyzer.cluster_assignments is not None:
            try:
                # Find which cluster this company belongs to
                if company_id in clustering_analyzer.company_ids:
                    company_idx = clustering_analyzer.company_ids.index(company_id)
                    company_cluster = clustering_analyzer.cluster_assignments[company_idx]
                    if company_cluster != -1:
                        print(f"   🎯 Cluster: {company_cluster}")
                    else:
                        print(f"   🎯 Cluster: Noise/Outlier")
            except Exception as e:
                logger.debug(f"Could not determine cluster for company {company_id}: {e}")
        
        # Show company keywords if available
        keywords = []
        if 'keywords' in result and result['keywords']:
            keywords = result['keywords']
        elif company_data is not None and company_id in company_data.index:
            # Try to get keywords from company data
            try:
                company_row = company_data.loc[company_data['hojin_id'].astype(str) == company_id]
                if not company_row.empty:
                    keywords_str = company_row.iloc[0]['company_keywords']
                    keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                    keywords = [k.strip() for k in keywords if k.strip()]
            except:
                pass
        
        if keywords:
            print(f"   🏷️  Company Keywords ({len(keywords)} total):")
            # Get max keywords display from config or use default
            max_keywords = config.get('max_keywords_display', MAX_KEYWORDS_DISPLAY) if config else MAX_KEYWORDS_DISPLAY
            displayed_keywords = keywords[:max_keywords]
            print(f"      {', '.join(displayed_keywords)}")
            if len(keywords) > max_keywords:
                print(f"      ... and {len(keywords) - max_keywords} more keywords")
        else:
            print(f"   🏷️  Company Keywords: Not available")
        
        # Show sample patents if available
        if firm_patent_ids and patent_text_map and company_id in firm_patent_ids:
            patent_ids = firm_patent_ids[company_id]
            total_patents = len(patent_ids)
            print(f"   📄 Patents: {total_patents} total")
            
            if total_patents > 0:
                sample_patents = patent_ids[:max_patents_per_company]
                print(f"   📋 Sample Patents (showing {len(sample_patents)}/{total_patents}):")
                
                for j, patent_id in enumerate(sample_patents):
                    abstract = patent_text_map.get(patent_id, "Abstract not available")
                    # Truncate abstract for display
                    abstract_preview = abstract[:150] + "..." if len(abstract) > 150 else abstract
                    print(f"      {j+1}. Patent ID: {patent_id}")
                    print(f"         Abstract: {abstract_preview}")
                
                if total_patents > max_patents_per_company:
                    print(f"      ... and {total_patents - max_patents_per_company} more patents")
        else:
            print(f"   📄 Patents: Information not available")
        
        print(f"   {'-'*60}")
    
    print(f"\n✅ Query completed successfully using {method}")
    print(f"{'='*80}\n")

def get_embedding_file_paths(embedding_type, country, model_type='linear', approx_method='sampling'):
    """Generate file paths based on embedding type and country"""
    paths = {
        'product_rep': os.path.join(DATA_DIR, f"{country}_{embedding_type}_product_rep.json"),
        'patent_rep': os.path.join(DATA_DIR, f"{country}_{embedding_type}_patent_rep.json"),
        'model_A': os.path.join(MODEL_DIR, f"{country}_{embedding_type}_{model_type}_{approx_method}_Patent2Product.pt"),
        'model_B': os.path.join(MODEL_DIR, f"{country}_{embedding_type}_{model_type}_{approx_method}_Product2Patent.pt"),
        'matrix_A': os.path.join(MODEL_DIR, f"{country}_{embedding_type}_{model_type}_{approx_method}_Patent2Product_transform.npy"),
        'matrix_B': os.path.join(MODEL_DIR, f"{country}_{embedding_type}_{model_type}_{approx_method}_Product2Patent_transform.npy"),
    }
    return paths

def save_representations_to_json(rep_dict, file_path):
    """Save a dictionary of representations to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_dict = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in rep_dict.items()}
    with open(file_path, 'w') as f:
        json.dump(serializable_dict, f)
    logger.info(f"Saved representations to {file_path}")

def load_representations_from_json(file_path):
    """Load a dictionary of representations from a JSON file."""
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} does not exist. Returning empty dict.")
        return {}
    with open(file_path, 'r') as f:
        serializable_dict = json.load(f)
    # Convert lists back to numpy arrays
    rep_dict = {key: np.array(value) if isinstance(value, list) else value for key, value in serializable_dict.items()}
    logger.info(f"Loaded representations from {file_path} with {len(rep_dict)} entries")
    return rep_dict

def initialize_embedder(config, testing=False):
    """Initialize embedder based on configuration"""
    if config.get('embedding_type', 'fasttext') == 'sentence_transformer':
        model_name = config.get('sentence_transformer_model', 'all-MiniLM-L6-v2')
        embedder = create_embedder('sentence_transformer', model_name)
        use_enhanced = True
        logger.info(f"Initialized sentence transformer: {model_name} (dim: {embedder.vector_size})")
    else:
        # Use FastText with EnhancedEmbedder wrapper for RAG compatibility
        country = config.get('country', 'US')
        if testing:
            ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else FASTTEXT_VEC_ORI
        else:
            ft_model_path = download_aligned_vec(country.lower(), DATA_DIR) if country != 'US' else FASTTEXT_VEC
        
        # Create EnhancedEmbedder wrapper for FastText to ensure RAG compatibility
        embedder = create_embedder('fasttext', ft_model_path)
        use_enhanced = True  # Now always using enhanced embedder for consistency
        logger.info(f"Initialized FastText: {ft_model_path} (dim: {embedder.vector_size})")
    
    return embedder, use_enhanced

def process_representations(country, product_df, patent_df, embedder, data_dir, use_enhanced_embedder=True):
    """Process and return patent and product representations with support for enhanced embedder."""
    patent_rep, product_rep = {}, {}
    
    # Process product representations
    for idx, row in tqdm(product_df.iterrows(), total=len(product_df), desc=f"[{country}] Processing products"):
        if use_enhanced_embedder:
            product_rep[row['Firm ID']] = embedder.encode_text(str(row['company_keywords']))
        else:
            # Fallback for legacy compatibility
            product_rep[row['Firm ID']] = text_to_vector(str(row['company_keywords']), embedder)
    logger.info(f"[{country}] Prepared product representations for {len(product_rep)} firms")

    # Process patent representations
    firm_count = 0
    for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patents"):
        firm_id = str(firm_id)
        abstracts = group['clean_abstract'].dropna().tolist()
        tokens = []
        for abstract in abstracts:
            tokens += eval(abstract) if abstract.startswith('[') else abstract.split()
        
        if use_enhanced_embedder:
            patent_rep[firm_id] = embedder.encode_text('|'.join(tokens))
        else:
            # Fallback for legacy compatibility
            patent_rep[firm_id] = text_to_vector('|'.join(tokens), embedder)
        
        # Log details for the first two firms
        if firm_count < 2:
            logger.info(f"[{country}] Processed firm {firm_id}: Tokens={tokens[:50]}... | Abstracts={len(abstracts)}")
            firm_count += 1
    
    logger.info(f"[{country}] Prepared patent representations for {len(patent_rep)} firms")
    
    return patent_rep, product_rep

def train_pipeline(config=None):
    """Unified training pipeline with configurable embedding types"""
    logger.info("[Pipeline] Start Training Patent ↔ Product")
    
    # Use provided config or default values
    if config is None:
        config = {
            'embedding_type': EMBEDDING_TYPE,
            'sentence_transformer_model': SENTENCE_TRANSFORMER_MODEL
        }
    
    logger.info(f"Training configuration: {config}")
    
    model_type = config.get('model_type', 'linear')  # default to 'linear'
    approx_method = config.get('approx_method', 'sampling')  # default to 'sampling'

    for country in COUNTRY:
        logger.info(f"[{country}] Training with {config['embedding_type']} embeddings...")
        
        # Initialize embedder based on configuration
        config['country'] = country
        embedder, use_enhanced_embedder = initialize_embedder(config)
        
        # Get embedding-specific file paths
        embedding_type = config['embedding_type']
        file_paths = get_embedding_file_paths(embedding_type, country, model_type=model_type, approx_method=approx_method)

        # Load product embeddings
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        logger.info(f"[{country}] Loaded product data with {len(product_df)} firms")

        # Load patent data
        patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else US_PATENT_DATA_CLEANED
        
        if not TEST_SIZE:
            patent_df = pd.read_csv(patent_file)
        else:
            patent_df = pd.read_csv(patent_file).sample(frac=TEST_SIZE, random_state=42)
        
        logger.info(f"[{country}] Loaded patent data with {len(patent_df)} records")

        # Process and save representations with embedding-specific names
        patent_rep, product_rep = process_representations(
            country, product_df, patent_df, embedder, DATA_DIR, use_enhanced_embedder
        )
        
        # Save with embedding-specific file names
        save_representations_to_json(patent_rep, file_paths['patent_rep'])
        save_representations_to_json(product_rep, file_paths['product_rep'])
        logger.info(f"[{country}] Saved {embedding_type} representations to embedding-specific files")
        
        shared_ids = list(set(product_rep.keys()) & set(patent_rep.keys()))
        if len(shared_ids) == 0:
            logger.info(f"[{country}] No shared IDs. Skipping.")
            continue

        # Prepare data
        X_patent = np.stack([patent_rep[i] for i in shared_ids])
        Y_product = np.stack([product_rep[i] for i in shared_ids])
        logger.info(f"[{country}] Prepared training data with {len(X_patent)} shared firms")
        
        if len(X_patent) == 0 or len(Y_product) == 0:
            logger.info(f"[{country}] No data to train. Skipping.")
            continue

        # Get actual embedding dimension from data
        actual_dim = X_patent.shape[1]
        logger.info(f"[{country}] Using embedding dimension: {actual_dim}")

        # Train Patent → Product (Model A) with correct dimension
        # old 
        # model_A = Patent2Product(dim=actual_dim)
        
        # new  
        from models.registry import MODEL_REGISTRY
        model_A = MODEL_REGISTRY[model_type]['Patent2Product'](dim=actual_dim)
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_patent, Y_product, test_size=0.2, random_state=42)
        logger.info(f"[{country}] Training Patent2Product model with {len(X_train)} training samples")
        logger.info(f"[{country}] Validation size: {len(X_val)} samples")
        
        model_A, hist_A = train_loop(model_A, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32),
                                     torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32),
                                     EPOCHS_TRANS_MATRIX)
        logger.info(f"[{country}] Finished training Patent2Product model")
        
        # Plot training history for Model A with embedding-specific name
        plot_train_history_trans_matrix(hist_A, f"{embedding_type}_{model_type}_{approx_method}_Patent2Product", country)

        # Save model and extract matrix with embedding-specific names
        A_matrix = extract_matrix(model_A, model_type=model_type, approx_method=approx_method)
        save_model_and_matrix(model_A, A_matrix, file_paths['model_A'])
        np.save(file_paths['matrix_A'], A_matrix)

        # Train Product → Patent (Model B) with correct dimension
        # old 
        # model_B = Product2Patent(dim=actual_dim)
        
        # new 
        from models.registry import MODEL_REGISTRY
        model_B = MODEL_REGISTRY[model_type]['Product2Patent'](dim=actual_dim)
        
        X_train, X_val, Y_train, Y_val = train_test_split(Y_product, X_patent, test_size=0.2, random_state=42)
        logger.info(f"[{country}] Training Product2Patent model with {len(X_train)} training samples")
        logger.info(f"[{country}] Validation size: {len(X_val)} samples")
        
        model_B, hist_B = train_loop(model_B, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32),
                                     torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32),
                                     EPOCHS_TRANS_MATRIX)
        logger.info(f"[{country}] Finished training Product2Patent model")
        
        # Plot training history for Model B with embedding-specific name
        plot_train_history_trans_matrix(hist_B, f"{embedding_type}_{model_type}_{approx_method}_Product2Patent", country)

        # Save model and extract matrix with embedding-specific names
        B_matrix = extract_matrix(model_B, model_type=model_type, approx_method=approx_method)
        save_model_and_matrix(model_B, B_matrix, file_paths['model_B'])
        np.save(file_paths['matrix_B'], B_matrix)

        # Export product embeddings with embedding-specific name
        product_embeddings_file = os.path.join(DATA_DIR, f"{country}_{embedding_type}_{model_type}_{approx_method}_Firm_Product_Keywords_Table_with_Vector.csv")
        product_df['Product embedding vector'] = product_df['Firm ID'].apply(
            lambda fid: '|'.join(map(str, model_B(torch.tensor(patent_rep[fid], dtype=torch.float32)).detach().numpy())) if fid in patent_rep else ""
        )

        product_df.to_csv(product_embeddings_file, index=False)
        logger.info(f"[{country}] Exported {embedding_type} product embeddings to {product_embeddings_file}")
        
        logger.info(f"[{country}] Training completed for {embedding_type} embeddings")
        logger.info(f"[{country}] Saved files:")
        logger.info(f"  - Models: {file_paths['model_A']}, {file_paths['model_B']}")
        logger.info(f"  - Matrices: {file_paths['matrix_A']}, {file_paths['matrix_B']}")
        logger.info(f"  - Representations: {file_paths['patent_rep']}, {file_paths['product_rep']}")
        logger.info(f"  - Product embeddings: {product_embeddings_file}")
        
    logger.info(f"[Pipeline] Training completed for all countries with {config['embedding_type']} embeddings")

def test_pipeline(config=None):
    """Unified test pipeline with configurable embedding types and RAG/Matrix choice"""
    logger.info("[Pipeline] Start Testing Patent ↔ Product")
    
    # Use provided config or default values
    if config is None:
        config = {
            'embedding_type': EMBEDDING_TYPE,
            'sentence_transformer_model': SENTENCE_TRANSFORMER_MODEL,
            'use_rag': USE_RAG,
            'rag_use_external_summaries': RAG_USE_EXTERNAL_SUMMARIES,
            'rag_top_k': RAG_TOP_K
        }
    
    logger.info(f"Test configuration: {config}")
    
    if config.get('use_rag', False):
        # RAG approach
        logger.info("Using RAG approach for testing")
        from utils.rag_utils import create_rag_processor
        
        # Initialize embedder
        embedder, _ = initialize_embedder(config, testing=True)
        
        # Create RAG processor once
        logger.info("Initializing RAG processor...")
        rag_processor = create_rag_processor(embedder)
        
        # Load company data for displaying keywords
        company_df = None
        if os.path.exists(EMBEDDINGS_OUTPUT):
            company_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        
        # Load clustering analyzer if available
        clustering_analyzer = None
        if ENABLE_CLUSTERING:
            try:
                from pipelines.clustering_pipeline import load_clustering_analyzer
                # Use the same country parameter as in config (default to 'US' if not specified)
                country = config.get('countries', ['US'])[0] if config.get('countries') else 'US'
                clustering_analyzer = load_clustering_analyzer(config.get('embedding_type', 'fasttext'), country)
                if clustering_analyzer:
                    logger.info(f"Loaded clustering analyzer for enhanced results display (country: {country})")
            except Exception as e:
                logger.warning(f"Could not load clustering analyzer: {e}")
        
        # Run RAG tests
        test_queries = [
            "machine learning algorithms",
            "renewable energy systems", 
            "medical devices diagnostics"
        ]
        
        for query in test_queries:
            logger.info(f"Testing RAG query: '{query}'")
            # Use the existing RAG processor instead of creating a new one
            results = rag_processor.rag_query_companies(query, config.get('rag_top_k', 5))
            
            # Get query embedding for cluster analysis
            query_embedding = None
            if clustering_analyzer is not None:
                try:
                    # Always use encode_text since we now use EnhancedEmbedder consistently
                    query_embedding = embedder.encode_text(query)
                except Exception as e:
                    logger.warning(f"Could not compute query embedding: {e}")
            
            # Display results using unified format
            display_unified_results(
                query=query,
                results=results,
                method="RAG (ChromaDB)",
                company_data=company_df,
                firm_patent_ids=None,  # RAG doesn't have patent mapping yet
                patent_text_map=None,
                max_patents_per_company=3,
                clustering_analyzer=clustering_analyzer,
                query_embedding=query_embedding,
                config=config
            )
    else:
        # Traditional ML approach
        for country in COUNTRY:
            logger.info(f"[{country}] Testing with {config['embedding_type']} embeddings...")

            # Initialize embedder
            config['country'] = country
            embedder, use_enhanced_embedder = initialize_embedder(config, testing=True)
            
            # Get embedding-specific file paths
            embedding_type = config['embedding_type']
            file_paths = get_embedding_file_paths(embedding_type, country, model_type=config.get('model_type', 'linear'), approx_method=config.get('approx_method', 'sampling'))

            # Load product data
            product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
            product_df['Firm ID'] = product_df['hojin_id'].astype(str)
            
            data_path = US_WEB_DATA
            us_web_with_company = pd.read_csv(data_path)
            
            # Keep only necessary columns from `us_web_with_company`
            company_name_map = us_web_with_company[['hojin_id', 'company_name']].drop_duplicates()

            # Merge based on hojin_id
            product_df = product_df.merge(company_name_map, on='hojin_id', how='left')
            
            # Load representations from embedding-specific files
            product_rep = load_representations_from_json(file_paths['product_rep'])
            if not product_rep:
                logger.warning(f"[{country}] No {embedding_type} product representations loaded. Computing from scratch.")
                product_rep = {}
                for idx, row in tqdm(product_df.iterrows(), total=len(product_df), desc=f"[{country}] Processing products"):
                    if use_enhanced_embedder:
                        product_rep[row['Firm ID']] = embedder.encode_text(str(row['company_keywords']))
                    else:
                        product_rep[row['Firm ID']] = text_to_vector(str(row['company_keywords']), embedder)
                
            # Load patent data
            patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else US_PATENT_DATA_CLEANED
            patent_df = pd.read_csv(patent_file)

            # Load patent representations from embedding-specific files
            patent_rep = load_representations_from_json(file_paths['patent_rep'])
            firm_patent_ids, patent_text_map = {}, {}
            
            # Always compute firm_patent_ids and patent_text_map for display
            for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patent IDs"):
                firm_id = str(firm_id)
                firm_patent_ids[firm_id] = group['appln_id'].tolist()
                for app_id, abs_text in zip(group['appln_id'], group['clean_abstract'].dropna()):
                    patent_text_map[app_id] = abs_text

            # Compute patent representations if not loaded
            if not patent_rep:
                logger.warning(f"[{country}] No {embedding_type} patent representations loaded. Computing from scratch.")
                patent_rep = {}
                for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patents"):
                    firm_id = str(firm_id)
                    abstracts = group['clean_abstract'].dropna().tolist()
                    tokens = []
                    for abstract in abstracts:
                        tokens += literal_eval(abstract) if abstract.startswith('[') else abstract.split()
                    
                    if use_enhanced_embedder:
                        patent_rep[firm_id] = embedder.encode_text('|'.join(tokens))
                    else:
                        patent_rep[firm_id] = text_to_vector('|'.join(tokens), embedder)

            logger.info(f"[{country}] Prepared {len(firm_patent_ids)} firm patent IDs and {len(patent_text_map)} patent text mappings")
            
            # Load clustering analyzer if available
            clustering_analyzer = None
            if ENABLE_CLUSTERING:
                try:
                    from pipelines.clustering_pipeline import load_clustering_analyzer
                    clustering_analyzer = load_clustering_analyzer(config.get('embedding_type', 'fasttext'), country)
                    if clustering_analyzer:
                        logger.info(f"[{country}] Loaded clustering analyzer for enhanced results display")
                except Exception as e:
                    logger.warning(f"[{country}] Could not load clustering analyzer: {e}")
            
            # Get embedding dimension and load models with correct dimension
            embedding_dim = 300  # Default for FastText
            if patent_rep:
                sample_embedding = next(iter(patent_rep.values()))
                embedding_dim = len(sample_embedding)
            logger.info(f"[{country}] Using embedding dimension: {embedding_dim}")
            
            # Check if embedding-specific model files exist
            if not os.path.exists(file_paths['matrix_A']) or not os.path.exists(file_paths['matrix_B']):
                logger.error(f"[{country}] Missing {embedding_type} model files. Please train the models first:")
                logger.error(f"  python main.py --pipeline patent_product --mode train --embedding_type {embedding_type}")
                continue
            
            # Load matrices with embedding-specific names
            A_matrix = np.load(file_paths['matrix_A'])
            B_matrix = np.load(file_paths['matrix_B'])

            # Load models for nonlinear query with correct dimension
            # old 
            # model_A = Patent2Product(dim=embedding_dim)
            # model_B = Product2Patent(dim=embedding_dim)
            
            # new 
            from models.registry import MODEL_REGISTRY
            model_type = config.get('model_type', 'linear')
            model_A = MODEL_REGISTRY[model_type]['Patent2Product'](dim=embedding_dim)
            model_B = MODEL_REGISTRY[model_type]['Product2Patent'](dim=embedding_dim)
            
            if os.path.exists(file_paths['model_A']) and os.path.exists(file_paths['model_B']):
                model_A.load_state_dict(torch.load(file_paths['model_A']))
                model_B.load_state_dict(torch.load(file_paths['model_B']))
                model_A.eval()
                model_B.eval()
            else:
                logger.warning(f"[{country}] Model files not found for {embedding_type}. Only matrix queries will work.")

            # Test queries
            test_queries = [
                "chemical waste machine",
                "computer vision and machine learning"
            ]

            for query in test_queries:
                # Get query embedding for cluster analysis
                query_embedding = None
                if clustering_analyzer is not None:
                    try:
                        if use_enhanced_embedder:
                            query_embedding = embedder.encode_text(query)
                        else:
                            # FastText embedder
                            query_tokens = query.strip().split()
                            token_vecs = [embedder[w] for w in query_tokens if w in embedder]
                            if token_vecs:
                                query_embedding = np.mean(token_vecs, axis=0)
                    except Exception as e:
                        logger.warning(f"[{country}] Could not compute query embedding: {e}")
                
                # Run Matrix Query
                logger.info(f"[{country}] Running matrix query for '{query}'")
                results_matrix = query_opportunity_product_matrix_only(
                    product_query_text=query,
                    ft_model=embedder,
                    mat_B=B_matrix,
                    mat_A=A_matrix,
                    patent_rep_dict=patent_rep,
                    product_rep_dict=product_rep,
                    firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
                    firm_patent_ids=firm_patent_ids,
                    patent_text_map=patent_text_map,
                    top_k=5
                )
                
                # Display results using unified format
                if results_matrix:
                    # Add company keywords to results
                    for result in results_matrix:
                        firm_id = str(result.get('firm_id', ''))
                        # Get keywords from product_df
                        company_row = product_df[product_df['Firm ID'] == firm_id]
                        if not company_row.empty and 'company_keywords' in company_row.columns:
                            keywords_str = company_row.iloc[0]['company_keywords']
                            if pd.notna(keywords_str):
                                keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                result['keywords'] = [k.strip() for k in keywords if k.strip()]
                            else:
                                result['keywords'] = []
                        else:
                            result['keywords'] = []
                    
                    display_unified_results(
                        query=query,
                        results=results_matrix,
                        method=f"ML-Matrix ({embedding_type})",
                        company_data=product_df,
                        firm_patent_ids=firm_patent_ids,
                        patent_text_map=patent_text_map,
                        max_patents_per_company=3,
                        clustering_analyzer=clustering_analyzer,
                        query_embedding=query_embedding,
                        config=config
                    )

                # Run Model Query (if models are available)
                if os.path.exists(file_paths['model_A']) and os.path.exists(file_paths['model_B']):
                    logger.info(f"[{country}] Running model query for '{query}'")
                    results_best = query_opportunity_product_best(
                        product_query_text=query,
                        ft_model=embedder,
                        model_B=model_B,
                        model_A=model_A,
                        patent_rep_dict=patent_rep,
                        product_rep_dict=product_rep,
                        firm_id_name_map=dict(zip(product_df['Firm ID'], product_df['company_name'])),
                        firm_patent_ids=firm_patent_ids,
                        patent_text_map=patent_text_map,
                        top_k=5
                    )
                    
                    # Display results using unified format
                    if results_best:
                        # Add company keywords to results
                        for result in results_best:
                            firm_id = str(result.get('firm_id', ''))
                            # Get keywords from product_df
                            company_row = product_df[product_df['Firm ID'] == firm_id]
                            if not company_row.empty and 'company_keywords' in company_row.columns:
                                keywords_str = company_row.iloc[0]['company_keywords']
                                if pd.notna(keywords_str):
                                    keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                    result['keywords'] = [k.strip() for k in keywords if k.strip()]
                                else:
                                    result['keywords'] = []
                            else:
                                result['keywords'] = []
                        
                        display_unified_results(
                            query=query,
                            results=results_best,
                            method=f"ML-Model ({embedding_type})",
                            company_data=product_df,
                            firm_patent_ids=firm_patent_ids,
                            patent_text_map=patent_text_map,
                            max_patents_per_company=3,
                            clustering_analyzer=clustering_analyzer,
                            query_embedding=query_embedding,
                            config=config
                        )

def chat_pipeline(config=None):
    """Unified chat pipeline with configurable embedding types and RAG/Matrix choice"""
    logger.info("[Pipeline] Start Interactive Testing (Chat Mode)")
    
    # Use provided config or default values
    if config is None:
        config = {
            'embedding_type': EMBEDDING_TYPE,
            'sentence_transformer_model': SENTENCE_TRANSFORMER_MODEL,
            'use_rag': USE_RAG,
            'rag_use_external_summaries': RAG_USE_EXTERNAL_SUMMARIES,
            'rag_top_k': RAG_TOP_K,
            'approx_method': 'sampling',
            'model_type': 'linear'
        }
    
    logger.info(f"Chat configuration: {config}")

    if config.get('use_rag', False):
        # RAG chat mode
        logger.info("Using RAG approach for interactive chat")
        from utils.rag_utils import create_rag_processor
        
        # Initialize embedder
        embedder, _ = initialize_embedder(config, testing=True)
        
        # Create RAG processor once
        logger.info("Initializing RAG processor for chat...")
        rag_processor = create_rag_processor(embedder)
        
        # Load company data for displaying keywords
        company_df = None
        if os.path.exists(EMBEDDINGS_OUTPUT):
            company_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        
        # Load clustering analyzer if available
        clustering_analyzer = None
        if ENABLE_CLUSTERING:
            try:
                from pipelines.clustering_pipeline import load_clustering_analyzer
                # Use the same country parameter as in config (default to 'US' if not specified)
                country = config.get('countries', ['US'])[0] if config.get('countries') else 'US'
                clustering_analyzer = load_clustering_analyzer(config.get('embedding_type', 'fasttext'), country)
                if clustering_analyzer:
                    logger.info(f"Loaded clustering analyzer for enhanced results display (country: {country})")
            except Exception as e:
                logger.warning(f"Could not load clustering analyzer: {e}")
        
        # Start RAG Chat Loop
        print("\n=== RAG Patent ↔ Product Search Chat ===")
        print(f"Embedding Type: {config.get('embedding_type', 'fasttext')}")
        print(f"RAG Mode: {'External Summaries' if config.get('rag_use_external_summaries', False) else 'Dual Attention Keywords'}")
        if clustering_analyzer:
            print(f"🎯 Clustering: Enabled ({clustering_analyzer.best_n_clusters} clusters)")
        print("Type 'exit' to quit.")
        print("You can enter product queries or patent abstracts.\n")

        while True:
            user_input = input("Enter product query or patent abstract: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break

            top_k = config.get('rag_top_k', 5)
            logger.info(f"Processing RAG query: '{user_input[:50]}...'")
            
            # Use the existing RAG processor instead of creating a new one
            results = rag_processor.rag_query_companies(user_input, top_k)
            
            # Get query embedding for cluster analysis
            query_embedding = None
            if clustering_analyzer is not None:
                try:
                    # Always use encode_text since we now use EnhancedEmbedder consistently
                    query_embedding = embedder.encode_text(user_input)
                except Exception as e:
                    logger.warning(f"Could not compute query embedding: {e}")
            
            # Display results using unified format
            display_unified_results(
                query=user_input,
                results=results,
                method="RAG (ChromaDB)",
                company_data=company_df,
                firm_patent_ids=None,  # RAG doesn't have patent mapping yet
                patent_text_map=None,
                max_patents_per_company=3,
                clustering_analyzer=clustering_analyzer,
                query_embedding=query_embedding,
                config=config
            )
    else:
        # Traditional ML chat mode
        for country in COUNTRY:
            logger.info(f"[{country}] Setting up interactive test for {country} with {config.get('embedding_type', 'fasttext')} embeddings...")

            # Initialize embedder
            config['country'] = country
            embedder, use_enhanced_embedder = initialize_embedder(config, testing=True)
            
            # Get embedding-specific file paths
            embedding_type = config['embedding_type']
            file_paths = get_embedding_file_paths(embedding_type, country, model_type=config.get('model_type', 'linear'), approx_method=config.get('approx_method', 'sampling'))

            # Load data (same as test_pipeline)
            product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
            product_df['Firm ID'] = product_df['hojin_id'].astype(str)
            
            data_path = US_WEB_DATA
            us_web_with_company = pd.read_csv(data_path)
            
            # Keep only necessary columns from `us_web_with_company`
            company_name_map = us_web_with_company[['hojin_id', 'company_name']].drop_duplicates()

            # Merge based on hojin_id
            product_df = product_df.merge(company_name_map, on='hojin_id', how='left')
            
            # Load representations from embedding-specific files
            product_rep = load_representations_from_json(file_paths['product_rep'])
            if not product_rep:
                logger.warning(f"[{country}] No {embedding_type} product representations loaded. Computing from scratch.")
                product_rep = {}
                for idx, row in tqdm(product_df.iterrows(), total=len(product_df), desc=f"[{country}] Processing products"):
                    if use_enhanced_embedder:
                        product_rep[row['Firm ID']] = embedder.encode_text(str(row['company_keywords']))
                    else:
                        product_rep[row['Firm ID']] = text_to_vector(str(row['company_keywords']), embedder)

            # Load patent data
            patent_file = f'./CN_JP_US_data/{country.lower()}_patent.csv' if country != 'US' else US_PATENT_DATA_CLEANED
            patent_df = pd.read_csv(patent_file)

            # Load patent representations from embedding-specific files
            patent_rep = load_representations_from_json(file_paths['patent_rep'])
            firm_patent_ids, patent_text_map = {}, {}
            
            # Always compute firm_patent_ids and patent_text_map for display
            for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patent IDs"):
                firm_id = str(firm_id)
                firm_patent_ids[firm_id] = group['appln_id'].tolist()
                for app_id, abs_text in zip(group['appln_id'], group['clean_abstract'].dropna()):
                    patent_text_map[app_id] = abs_text

            # Compute patent representations if not loaded
            if not patent_rep:
                logger.warning(f"[{country}] No {embedding_type} patent representations loaded. Computing from scratch.")
                patent_rep = {}
                for firm_id, group in tqdm(patent_df.groupby('hojin_id'), desc=f"[{country}] Processing patents"):
                    firm_id = str(firm_id)
                    abstracts = group['clean_abstract'].dropna().tolist()
                    tokens = []
                    for abstract in abstracts:
                        tokens += literal_eval(abstract) if abstract.startswith('[') else abstract.split()
                    
                    if use_enhanced_embedder:
                        patent_rep[firm_id] = embedder.encode_text('|'.join(tokens))
                    else:
                        patent_rep[firm_id] = text_to_vector('|'.join(tokens), embedder)

            logger.info(f"[{country}] Prepared {len(firm_patent_ids)} firm patent IDs and {len(patent_text_map)} patent text mappings")

            # Load clustering analyzer if available
            clustering_analyzer = None
            if ENABLE_CLUSTERING:
                try:
                    from pipelines.clustering_pipeline import load_clustering_analyzer
                    clustering_analyzer = load_clustering_analyzer(config.get('embedding_type', 'fasttext'), country)
                    if clustering_analyzer:
                        logger.info(f"[{country}] Loaded clustering analyzer for enhanced results display")
                except Exception as e:
                    logger.warning(f"[{country}] Could not load clustering analyzer: {e}")

            # Get embedding dimension and load models with correct dimension
            embedding_dim = 300  # Default for FastText
            if patent_rep:
                sample_embedding = next(iter(patent_rep.values()))
                embedding_dim = len(sample_embedding)
            logger.info(f"[{country}] Using embedding dimension: {embedding_dim}")
            
            # Check if embedding-specific model files exist
            if not os.path.exists(file_paths['matrix_A']) or not os.path.exists(file_paths['matrix_B']):
                logger.error(f"[{country}] Missing {embedding_type} model files. Please train the models first:")
                logger.error(f"  python main.py --pipeline patent_product --mode train --embedding_type {embedding_type}")
                continue

            # Load matrices with embedding-specific names
            A_matrix = np.load(file_paths['matrix_A'])
            B_matrix = np.load(file_paths['matrix_B'])

            # Load models for nonlinear retrieval with correct dimension
            model_A = Patent2Product(dim=embedding_dim)
            model_B = Product2Patent(dim=embedding_dim)
            
            models_available = False
            if os.path.exists(file_paths['model_A']) and os.path.exists(file_paths['model_B']):
                model_A.load_state_dict(torch.load(file_paths['model_A']))
                model_B.load_state_dict(torch.load(file_paths['model_B']))
                model_A.eval()
                model_B.eval()
                models_available = True
            else:
                logger.warning(f"[{country}] Model files not found for {embedding_type}. Only matrix queries will work.")

            # Start Chat Loop
            print(f"\n=== Patent ↔ Product Search Chat ({country}) ===")
            print(f"Embedding Type: {config.get('embedding_type', 'fasttext')}")
            if clustering_analyzer:
                print(f"🎯 Clustering: Enabled ({clustering_analyzer.best_n_clusters} clusters)")
            print("Type 'exit' to quit.")
            if models_available:
                print("Choose mode: 'matrix' (linear) or 'model' (nonlinear)")
            else:
                print("Only 'matrix' mode available (model files not found)")
            print("You can enter product queries or patent abstracts.\n")

            while True:
                user_input = input("Enter product query or patent abstract: ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Exiting chat.")
                    break

                if models_available:
                    mode_input = input("Choose mode (matrix/model): ").strip().lower()
                    if mode_input not in ["matrix", "model"]:
                        print("Invalid mode. Try again.")
                        continue
                else:
                    mode_input = "matrix"

                top_k = 5
                logger.info(f"[{country}] Processing query: '{user_input[:50]}...' in {mode_input} mode")
                
                # Get query embedding for cluster analysis
                query_embedding = None
                if clustering_analyzer is not None:
                    try:
                        if use_enhanced_embedder:
                            query_embedding = embedder.encode_text(user_input)
                        else:
                            # FastText embedder
                            query_tokens = user_input.strip().split()
                            token_vecs = [embedder[w] for w in query_tokens if w in embedder]
                            if token_vecs:
                                query_embedding = np.mean(token_vecs, axis=0)
                    except Exception as e:
                        logger.warning(f"[{country}] Could not compute query embedding: {e}")
                
                if mode_input == "matrix":
                    logger.info(f"[{country}] Running matrix query")
                    results_matrix = query_opportunity_product_matrix_only(
                        product_query_text=user_input.lower(),
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
                    
                    # Add company keywords to results
                    if results_matrix:
                        for result in results_matrix:
                            firm_id = str(result.get('firm_id', ''))
                            # Get keywords from product_df
                            company_row = product_df[product_df['Firm ID'] == firm_id]
                            if not company_row.empty and 'company_keywords' in company_row.columns:
                                keywords_str = company_row.iloc[0]['company_keywords']
                                if pd.notna(keywords_str):
                                    keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                    result['keywords'] = [k.strip() for k in keywords if k.strip()]
                                else:
                                    result['keywords'] = []
                            else:
                                result['keywords'] = []
                        
                        display_unified_results(
                            query=user_input,
                            results=results_matrix,
                            method=f"ML-Matrix ({embedding_type})",
                            company_data=product_df,
                            firm_patent_ids=firm_patent_ids,
                            patent_text_map=patent_text_map,
                            max_patents_per_company=3,
                            clustering_analyzer=clustering_analyzer,
                            query_embedding=query_embedding,
                            config=config
                        )

                elif mode_input == "model":
                    logger.info(f"[{country}] Running model query")
                    results_best = query_opportunity_product_best(
                        product_query_text=user_input.lower(),
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
                    
                    # Add company keywords to results
                    if results_best:
                        for result in results_best:
                            firm_id = str(result.get('firm_id', ''))
                            # Get keywords from product_df
                            company_row = product_df[product_df['Firm ID'] == firm_id]
                            if not company_row.empty and 'company_keywords' in company_row.columns:
                                keywords_str = company_row.iloc[0]['company_keywords']
                                if pd.notna(keywords_str):
                                    keywords = keywords_str.split('|') if '|' in str(keywords_str) else [str(keywords_str)]
                                    result['keywords'] = [k.strip() for k in keywords if k.strip()]
                                else:
                                    result['keywords'] = []
                            else:
                                result['keywords'] = []
                        
                        display_unified_results(
                            query=user_input,
                            results=results_best,
                            method=f"ML-Model ({embedding_type})",
                            company_data=product_df,
                            firm_patent_ids=firm_patent_ids,
                            patent_text_map=patent_text_map,
                            max_patents_per_company=3,
                            clustering_analyzer=clustering_analyzer,
                            query_embedding=query_embedding,
                            config=config
                        )

def rag_only_pipeline(config=None):
    """RAG-only pipeline for direct querying with unified output"""
    if not config or not config.get('query'):
        raise ValueError("Error: query is required for rag_only pipeline.")
    
    logger.info(f"[RAG Pipeline] Running with query: '{config['query'][:50]}...'")
    logger.info(f"Configuration: {config}")
    
    # Initialize embedder
    embedder, _ = initialize_embedder(config)
    
    # Create RAG processor
    from utils.rag_utils import create_rag_processor
    rag_processor = create_rag_processor(embedder)
    
    # Load company data for displaying keywords
    company_df = None
    if os.path.exists(EMBEDDINGS_OUTPUT):
        company_df = pd.read_csv(EMBEDDINGS_OUTPUT)
    
    # Run RAG query
    results = rag_processor.rag_query_companies(config['query'], config.get('rag_top_k', 5))
    
    # Display results using unified format
    display_unified_results(
        query=config['query'],
        results=results,
        method="RAG (ChromaDB)",
        company_data=company_df,
        firm_patent_ids=None,  # RAG doesn't have patent mapping yet
        patent_text_map=None,
        max_patents_per_company=3
    )

# Legacy function names for backward compatibility (these just call the unified functions)
def train_pipeline_enhanced(config):
    """Legacy function - calls unified train_pipeline"""
    return train_pipeline(config)

def test_pipeline_enhanced(config):
    """Legacy function - calls unified test_pipeline"""
    return test_pipeline(config)

def test_pipeline_chat_enhanced(config):
    """Legacy function - calls unified chat_pipeline"""
    return chat_pipeline(config)




