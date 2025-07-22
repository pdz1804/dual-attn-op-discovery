"""
Clustering pipeline for company embeddings analysis
"""

import logging
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from configs.paths import *
from configs.hyperparams import *
from utils.clustering_utils import run_clustering_analysis, CompanyClusteringAnalyzer
from utils.vector_utils import create_embedder
from pipelines.patent_product_pipeline import load_representations_from_json, initialize_embedder

logger = logging.getLogger(__name__)

def clustering_pipeline(config=None):
    """
    Main clustering pipeline for company embeddings analysis
    
    Args:
        config: Configuration dictionary containing:
            - embedding_type: 'fasttext' or 'sentence_transformer'
            - enable_clustering: bool to enable/disable clustering
            - force_rebuild_clustering: bool to force rebuild existing clustering
    """
    logger.info("[Pipeline] Start Company Embeddings Clustering Analysis")
    
    # Use provided config or default values
    if config is None:
        config = {
            'embedding_type': EMBEDDING_TYPE,
            'sentence_transformer_model': SENTENCE_TRANSFORMER_MODEL,
            'enable_clustering': ENABLE_CLUSTERING,
            'force_rebuild_clustering': False
        }
    
    # Check if clustering is enabled
    if not config.get('enable_clustering', ENABLE_CLUSTERING):
        logger.info("Clustering is disabled in configuration. Skipping clustering pipeline.")
        return
    
    logger.info(f"Clustering configuration: {config}")
    
    for country in COUNTRY:
        logger.info(f"[{country}] Starting clustering analysis for {config['embedding_type']} embeddings...")
        
        # Initialize embedder
        config['country'] = country
        embedder, use_enhanced_embedder = initialize_embedder(config)
        
        # Get embedding-specific file paths for loading representations
        from pipelines.patent_product_pipeline import get_embedding_file_paths
        embedding_type = config['embedding_type']
        file_paths = get_embedding_file_paths(embedding_type, country)
        
        # Load company data
        product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        product_df['Firm ID'] = product_df['hojin_id'].astype(str)
        
        # Load company names
        data_path = US_WEB_DATA
        us_web_with_company = pd.read_csv(data_path)
        company_name_map = us_web_with_company[['hojin_id', 'company_name']].drop_duplicates()
        product_df = product_df.merge(company_name_map, on='hojin_id', how='left')
        
        logger.info(f"[{country}] Loaded company data with {len(product_df)} companies")
        
        # Load company embeddings (product representations)
        product_rep = load_representations_from_json(file_paths['product_rep'])
        
        if not product_rep:
            logger.warning(f"[{country}] No {embedding_type} company representations found. Please run training first:")
            logger.warning(f"  python main.py --pipeline patent_product --mode train --embedding_type {embedding_type}")
            continue
        
        logger.info(f"[{country}] Loaded {len(product_rep)} company embeddings")
        
        # Run clustering analysis
        try:
            clustering_analyzer = run_clustering_analysis(
                embeddings_dict=product_rep,
                company_df=product_df,
                embedding_type=embedding_type,
                country=country,
                force_rebuild=config.get('force_rebuild_clustering', False)
            )
            
            # Display clustering results summary
            display_clustering_summary(clustering_analyzer, country, embedding_type)
            
            logger.info(f"[{country}] Clustering analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"[{country}] Clustering analysis failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    logger.info("[Pipeline] Clustering analysis completed for all countries")

def display_clustering_summary(analyzer: CompanyClusteringAnalyzer, country: str, embedding_type: str):
    """Display a summary of clustering results"""
    
    print(f"\n{'='*80}")
    print(f"🎯 CLUSTERING ANALYSIS SUMMARY ({country} - {embedding_type})")
    print(f"{'='*80}")
    
    if analyzer.best_model is None:
        print("❌ No successful clustering found")
        return
    
    print(f"📊 **Multi-Metric Ranking Results:**")
    print(f"   🔢 Best Number of Clusters: {analyzer.best_n_clusters}")
    print(f"   🏆 Total Rank Score: {analyzer.best_score:.0f} (lower is better)")
    print(f"   🏢 Total Companies: {len(analyzer.company_ids)}")
    print(f"   📏 Embedding Dimension: {analyzer.embeddings_matrix.shape[1]}")
    
    # Try to load ranking results for detailed display
    try:
        import json
        ranking_file = analyzer.file_paths.get('ranking_results')
        if ranking_file and os.path.exists(ranking_file):
            with open(ranking_file, 'r') as f:
                ranking_results = json.load(f)
            
            best_config = ranking_results['best_configuration']
            print(f"\n📈 **Individual Metric Performance:**")
            print(f"   🔵 Silhouette Score: {best_config['silhouette_score']:.4f} (Rank: {best_config['silhouette_rank']:.0f})")
            print(f"   🟢 Calinski-Harabasz: {best_config['calinski_score']:.2f} (Rank: {best_config['calinski_rank']:.0f})")
            print(f"   🟡 Davies-Bouldin: {best_config['davies_score']:.4f} (Rank: {best_config['davies_rank']:.0f})")
            
            summary = ranking_results['summary']
            print(f"\n📋 **Ranking Summary:**")
            print(f"   🔬 Configurations Tested: {summary['total_configurations_tested']}")
            print(f"   🏅 Tied Configurations: {summary['num_tied_configurations']}")
            print(f"   🎯 Tie-breaking Rule: {summary['tie_breaking_rule'].replace('_', ' ').title()}")
    except:
        pass
    
    # Show cluster size distribution
    print(f"\n📋 **Cluster Size Distribution:**")
    unique_clusters, cluster_counts = np.unique(analyzer.cluster_assignments, return_counts=True)
    
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        if cluster_id == -1:
            print(f"   🔴 Noise Points: {count} companies")
        else:
            percentage = (count / len(analyzer.company_ids)) * 100
            print(f"   🟢 Cluster {cluster_id}: {count} companies ({percentage:.1f}%)")
    
    # Show sample companies from largest clusters
    print(f"\n🏢 **Sample Companies by Cluster:**")
    
    # Get top 3 largest clusters (excluding noise)
    valid_clusters = [(cid, count) for cid, count in zip(unique_clusters, cluster_counts) if cid != -1]
    valid_clusters.sort(key=lambda x: x[1], reverse=True)
    
    for cluster_id, count in valid_clusters[:3]:
        cluster_info = analyzer.get_cluster_info(cluster_id)
        
        print(f"\n   📁 **Cluster {cluster_id}** ({count} companies):")
        
        # Show top 3 companies in this cluster
        cluster_companies = cluster_info['companies'].head(3)
        
        for idx, (_, company) in enumerate(cluster_companies.iterrows()):
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
            
            print(f"     {idx+1}. {company_name}")
            print(f"        Keywords: {keywords_display}")
    
    # Show information about saved plots
    print(f"\n📊 **Saved Visualizations:**")
    performance_plot = analyzer.file_paths.get('performance_plot', 'N/A')
    if performance_plot != 'N/A':
        print(f"   📈 Performance Plots: {performance_plot}")
        ranking_table = performance_plot.replace('.png', '_ranking_table.png')
        print(f"   📋 Ranking Table: {ranking_table}")
        cluster_viz = analyzer.file_paths.get('visualization', 'N/A')
        print(f"   🎯 Cluster Visualization: {cluster_viz}")
    
    print(f"\n✅ Multi-metric clustering analysis completed successfully!")
    print(f"🎯 Best configuration selected using combined ranking of 3 metrics!")
    print(f"{'='*80}\n")

def load_clustering_analyzer(embedding_type: str = 'fasttext', country: str = 'US') -> CompanyClusteringAnalyzer:
    """
    Load existing clustering analyzer for use in other pipelines
    
    Returns:
        CompanyClusteringAnalyzer: Loaded analyzer or None if not available
    """
    try:
        analyzer = CompanyClusteringAnalyzer(embedding_type, country)
        
        if analyzer.load_clustering_results():
            # Load company data for operations
            product_df = pd.read_csv(EMBEDDINGS_OUTPUT)
            product_df['Firm ID'] = product_df['hojin_id'].astype(str)
            
            # Load company names
            us_web_data = pd.read_csv(US_WEB_DATA)
            company_name_map = us_web_data[['hojin_id', 'company_name']].drop_duplicates()
            product_df = product_df.merge(company_name_map, on='hojin_id', how='left')
            
            # Load embeddings
            from pipelines.patent_product_pipeline import get_embedding_file_paths
            file_paths = get_embedding_file_paths(embedding_type, country)
            product_rep = load_representations_from_json(file_paths['product_rep'])
            
            if product_rep:
                analyzer.prepare_data(product_rep, product_df)
                logger.info(f"Loaded clustering analyzer for {embedding_type} embeddings")
                return analyzer
        
        logger.warning(f"No clustering results found for {embedding_type} embeddings")
        return None
        
    except Exception as e:
        logger.warning(f"Could not load clustering analyzer: {e}")
        return None

# Legacy function for backward compatibility
def run_clustering_pipeline(config=None):
    """Legacy function name for backward compatibility"""
    return clustering_pipeline(config) 


    