"""
Clustering utilities for company embeddings analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import pickle
import os
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from configs.paths import get_clustering_file_paths, CLUSTERING_MODELS_DIR, CLUSTERING_RESULTS_DIR
from configs.hyperparams import *

logger = logging.getLogger(__name__)

class CompanyClusteringAnalyzer:
    """
    Analyzes company embeddings using clustering algorithms
    """
    
    def __init__(self, embedding_type='fasttext', country='US'):
        self.embedding_type = embedding_type
        self.country = country
        self.file_paths = get_clustering_file_paths(embedding_type, country)
        self.best_model = None
        self.best_n_clusters = None
        self.best_score = None
        self.cluster_assignments = None
        self.cluster_centers = None
        self.company_data = None
        self.embeddings_matrix = None
        self.company_ids = None
        
        # Create directories
        os.makedirs(CLUSTERING_MODELS_DIR, exist_ok=True)
        os.makedirs(CLUSTERING_RESULTS_DIR, exist_ok=True)
    
    def prepare_data(self, embeddings_dict: Dict, company_df: pd.DataFrame):
        """Prepare embeddings and company data for clustering"""
        logger.info(f"Preparing data for clustering analysis...")
        
        # Filter company data to only include companies with embeddings
        valid_company_ids = set(embeddings_dict.keys()) & set(company_df['hojin_id'].astype(str))
        
        self.company_data = company_df[company_df['hojin_id'].astype(str).isin(valid_company_ids)].copy()
        self.company_data['hojin_id'] = self.company_data['hojin_id'].astype(str)
        
        # Create embeddings matrix
        self.company_ids = list(valid_company_ids)
        self.embeddings_matrix = np.array([embeddings_dict[company_id] for company_id in self.company_ids])
        
        logger.info(f"Prepared {len(self.company_ids)} companies for clustering")
        logger.info(f"Embedding dimensions: {self.embeddings_matrix.shape[1]}")
        
        return self.embeddings_matrix, self.company_data
    
    def evaluate_clustering(self, n_clusters: int, algorithm: str = 'kmeans') -> Dict[str, float]:
        """Evaluate clustering with given number of clusters"""
        
        if algorithm == 'kmeans':
            model = KMeans(
                n_clusters=n_clusters,
                random_state=CLUSTERING_RANDOM_STATE,
                max_iter=CLUSTERING_MAX_ITER,
                n_init=CLUSTERING_N_INIT
            )
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            # For DBSCAN, n_clusters is actually eps parameter
            model = DBSCAN(eps=n_clusters, min_samples=DBSCAN_MIN_SAMPLES)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Fit the model
        cluster_labels = model.fit_predict(self.embeddings_matrix)
        
        # Handle DBSCAN case where we might get noise points (-1 labels)
        if algorithm == 'dbscan':
            n_clusters_actual = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters_actual < 2:
                # Not enough clusters formed
                return {
                    'silhouette': -1,
                    'calinski_harabasz': 0,
                    'davies_bouldin': float('inf'),
                    'n_clusters_actual': n_clusters_actual,
                    'n_noise_points': sum(1 for x in cluster_labels if x == -1)
                }
        
        # Calculate evaluation metrics
        metrics = {}
        
        try:
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for metrics
                metrics['silhouette'] = silhouette_score(self.embeddings_matrix, cluster_labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(self.embeddings_matrix, cluster_labels)
                metrics['davies_bouldin'] = davies_bouldin_score(self.embeddings_matrix, cluster_labels)
            else:
                metrics['silhouette'] = -1
                metrics['calinski_harabasz'] = 0
                metrics['davies_bouldin'] = float('inf')
        except Exception as e:
            logger.warning(f"Error calculating metrics for {n_clusters} clusters: {e}")
            metrics['silhouette'] = -1
            metrics['calinski_harabasz'] = 0
            metrics['davies_bouldin'] = float('inf')
        
        # Add additional info
        metrics['n_clusters_actual'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        if algorithm == 'dbscan':
            metrics['n_noise_points'] = sum(1 for x in cluster_labels if x == -1)
        
        return metrics, model, cluster_labels
    
    def hyperparameter_tuning(self, algorithm: str = None) -> Dict:
        """Perform hyperparameter tuning to find optimal number of clusters using multi-metric ranking"""
        
        if algorithm is None:
            algorithm = CLUSTERING_ALGORITHM
        
        logger.info(f"Starting hyperparameter tuning for {algorithm} clustering...")
        logger.info(f"Using multi-metric ranking system with metrics: {CLUSTERING_METRICS}")
        
        if algorithm == 'dbscan':
            # For DBSCAN, we tune the eps parameter
            cluster_range = np.linspace(0.1, 2.0, 20)  # eps values to test
            logger.info(f"Testing eps values: {cluster_range}")
        else:
            cluster_range = CLUSTER_NUMBERS_RANGE
            logger.info(f"Testing cluster numbers: {cluster_range}")
        
        results = {}
        valid_results = {}  # Only results with valid clustering
        
        for param in tqdm(cluster_range, desc=f"Evaluating {algorithm} clustering"):
            try:
                metrics, model, labels = self.evaluate_clustering(param, algorithm)
                
                # Store results
                param_key = str(param)
                results[param_key] = metrics
                
                # Only consider valid clustering results for ranking
                if metrics['n_clusters_actual'] >= 2 and metrics['silhouette'] > -1:
                    valid_results[param_key] = {
                        'param': param,
                        'metrics': metrics,
                        'model': model,
                        'labels': labels
                    }
                
                logger.debug(f"Clusters: {param}, Silhouette: {metrics['silhouette']:.4f}, "
                           f"Calinski: {metrics['calinski_harabasz']:.2f}, Davies: {metrics['davies_bouldin']:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {param} clusters: {e}")
                results[str(param)] = {
                    'silhouette': -1,
                    'calinski_harabasz': 0,
                    'davies_bouldin': float('inf'),
                    'error': str(e)
                }
        
        if not valid_results:
            logger.error("No valid clustering results found!")
            evaluation_results = {
                'algorithm': algorithm,
                'embedding_type': self.embedding_type,
                'country': self.country,
                'best_params': None,
                'best_score': None,
                'all_results': results,
                'ranking_results': {},
                'data_info': {
                    'n_companies': len(self.company_ids),
                    'embedding_dim': self.embeddings_matrix.shape[1]
                }
            }
            
            # Save evaluation results
            with open(self.file_paths['evaluation_results'], 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            return evaluation_results
        
        # Multi-metric ranking system
        ranking_results = self.calculate_multi_metric_ranking(valid_results)
        
        # Select best configuration
        best_config = ranking_results['best_configuration']
        best_param = int(best_config['param'])  # Ensure integer type
        best_total_rank = best_config['total_rank_score']
        
        # Get the best model and labels
        best_model = valid_results[str(best_param)]['model']
        best_labels = valid_results[str(best_param)]['labels']
        best_metrics = valid_results[str(best_param)]['metrics']
        
        # Store results
        evaluation_results = {
            'algorithm': algorithm,
            'embedding_type': self.embedding_type,
            'country': self.country,
            'ranking_method': 'multi_metric_ranking',
            'best_params': best_param,
            'best_total_rank_score': best_total_rank,
            'best_metrics': best_metrics,
            'all_results': results,
            'ranking_results': ranking_results,
            'data_info': {
                'n_companies': len(self.company_ids),
                'embedding_dim': self.embeddings_matrix.shape[1]
            }
        }
        
        # Save evaluation results
        with open(self.file_paths['evaluation_results'], 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Save ranking results separately
        with open(self.file_paths['ranking_results'], 'w') as f:
            json.dump(ranking_results, f, indent=2)
        
        # Create and save performance plots
        self.create_performance_plots(valid_results, ranking_results)
        
        if best_model is not None:
            # Save best model
            with open(self.file_paths['best_model'], 'wb') as f:
                pickle.dump(best_model, f)
            
            # Store best results
            self.best_model = best_model
            self.best_n_clusters = best_param
            self.best_score = best_total_rank
            self.cluster_assignments = best_labels
            
            # Save cluster assignments
            assignments_df = pd.DataFrame({
                'company_id': self.company_ids,
                'cluster': best_labels
            })
            assignments_df.to_csv(self.file_paths['assignments'], index=False)
            
            # Save cluster centers (if available)
            if hasattr(best_model, 'cluster_centers_'):
                np.save(self.file_paths['centers'], best_model.cluster_centers_)
                self.cluster_centers = best_model.cluster_centers_
            
            # Save metadata
            metadata = {
                'algorithm': algorithm,
                'best_n_clusters': int(best_param) if algorithm != 'dbscan' else None,
                'best_eps': float(best_param) if algorithm == 'dbscan' else None,
                'best_total_rank_score': float(best_total_rank),
                'best_metrics': best_metrics,
                'ranking_method': 'multi_metric_ranking',
                'n_companies': len(self.company_ids),
                'embedding_type': self.embedding_type,
                'country': self.country
            }
            
            with open(self.file_paths['metadata'], 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Best clustering: {algorithm} with {best_param} clusters")
            logger.info(f"Total rank score: {best_total_rank} (lower is better)")
            logger.info(f"Best metrics - Silhouette: {best_metrics['silhouette']:.4f}, "
                       f"Calinski: {best_metrics['calinski_harabasz']:.2f}, "
                       f"Davies: {best_metrics['davies_bouldin']:.4f}")
        
        return evaluation_results
    
    def calculate_multi_metric_ranking(self, valid_results: Dict) -> Dict:
        """Calculate multi-metric ranking for cluster configurations"""
        
        logger.info("Calculating multi-metric ranking...")
        
        # Extract metrics for ranking
        params = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        for param_key, result in valid_results.items():
            param = result['param']
            metrics = result['metrics']
            
            params.append(param)
            silhouette_scores.append(metrics['silhouette'])
            calinski_scores.append(metrics['calinski_harabasz'])
            davies_scores.append(metrics['davies_bouldin'])
        
        # Create dataframe for easier ranking
        df = pd.DataFrame({
            'param': [int(p) for p in params],  # Ensure integers
            'silhouette': silhouette_scores,
            'calinski_harabasz': calinski_scores,
            'davies_bouldin': davies_scores
        })
        
        # Rank each metric (1 = best rank)
        # Silhouette and Calinski: higher is better
        df['silhouette_rank'] = df['silhouette'].rank(ascending=False, method='min')
        df['calinski_rank'] = df['calinski_harabasz'].rank(ascending=False, method='min')
        
        # Davies-Bouldin: lower is better
        df['davies_rank'] = df['davies_bouldin'].rank(ascending=True, method='min')
        
        # Calculate total rank score (sum of ranks - lower is better)
        df['total_rank_score'] = df['silhouette_rank'] + df['calinski_rank'] + df['davies_rank']
        
        # Find best configuration(s)
        min_total_rank = df['total_rank_score'].min()
        best_candidates = df[df['total_rank_score'] == min_total_rank]
        
        # If tie, select the one with largest number of clusters
        best_config = best_candidates.loc[best_candidates['param'].idxmax()]
        
        # Prepare ranking results
        ranking_results = {
            'ranking_method': 'multi_metric_total_rank',
            'metrics_used': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
            'ranking_details': df.sort_values('total_rank_score').to_dict('records'),
            'best_configuration': {
                'param': int(best_config['param']),  # Ensure integer
                'total_rank_score': float(best_config['total_rank_score']),
                'silhouette_score': float(best_config['silhouette']),
                'silhouette_rank': int(best_config['silhouette_rank']),
                'calinski_score': float(best_config['calinski_harabasz']),
                'calinski_rank': int(best_config['calinski_rank']),
                'davies_score': float(best_config['davies_bouldin']),
                'davies_rank': int(best_config['davies_rank'])
            },
            'summary': {
                'total_configurations_tested': len(df),
                'best_total_rank_score': float(min_total_rank),
                'num_tied_configurations': len(best_candidates),
                'tie_breaking_rule': 'largest_cluster_number'
            }
        }
        
        logger.info(f"Ranking completed: Best config has {int(best_config['param'])} clusters with total rank score {min_total_rank}")
        logger.info(f"Ranks - Silhouette: {best_config['silhouette_rank']:.0f}, "
                   f"Calinski: {best_config['calinski_rank']:.0f}, Davies: {best_config['davies_rank']:.0f}")
        
        return ranking_results
    
    def create_performance_plots(self, valid_results: Dict, ranking_results: Dict):
        """Create and save performance plots for all metrics"""
        
        logger.info("Creating performance plots...")
        
        # Extract data for plotting and sort by cluster number for smooth lines
        ranking_details = ranking_results['ranking_details']
        
        # Sort by cluster number (param) for proper x-axis ordering
        sorted_details = sorted(ranking_details, key=lambda x: int(x['param']))
        
        params = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        total_ranks = []
        
        for detail in sorted_details:
            params.append(int(detail['param']))  # Ensure integer
            silhouette_scores.append(detail['silhouette'])
            calinski_scores.append(detail['calinski_harabasz'])
            davies_scores.append(detail['davies_bouldin'])
            total_ranks.append(detail['total_rank_score'])
        
        # Create the plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Best configuration info
        best_config = ranking_results['best_configuration']
        best_param = int(best_config['param'])  # Ensure integer
        
        # Plot 1: Silhouette Score
        ax1.plot(params, silhouette_scores, 'b-o', linewidth=2, markersize=6)
        ax1.axvline(x=best_param, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_param} clusters')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score vs Number of Clusters\n(Higher is Better)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Calinski-Harabasz Index
        ax2.plot(params, calinski_scores, 'g-o', linewidth=2, markersize=6)
        ax2.axvline(x=best_param, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_param} clusters')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Calinski-Harabasz Index')
        ax2.set_title('Calinski-Harabasz Index vs Number of Clusters\n(Higher is Better)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Davies-Bouldin Index
        ax3.plot(params, davies_scores, 'orange', linewidth=2, marker='o', markersize=6)
        ax3.axvline(x=best_param, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_param} clusters')
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Davies-Bouldin Index')
        ax3.set_title('Davies-Bouldin Index vs Number of Clusters\n(Lower is Better)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Total Rank Score
        ax4.plot(params, total_ranks, 'purple', linewidth=2, marker='o', markersize=6)
        ax4.axvline(x=best_param, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_param} clusters')
        ax4.set_xlabel('Number of Clusters')
        ax4.set_ylabel('Total Rank Score')
        ax4.set_title('Total Rank Score vs Number of Clusters\n(Lower is Better - Sum of All Ranks)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Main title
        fig.suptitle(f'Clustering Performance Analysis ({self.embedding_type.title()} Embeddings)\n'
                    f'Best Configuration: {int(best_param)} clusters (Total Rank Score: {best_config["total_rank_score"]:.0f})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.file_paths['performance_plot'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(self.file_paths['performance_plot'], dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {self.file_paths['performance_plot']}")
        
        # Also create a summary table plot
        self.create_ranking_summary_plot(ranking_results)
    
    def create_ranking_summary_plot(self, ranking_results: Dict):
        """Create a summary table plot showing ranking details"""
        
        # Create a nice table visualization
        ranking_details = ranking_results['ranking_details']
        best_config = ranking_results['best_configuration']
        
        # Prepare data for table (top 10 configurations)
        top_configs = ranking_details[:10]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Rank', 'Clusters', 'Silhouette\nScore', 'Silhouette\nRank', 
                  'Calinski\nScore', 'Calinski\nRank', 'Davies\nScore', 'Davies\nRank', 'Total\nRank Score']
        
        for i, config in enumerate(top_configs):
            row = [
                i + 1,  # Overall rank
                int(config['param']),  # Ensure integer display
                f"{config['silhouette']:.4f}",
                int(config['silhouette_rank']),
                f"{config['calinski_harabasz']:.2f}",
                int(config['calinski_rank']),
                f"{config['davies_bouldin']:.4f}",
                int(config['davies_rank']),
                f"{config['total_rank_score']:.0f}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color the best configuration row
        best_param = int(best_config['param'])  # Ensure integer
        for i, config in enumerate(top_configs):
            if int(config['param']) == best_param:  # Ensure integer comparison
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#90EE90')  # Light green
                break
        
        # Style header
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#4CAF50')  # Green
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # Add title
        title = f'Top 10 Clustering Configurations - Multi-Metric Ranking\n'
        title += f'({self.embedding_type.title()} Embeddings)\n'
        title += f'Best: {int(best_config["param"])} clusters with Total Rank Score {best_config["total_rank_score"]:.0f}'
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add explanation
        explanation = 'Ranking Method: Lower Total Rank Score is Better (Sum of Individual Ranks)\n'
        explanation += 'Silhouette & Calinski: Higher values get better ranks | Davies-Bouldin: Lower values get better ranks\n'
        explanation += 'Tie-breaking: Largest number of clusters selected if total rank scores are equal'
        
        plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=10, style='italic')
        
        # Save table plot
        table_plot_path = self.file_paths['performance_plot'].replace('.png', '_ranking_table.png')
        plt.savefig(table_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Ranking summary table saved to {table_plot_path}")
    
    def load_clustering_results(self) -> bool:
        """Load previously computed clustering results"""
        try:
            # Load best model
            if os.path.exists(self.file_paths['best_model']):
                with open(self.file_paths['best_model'], 'rb') as f:
                    self.best_model = pickle.load(f)
            
            # Load assignments
            if os.path.exists(self.file_paths['assignments']):
                assignments_df = pd.read_csv(self.file_paths['assignments'])
                self.cluster_assignments = assignments_df['cluster'].values
                self.company_ids = assignments_df['company_id'].tolist()
            
            # Load cluster centers
            if os.path.exists(self.file_paths['centers']):
                self.cluster_centers = np.load(self.file_paths['centers'])
            
            # Load metadata
            if os.path.exists(self.file_paths['metadata']):
                with open(self.file_paths['metadata'], 'r') as f:
                    metadata = json.load(f)
                    self.best_n_clusters = metadata.get('best_n_clusters')
                    self.best_score = metadata.get('best_score')
            
            logger.info(f"Loaded clustering results for {self.embedding_type} embeddings")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load clustering results: {e}")
            return False
    
    def get_cluster_info(self, cluster_id: int) -> Dict:
        """Get information about a specific cluster"""
        if self.cluster_assignments is None or self.company_data is None:
            return {}
        
        # Get companies in this cluster
        cluster_mask = self.cluster_assignments == cluster_id
        cluster_company_ids = [self.company_ids[i] for i in range(len(self.company_ids)) if cluster_mask[i]]
        
        # Get company information
        cluster_companies = self.company_data[
            self.company_data['hojin_id'].isin(cluster_company_ids)
        ].copy()
        
        # Add cluster info
        cluster_companies['cluster_id'] = cluster_id
        
        return {
            'cluster_id': cluster_id,
            'n_companies': len(cluster_company_ids),
            'companies': cluster_companies,
            'company_ids': cluster_company_ids
        }
    
    def find_nearest_cluster(self, query_embedding: np.ndarray) -> Tuple[int, float]:
        """Find the nearest cluster for a query embedding"""
        if self.cluster_centers is None:
            # Use company embeddings to find nearest cluster
            if self.best_model is not None and hasattr(self.best_model, 'predict'):
                predicted_cluster = self.best_model.predict([query_embedding])[0]
                
                # Calculate distance to cluster center
                if hasattr(self.best_model, 'cluster_centers_'):
                    center = self.best_model.cluster_centers_[predicted_cluster]
                    distance = np.linalg.norm(query_embedding - center)
                else:
                    distance = 0.0
                
                return predicted_cluster, distance
        else:
            # Calculate distances to all cluster centers
            distances = np.linalg.norm(self.cluster_centers - query_embedding, axis=1)
            nearest_cluster = np.argmin(distances)
            min_distance = distances[nearest_cluster]
            
            return nearest_cluster, min_distance
        
        return -1, float('inf')
    
    def visualize_clusters(self, save_path: str = None):
        """Create visualization of clusters using PCA"""
        if self.embeddings_matrix is None or self.cluster_assignments is None:
            logger.warning("No clustering data available for visualization")
            return
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2, random_state=CLUSTERING_RANDOM_STATE)
        embeddings_2d = pca.fit_transform(self.embeddings_matrix)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Get unique clusters
        unique_clusters = np.unique(self.cluster_assignments)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_assignments == cluster_id
            label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7, s=50)
        
        # Add cluster centers if available
        if self.cluster_centers is not None:
            centers_2d = pca.transform(self.cluster_centers)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centers')
        
        plt.title(f'Company Embeddings Clustering ({self.embedding_type})\n'
                 f'Best: {self.best_n_clusters} clusters, {PRIMARY_CLUSTERING_METRIC}: {self.best_score:.3f}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.file_paths['visualization']
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved clustering visualization to {save_path}")

def run_clustering_analysis(embeddings_dict: Dict, company_df: pd.DataFrame, 
                          embedding_type: str = 'fasttext', country: str = 'US',
                          force_rebuild: bool = False) -> CompanyClusteringAnalyzer:
    """
    Run complete clustering analysis for company embeddings
    """
    logger.info(f"Starting clustering analysis for {embedding_type} embeddings...")
    
    analyzer = CompanyClusteringAnalyzer(embedding_type, country)
    
    # Check if results already exist
    if not force_rebuild and analyzer.load_clustering_results():
        logger.info("Loaded existing clustering results")
        # Still need to prepare data for query operations
        analyzer.prepare_data(embeddings_dict, company_df)
        return analyzer
    
    # Prepare data
    analyzer.prepare_data(embeddings_dict, company_df)
    
    # Run hyperparameter tuning
    evaluation_results = analyzer.hyperparameter_tuning()
    
    # Create visualization
    analyzer.visualize_clusters()
    
    logger.info(f"Clustering analysis completed!")
    logger.info(f"Best configuration: {evaluation_results['best_params']} clusters")
    if 'best_total_rank_score' in evaluation_results:
        logger.info(f"Best total rank score: {evaluation_results['best_total_rank_score']:.4f}")
    
    return analyzer 