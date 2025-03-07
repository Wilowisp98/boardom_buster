import polars as pl
import numpy as np
from typing import List, Dict, Any
from sklearn.manifold import TSNE
import polars as pl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import pickle
import hashlib
import time

# Constants
RELEVANT_COLUMNS: List[str] = [
    "AGE_GROUP",
    "GAME_CAT",
    "LANGUAGE_DEPENDENCY",
    "GAME_DURATION",
    "GAME_DIFFICULTY"
]

# Helper functions
def get_feature_columns(df: pl.DataFrame) -> List[str]:
    """
    Gets all feature columns based on prefixes.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing board game data
        
    Returns:
    --------
    List[str]
        List of feature column names
    """
    feature_columns = []
    for prefix in RELEVANT_COLUMNS:
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        feature_columns.extend(matching_cols)

    return feature_columns

class bgClusters:
    """
    Class for clustering board games based on feature data with support for constrained clustering.
    """
    
    def __init__(self):
        """Initialize the clustering model with default settings."""
        self.scaler = StandardScaler()
        self.games_df = None
        self.feature_data = None
        self.cluster_descriptions = None
        self.id_column = None
        self.name_column = None
        self.constraint_columns = []
        self.clusters = {}
        self.scaled_features = None

    #############################################
    # MAIN PUBLIC INTERFACE
    #############################################
    
    def fit(self, games_df, constraint_columns, name_column=None, cache_dir='model/clustering_results/cluster', cache_filename='latest_clusters.pkl'):
        """
        Fit the model with constrained clustering.

        Parameters:
        -----------
        games_df : polars.DataFrame
            DataFrame containing board game features
        constraint_columns : list
            List of column names that will be used to force clustering
            Example: ['GAME_CAT_GROUP_card_game', 'GAME_CAT_GROUP_abstract_strategy']
        name_column : str, optional
            Name of the column containing game names
        cache_dir : str, optional
            Directory to store cached clustering results
        cache_filename : str, optional
            Name of the cache file to use
            
        Returns:
        --------
        self : bgClusters
            The fitted model instance
        """
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Use a simple filename for the cache
        cache_file = os.path.join(cache_dir, cache_filename)

        # Load from cache if it exists
        if os.path.exists(cache_file):
            print(f"Loading cached clustering results from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

                # Reconstruct clusters from simplified cache
                self.clusters = {}

                if 'cluster_to_games' in cached_data:
                    # New simplified format
                    cluster_mapping = cached_data['cluster_to_games']

                    for cluster_id, game_names in cluster_mapping.items():
                        self.clusters[cluster_id] = {
                            'constraint': f"cluster_{cluster_id}",
                            'game_names': game_names,
                            'count': len(game_names)
                        }

                    # Store the games_df
                    self.games_df = games_df.clone()

                    # Infer the rest based on the games_df
                    self._add_cluster_labels()
                    self.generate_cluster_descriptions()

                    # Set required attributes for proper functioning
                    self.feature_columns = get_feature_columns(self.games_df)
                    self.name_column = name_column
                    self.constraint_columns = constraint_columns

                    return self

        # If no cache exists, proceed with clustering
        self.games_df = games_df.clone()
        self.unconstrained_df = self.games_df.clone()  # Fixed typo in variable name
        self.name_column = name_column
        self.constraint_columns = constraint_columns

        # Extract feature data
        self.feature_columns = get_feature_columns(self.games_df)
        self.feature_data = games_df.select(self.feature_columns).to_numpy()

        # Scale the data for future similarity calculations
        self.scaler.fit(self.feature_data)
        self.scaled_features = self.scaler.transform(self.feature_data)

        # Handle constrained and unconstrained clustering
        unconstrained_df = self.games_df.clone()
        for column in self.constraint_columns:
            constrained_df = self.games_df.filter(pl.col(column) == 1)
            unconstrained_df = unconstrained_df.filter(pl.col(column) != 1)
            self._create_constrained_clusters(constrained_df, is_constraint=True, constraint_name=column.split('GAME_CAT_GROUP_')[1])

        # Create clusters for unconstrained data
        self._create_constrained_clusters(unconstrained_df)

        # Add cluster labels to the DataFrame
        self._add_cluster_labels()

        # Generate cluster descriptions
        self.generate_cluster_descriptions()

        # Generate visualizations of clustering results
        self._visualize_clustering_results()
        
        # Save results to cache
        self._save_to_cache(cache_file)

        return self
    
    def generate_cluster_descriptions(self):
        """
        Generate descriptions for each cluster based on feature importance.
        
        Returns:
        --------
        dict
            Dictionary containing descriptions for each cluster
        """
        self.cluster_descriptions = {}
        
        for cluster_id, cluster_info in self.clusters.items():
            # Get games in this cluster
            cluster_games = self.games_df.filter(pl.col("cluster") == cluster_id)
            
            if cluster_games.height == 0:
                continue
                
            # Calculate average feature values for this cluster
            cluster_means = {}
            for feature in self.feature_columns:
                cluster_means[feature] = cluster_games.select(pl.col(feature)).mean().item()
            
            # Find the most important features (highest average values)
            sorted_features = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, _ in sorted_features[:5]]
            
            # Create a description
            constraint_name = cluster_info['constraint'].replace("GAME_CAT_GROUP_", "").replace("_", " ")
            
            if cluster_info.get('is_subcluster', False):
                parent_constraint = cluster_info.get('parent_constraint', '')
                parent_name = parent_constraint.replace("_", " ")
                subcluster_id = cluster_info.get('subcluster_id', 0)
                
                description = f"Cluster {cluster_id}: {parent_name.title()} (Subcluster {subcluster_id}) games characterized by "
            elif any(f"GAME_CAT_GROUP_{constraint_name.split('_')[0]}" in self.constraint_columns for constraint_name in [cluster_info['constraint']]):
                description = f"Cluster {cluster_id}: {constraint_name.title()} games characterized by "
            else:
                description = f"Cluster {cluster_id}: Other games characterized by "
                
            description += ", ".join([
                f.replace("GAME_CAT_GROUP_", "").replace("AGE_GROUP_", "")
                .replace("LANGUAGE_DEPENDENCY_", "language: ")
                .replace("GAME_DURATION_", "duration: ")
                .replace("GAME_DIFFICULTY_", "difficulty: ")
                .replace("_", " ") 
                for f in top_features
            ])
            
            # Get sample games
            if self.name_column:
                sample_games = cluster_games.select(pl.col(self.name_column)).to_series().to_list()[:5]
            else:
                sample_games = [f"Game #{i}" for i in range(min(5, cluster_games.height))]
            
            self.cluster_descriptions[cluster_id] = {
                'description': description,
                'constraint': cluster_info['constraint'],
                'is_subcluster': cluster_info.get('is_subcluster', False),
                'parent_constraint': cluster_info.get('parent_constraint', ''),
                'top_features': top_features,
                'count': cluster_games.height,
                'sample_games': sample_games
            }
        
        return self.cluster_descriptions
    
    def get_all_clusters_info(self):
        """
        Get information about all clusters.
        
        Returns:
        --------
        dict
            Dictionary containing descriptions for each cluster
        """
        return self.cluster_descriptions
    
    #############################################
    # CLUSTERING IMPLEMENTATION
    #############################################
    
    def _create_constrained_clusters(self, data_to_cluster, is_constraint=False, constraint_name=None):
        """
        Create clusters based on constraint columns, with optional subclustering within constraints.
        
        Parameters:
        -----------
        data_to_cluster : polars.DataFrame
            DataFrame containing board game features to cluster
        is_constraint : bool, optional
            Whether this clustering is based on a constraint
        constraint_name : str, optional
            Name of the constraint
        """
        # Don't reset clusters on each call, initialize if needed
        if not hasattr(self, 'clusters') or self.clusters is None:
            self.clusters = {}

        # Get current cluster ID
        cluster_id = len(self.clusters)

        # Extract feature data for clustering
        feature_data = data_to_cluster.select(self.feature_columns).to_numpy()

        # Find optimal number of clusters
        optimal_k = self._find_optimal_k(
            data=feature_data, 
            is_constraint=is_constraint
        )

        # Apply K-means to create subclusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_data)

        # Create cluster entries for each subcluster
        for sub_id in range(optimal_k):
            # Get indices of games in this subcluster
            sub_indices = [i for i in range(len(labels)) if labels[i] == sub_id]

            if sub_indices:
                # Get game names if name column exists
                if self.name_column and self.name_column in data_to_cluster.columns:
                    game_ids = data_to_cluster.select(pl.col(self.name_column)).to_series().to_list()
                    sub_games = [game_ids[i] for i in sub_indices]
                else:
                    # Fall back to indices if no name column
                    sub_games = sub_indices

                # Handle naming based on whether this is a constraint-based cluster
                if constraint_name:
                    subcluster_name = f"{constraint_name}_subcluster_{sub_id}"
                    parent_constraint = constraint_name
                    is_subcluster = True
                else:
                    subcluster_name = f"unconstrained_cluster_{sub_id}"
                    parent_constraint = None
                    is_subcluster = False

                self.clusters[cluster_id] = {
                    'constraint': subcluster_name,
                    'parent_constraint': parent_constraint,
                    'game_names': sub_games,  # Store game names instead of indices
                    'indices': sub_indices,   # Keep indices for backward compatibility
                    'count': len(sub_indices),
                    'is_subcluster': is_subcluster,
                    'subcluster_id': sub_id
                }
                cluster_id += 1

    def _find_optimal_k(self, data, max_k=20, min_k=2, is_constraint=False):
        """
        Find the optimal number of clusters using multiple evaluation metrics with adaptive size constraints.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Feature data to cluster
        max_k : int, optional
            Maximum number of clusters to consider
        min_k : int, optional
            Minimum number of clusters to consider
        is_constraint : bool, optional
            Whether this clustering is based on a constraint
            
        Returns:
        --------
        int
            Optimal number of clusters
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        import numpy as np
        from collections import Counter
        import matplotlib.pyplot as plt
        import os
        import io
        from contextlib import redirect_stdout

        # Create output directory if it doesn't exist
        os.makedirs("model/clustering_results", exist_ok=True)

        # Prepare to capture printed output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            # Calculate adaptive optimal range based on data size
            data_size = len(data)
            adaptive_min_k = max(2, int(np.log(data_size)))
            adaptive_max_k = max(5, int(np.sqrt(data_size)))

            # Set appropriate minimum k based on constraints and adaptive range
            if is_constraint:
                actual_min_k = max(min_k, adaptive_min_k - 1)  # Slightly more flexible for constraints
            else:
                actual_min_k = max(min_k, adaptive_min_k)

            # Handle edge cases with limited data
            if data_size < actual_min_k + 1:
                return max(2, data_size - 1) if is_constraint else max(1, data_size - 1)

            # Adjust max_k based on data dimensions and adaptive range
            feature_count = data.shape[1]
            max_k = min(max_k, feature_count, data_size - 1, adaptive_max_k * 2)  # Allow exploring up to 2x the adaptive max

            if max_k < actual_min_k:
                return max_k

            # Initialize storage for metrics
            k_range = range(actual_min_k, max_k + 1)
            silhouette_scores = []
            db_scores = []          # Davies-Bouldin Index
            cluster_size_scores = []  # New metric for cluster count appropriateness

            # Evaluate each k using multiple metrics
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)

                # Calculate Silhouette score (higher is better)
                if data_size > k and k > 1:
                    sil_score = silhouette_score(data, labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)

                # Calculate Davies-Bouldin Index (lower is better)
                if k > 1:
                    db_score = davies_bouldin_score(data, labels)
                    db_scores.append(db_score)
                else:
                    db_scores.append(float('inf'))

                # Calculate cluster size appropriateness score
                if adaptive_min_k <= k <= adaptive_max_k:
                    # Full score within the optimal range
                    cluster_size_scores.append(1.0)
                else:
                    # Gradually decreasing score outside optimal range
                    distance = min(abs(k - adaptive_min_k), abs(k - adaptive_max_k))
                    max_reasonable = int(np.sqrt(data_size) / 2)  # Upper limit for reasonable clusters
                    score = max(0, 1.0 - (distance / max(1, max_reasonable)))
                    cluster_size_scores.append(score)

            # Normalize scores to [0, 1] range for fair comparison
            def normalize(scores, higher_is_better=True):
                if all(s == scores[0] for s in scores):
                    return [1] * len(scores) if higher_is_better else [0] * len(scores)

                min_val, max_val = min(scores), max(scores)
                if min_val == max_val:
                    return [0.5] * len(scores)

                # For metrics where higher is better, normalize to [0,1]
                # For metrics where lower is better, invert the normalized score
                normalized = [(s - min_val) / (max_val - min_val) for s in scores]

                if not higher_is_better:
                    normalized = [1 - n for n in normalized]  # Invert scores where lower is better

                return normalized

            # Apply normalized scores
            norm_silhouette = normalize(silhouette_scores, higher_is_better=True)  # Higher is better
            norm_db = normalize(db_scores, higher_is_better=False)                 # Lower is better, so invert
            norm_cluster_size = cluster_size_scores  # Already normalized

            # Weighted voting - combine all metrics
            # Adjusted weights to account for removed metrics
            weights = {
                'silhouette': 0.35,    # Good for finding well-separated clusters
                'db': 0.35,            # Good for identifying distinct clusters
                'cluster_size': 0.30   # Weight for appropriate cluster count
            }

            final_scores = []
            for i in range(len(k_range)):
                score = (
                    weights['silhouette'] * norm_silhouette[i] +
                    weights['db'] * norm_db[i] +
                    weights['cluster_size'] * norm_cluster_size[i]
                )
                final_scores.append(score)

            # Find the optimal k directly from final_scores without smoothing
            best_idx = np.argmax(final_scores)
            optimal_k = k_range[best_idx]

            # Create iteration history to show the progression of values
            history = {
                'k': list(k_range),
                'silhouette': norm_silhouette,
                'db': norm_db,
                'cluster_size': norm_cluster_size,
                'final': final_scores
            }

            # Generate a timestamp for unique filenames
            import time
            timestamp = int(time.time())

            # VISUALIZATION 0: Optimization process
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})

            # Top plot: Detailed progression of each k evaluation
            ax = axes[0]
            x = np.arange(len(k_range))
            width = 0.2
            offsets = [-1, 0, 1]

            # Plot bars for each metric side by side
            ax.bar(x + offsets[0]*width, norm_silhouette, width, label='Silhouette', color='blue', alpha=0.7)
            ax.bar(x + offsets[1]*width, norm_db, width, label='Davies-Bouldin', color='red', alpha=0.7)
            ax.bar(x + offsets[2]*width, norm_cluster_size, width, label='Cluster Size', color='orange', alpha=0.7)

            # Add line showing weighted total
            ax.plot(x, final_scores, 'ko-', linewidth=2, label='Weighted Total')

            # Highlight the best k
            best_k_idx = list(k_range).index(optimal_k)
            ax.axvline(x=best_k_idx, color='black', linestyle='--')
            ax.text(best_k_idx, max(final_scores)+0.05, f'Optimal k={optimal_k}', 
                    ha='center', va='bottom', fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(k_range)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Normalized Score')
            ax.set_title('Detailed View of Each Metric by Cluster Count')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left')

            # Bottom plot: Progression of metrics across k values
            ax = axes[1]
            metrics = [
                ('Silhouette', norm_silhouette, 'blue', 'o-'), 
                ('Davies-Bouldin', norm_db, 'red', 'v-'),
                ('Cluster Size', norm_cluster_size, 'orange', 'p-'),
                ('Weighted Total', final_scores, 'black', '*-')
            ]

            for name, values, color, style in metrics:
                ax.plot(k_range, values, style, label=name, color=color)

            ax.axvline(x=optimal_k, color='black', linestyle='--')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Score')
            ax.set_title('Progression of Evaluation Metrics Across k Values')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')

            plt.tight_layout()
            # Save to file
            fig.savefig(f'model/clustering_results/optimization_process_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # VISUALIZATION 1: Individual metrics (original)
            fig = plt.figure(figsize=(12, 6))
            plt.plot(k_range, norm_silhouette, 'o-', label='Silhouette Score (normalized)', color='blue')
            plt.plot(k_range, norm_db, 'v-', label='Davies-Bouldin Score (normalized)', color='red')
            plt.plot(k_range, norm_cluster_size, 'p-', label='Cluster Size Score', color='orange')
            plt.axvline(x=optimal_k, color='black', linestyle='--', label=f'Optimal k = {optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Normalized Score')
            plt.title('Individual Evaluation Metrics for Cluster Optimization')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            # Save to file
            fig.savefig(f'model/clustering_results/individual_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # VISUALIZATION 2: Combined scores with iteration markers
            fig = plt.figure(figsize=(10, 5))
            plt.plot(k_range, final_scores, 'o-', color='blue', label='Combined Score')

            # Add arrows showing the progression of testing
            for i in range(len(k_range)-1):
                plt.annotate('', 
                    xy=(k_range[i+1], final_scores[i+1]),
                    xytext=(k_range[i], final_scores[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5))

            # Add iteration numbers
            for i in range(len(k_range)):
                plt.text(k_range[i], final_scores[i] + 0.02, f"{i+1}", 
                        ha='center', va='bottom', fontsize=9, color='darkblue')

            plt.axvline(x=optimal_k, color='black', linestyle='--', label=f'Optimal k = {optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Score')
            plt.title('Combined Cluster Evaluation Scores with Iteration Path')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            # Save to file
            fig.savefig(f'model/clustering_results/combined_scores_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Print iterations summary
            print("\nClustering Optimization Iterations:")
            print("-" * 50)
            print(f"{'k':>3} {'Silhouette':>12} {'DB':>12} {'Size':>12} {'Total':>12}")
            print("-" * 50)
            for i, k in enumerate(k_range):
                print(f"{k:3d} {norm_silhouette[i]:12.4f} {norm_db[i]:12.4f} "
                      f"{norm_cluster_size[i]:12.4f} {final_scores[i]:12.4f}")
            print("-" * 50)
            print(f"Selected optimal k = {optimal_k} (data size: {data_size})\n")

        # Write the captured output to a file
        with open(f'model/clustering_results/clustering_log_{timestamp}.txt', 'w') as f:
            f.write(captured_output.getvalue())

        return optimal_k
    
    def _add_cluster_labels(self):
        """
        Add cluster labels to the DataFrame.
        """
        # Initialize all cluster labels to -1
        cluster_labels = [-1] * self.games_df.height
        subcluster_labels = [-1] * self.games_df.height
        parent_constraint_labels = [""] * self.games_df.height

        # Get the game names from the DataFrame
        if self.name_column and self.name_column in self.games_df.columns:
            game_names = self.games_df.select(pl.col(self.name_column)).to_series().to_list()

            # Create a mapping from game name to index
            name_to_idx = {name: idx for idx, name in enumerate(game_names)}

            # Assign cluster labels using game names
            for cluster_id, cluster_info in self.clusters.items():
                for game_name in cluster_info.get('game_names', []):
                    if game_name in name_to_idx:
                        idx = name_to_idx[game_name]
                        cluster_labels[idx] = cluster_id

                        # If this is a subcluster, store the subcluster ID and parent constraint
                        if cluster_info.get('is_subcluster', False):
                            subcluster_labels[idx] = cluster_info.get('subcluster_id', -1)
                            parent_constraint_labels[idx] = cluster_info.get('parent_constraint', "")
        else:
            # Fall back to using indices if no name column
            for cluster_id, cluster_info in self.clusters.items():
                for idx in cluster_info.get('indices', []):
                    if 0 <= idx < len(cluster_labels):
                        cluster_labels[idx] = cluster_id

                        # If this is a subcluster, store the subcluster ID and parent constraint
                        if cluster_info.get('is_subcluster', False):
                            subcluster_labels[idx] = cluster_info.get('subcluster_id', -1)
                            parent_constraint_labels[idx] = cluster_info.get('parent_constraint', "")

        # Add to DataFrame
        self.games_df = self.games_df.with_columns([
            pl.Series("cluster", cluster_labels),
            pl.Series("subcluster", subcluster_labels),
            pl.Series("parent_constraint", parent_constraint_labels)
        ])
    
    #############################################
    # CACHING AND PERSISTENCE
    #############################################

    def _save_to_cache(self, cache_file):
        """
        Save minimal clustering results to cache file - only cluster numbers and game names.

        Parameters:
        -----------
        cache_file : str
            Path to the cache file
        """
        # Create a simplified data structure with only essential information
        minimal_clusters = {}

        # For each cluster, only store its ID and the games belonging to it
        for cluster_id, cluster_info in self.clusters.items():
            game_names = cluster_info.get('game_names', [])
            minimal_clusters[cluster_id] = game_names

        # Save the minimal data structure to a file
        simplified_cache_data = {
            'cluster_to_games': minimal_clusters,
            'creation_timestamp': time.time()
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(simplified_cache_data, f)

        print(f"Saved simplified cluster data to {cache_file}")

    def _visualize_clustering_results(self):
        """
        Generate TSNE and PCA visualizations of the clustering results.
        Saves visualizations to the clustering_results directory.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import time
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Create output directory if it doesn't exist
        os.makedirs("model/clustering_results", exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = int(time.time())
        
        # Extract feature data and cluster labels
        feature_data = self.games_df.select(self.feature_columns).to_numpy()
        cluster_labels = self.games_df["cluster"].to_numpy()
        
        # Skip visualization if there's no clustering data
        if len(set(cluster_labels)) <= 1 or len(feature_data) == 0:
            print("Not enough clusters or data for visualization")
            return
        
        # Generate a colormap with distinct colors for each cluster
        unique_clusters = sorted(set(cluster_labels))
        num_clusters = len(unique_clusters)
        
        # Create a colormap with good separation between colors
        base_colors = list(mcolors.TABLEAU_COLORS.values())
        if num_clusters > len(base_colors):
            # Add more colors if needed
            import random
            random.seed(42)  # For reproducibility
            additional_colors = []
            while len(base_colors) + len(additional_colors) < num_clusters:
                new_color = '#%02X%02X%02X' % (random.randint(0, 255), 
                                             random.randint(0, 255),
                                             random.randint(0, 255))
                additional_colors.append(new_color)
            colors = base_colors + additional_colors
        else:
            colors = base_colors[:num_clusters]
        
        # Create a mapping from cluster ID to color
        cluster_to_color = {cluster_id: colors[i % len(colors)] 
                            for i, cluster_id in enumerate(unique_clusters)}
        
        # Create point sizes based on data size for better visibility
        if len(feature_data) > 1000:
            point_size = 10
        elif len(feature_data) > 500:
            point_size = 20
        else:
            point_size = 30
            
        # 1. PCA Visualization
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(feature_data)
            
            plt.figure(figsize=(12, 10))
            for cluster_id in unique_clusters:
                # Skip cluster -1 which represents unclustered items
                if cluster_id == -1:
                    continue
                    
                mask = cluster_labels == cluster_id
                plt.scatter(
                    pca_result[mask, 0], 
                    pca_result[mask, 1],
                    s=point_size, 
                    color=cluster_to_color[cluster_id],
                    label=f'Cluster {cluster_id}'
                )
                
                # Add cluster centroid marker
                if np.any(mask):
                    centroid = np.mean(pca_result[mask], axis=0)
                    plt.scatter(
                        centroid[0], centroid[1],
                        s=point_size*4, 
                        color=cluster_to_color[cluster_id],
                        edgecolors='black',
                        linewidth=1.5,
                        marker='*'
                    )
                    
                    # Add cluster number label
                    plt.text(
                        centroid[0], centroid[1],
                        str(cluster_id),
                        fontsize=12,
                        ha='center', va='center',
                        weight='bold',
                        color='white',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
                    )
            
            # Add title and labels
            explained_var = pca.explained_variance_ratio_
            plt.title(f'PCA Visualization of Game Clusters\nExplained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}',
                     fontsize=14)
            plt.xlabel(f'Principal Component 1 ({explained_var[0]:.2f}%)', fontsize=12)
            plt.ylabel(f'Principal Component 2 ({explained_var[1]:.2f}%)', fontsize=12)
            
            # Add legend if there aren't too many clusters
            if num_clusters <= 20:
                plt.legend(loc='best', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.grid(alpha=0.3)
            plt.savefig(f'model/clustering_results/pca_clusters_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating PCA visualization: {e}")
        
        # 2. t-SNE Visualization
        try:
            # Use fewer perplexity for smaller datasets
            perplexity = min(30, max(5, len(feature_data) // 10))
            
            tsne = TSNE(
                n_components=2, 
                perplexity=perplexity,
                learning_rate='auto',
                init='random', 
                random_state=42
            )
            
            tsne_result = tsne.fit_transform(feature_data)
            
            plt.figure(figsize=(12, 10))
            for cluster_id in unique_clusters:
                # Skip cluster -1 which represents unclustered items
                if cluster_id == -1:
                    continue
                    
                mask = cluster_labels == cluster_id
                plt.scatter(
                    tsne_result[mask, 0], 
                    tsne_result[mask, 1],
                    s=point_size, 
                    color=cluster_to_color[cluster_id],
                    label=f'Cluster {cluster_id}'
                )
                
                # Add cluster centroid marker
                if np.any(mask):
                    centroid = np.mean(tsne_result[mask], axis=0)
                    plt.scatter(
                        centroid[0], centroid[1],
                        s=point_size*4, 
                        color=cluster_to_color[cluster_id],
                        edgecolors='black',
                        linewidth=1.5,
                        marker='*'
                    )
                    
                    # Add cluster information
                    if self.cluster_descriptions and cluster_id in self.cluster_descriptions:
                        # Get number of games in this cluster
                        count = self.cluster_descriptions[cluster_id]['count']
                        label = f"{cluster_id}\n({count})"
                    else:
                        label = str(cluster_id)
                    
                    plt.text(
                        centroid[0], centroid[1],
                        label,
                        fontsize=12,
                        ha='center', va='center',
                        weight='bold',
                        color='white',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
                    )
            
            # Add title and labels
            plt.title(f't-SNE Visualization of Game Clusters\nPerplexity: {perplexity}', fontsize=14)
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            
            # Add legend if there aren't too many clusters
            if num_clusters <= 20:
                plt.legend(loc='best', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.grid(alpha=0.3)
            plt.savefig(f'model/clustering_results/tsne_clusters_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Combined visualization (PCA + t-SNE side by side)
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # PCA plot
            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    continue
                    
                mask = cluster_labels == cluster_id
                axes[0].scatter(
                    pca_result[mask, 0], 
                    pca_result[mask, 1],
                    s=point_size, 
                    color=cluster_to_color[cluster_id],
                    label=f'Cluster {cluster_id}'
                )
            
            axes[0].set_title('PCA Projection', fontsize=14)
            axes[0].set_xlabel(f'PC1 ({explained_var[0]:.2f}%)', fontsize=12)
            axes[0].set_ylabel(f'PC2 ({explained_var[1]:.2f}%)', fontsize=12)
            axes[0].grid(alpha=0.3)
            
            # t-SNE plot
            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    continue
                    
                mask = cluster_labels == cluster_id
                axes[1].scatter(
                    tsne_result[mask, 0], 
                    tsne_result[mask, 1],
                    s=point_size, 
                    color=cluster_to_color[cluster_id],
                    label=f'Cluster {cluster_id}'
                )
            
            axes[1].set_title('t-SNE Projection', fontsize=14)
            axes[1].set_xlabel('t-SNE 1', fontsize=12)
            axes[1].set_ylabel('t-SNE 2', fontsize=12)
            axes[1].grid(alpha=0.3)
            
            # Single legend for both plots
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
                       ncol=min(5, len(unique_clusters)))
            
            plt.suptitle('Comparison of Dimensionality Reduction Techniques for Cluster Visualization', 
                         fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjust for the suptitle
            plt.savefig(f'model/clustering_results/combined_visualization_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating t-SNE visualization: {e}")