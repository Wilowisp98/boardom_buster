# -*- coding: utf-8 -*-
import os
import time
import pickle

import numpy as np
import polars as pl

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .utils import *
from .configs import *

# TO DO:
# - Change prints to logging.
# - Add error handling.
# - Refactor plotting.

class bgClusters:
    """
    Class for clustering board games based on feature data with support for constrained clustering.
    """
    
    def __init__(self):
        self.clusters = {}
        self.clusters_descriptions = {}
        self.constraint_columns = []
    
    def fit(self, games_df, constraint_columns, name_column: str = None, restart_model: bool = False, plot: bool = PLOT):
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
        final_model_dir : str, optional
            Directory to store clustering results
        model_filename : str, optional
            Name of the cache file to use
        restart_mode: bool, optional
            If the model should be rebuilt. Ex: There is new data.
            
        Returns:
        --------
        self : bgClusters
            The fitted model instance
        """
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        model_file = os.path.join(FINAL_MODEL_DIR, MODEL_FILENAME)

        if os.path.exists(model_file) and not restart_model:
            print(f"Loading cached clustering results from {model_file}")

            with open(model_file, 'rb') as f:
                saved_model = pickle.load(f)

                if 'cluster_to_games' in saved_model:
                    for cluster_id, game_names in saved_model['cluster_to_games'].items():
                        self.clusters[cluster_id] = {
                            'constraint': f"cluster_{cluster_id}",
                            'game_names': game_names,
                            'count': len(game_names)
                        }

                    self.games_df = games_df
                    self.name_column = name_column
                    self.feature_columns = get_feature_columns(self.games_df, RELEVANT_COLUMNS)
                    self._add_cluster_labels()
                    self._generate_cluster_descriptions()
                    self.constraint_columns = constraint_columns
                    return self

        self.feature_columns = get_feature_columns(games_df, RELEVANT_COLUMNS)
        # self.games_df = games_df.select(self.feature_columns).to_numpy()
        self.games_df = games_df.select(self.feature_columns + [name_column] if name_column else self.feature_columns)
        # self.games_df = games_df.select(self.feature_columns)
        # self.unconstrained_df = self.games_df.drop(name_column)
        self.unconstrained_df = self.games_df
        self.feature_count = self.games_df.shape[1]
        self.name_column = name_column
        self.constraint_columns = constraint_columns

        for column in self.constraint_columns:
            # constrained_df = self.games_df.filter(pl.col(column) == 1).drop(name_column)
            constrained_df = self.games_df.filter(pl.col(column) == 1)
            self.unconstrained_df = self.unconstrained_df.filter(pl.col(column) != 1)
            self._create_constrained_clusters(constrained_df, constraint_name=column.split('GAME_CAT_GROUP_')[1], plot=plot)

        self._create_constrained_clusters(self.unconstrained_df, plot=plot)
        self._add_cluster_labels()
        self._generate_cluster_descriptions()
        if plot:
            self._visualize_clustering_results()
        self._save(model_file)
        return self
    
    def _generate_cluster_descriptions(self):
        """
        Generate descriptions for each cluster based on feature importance.
        
        Returns:
        --------
        dict
            Dictionary containing descriptions for each cluster
        """
        
        for cluster_id, cluster_info in self.clusters.items():
            cluster_games = self.games_df.filter(pl.col("cluster") == cluster_id)
            if cluster_games.height == 0:
                continue
                
            cluster_means = {}
            for feature in self.feature_columns:
                cluster_means[feature] = cluster_games.select(pl.col(feature)).mean().item()
            
            sorted_features = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, _ in sorted_features[:5]]
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
            
            if self.name_column:
                sample_games = cluster_games.select(pl.col(self.name_column)).to_series().to_list()[:5]
            else:
                sample_games = [f"Game #{i}" for i in range(min(5, cluster_games.height))]
            
            self.clusters_descriptions[cluster_id] = {
                'description': description,
                'constraint': cluster_info['constraint'],
                'is_subcluster': cluster_info.get('is_subcluster', False),
                'parent_constraint': cluster_info.get('parent_constraint', ''),
                'top_features': top_features,
                'count': cluster_games.height,
                'sample_games': sample_games
            }
    
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
    
    def _create_constrained_clusters(self, data_to_cluster, constraint_name: str = None, plot: bool = False):
        """
        Create clusters based on constraint columns.
        
        Parameters:
        -----------
        data_to_cluster : polars.DataFrame
            DataFrame containing board game features to cluster
        is_constraint : bool, optional
            Whether this clustering is based on a constraint
        constraint_name : str, optional
            Name of the constraint
        """
        cluster_id = len(self.clusters)
        data_to_cluster_features = data_to_cluster.drop(self.name_column)
        optimal_k = self._find_optimal_k(
            data=data_to_cluster_features,
            constraint_name=constraint_name,
            plot=plot
        )

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_to_cluster_features)

        for sub_id in range(optimal_k):
            sub_indices = [i for i in range(len(labels)) if labels[i] == sub_id]

            if sub_indices:
                if self.name_column and self.name_column in data_to_cluster.columns:
                    game_ids = data_to_cluster.select(pl.col(self.name_column)).to_series().to_list()
                    sub_games = [game_ids[i] for i in sub_indices]
                else:
                    sub_games = sub_indices

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
                    'game_names': sub_games,
                    'indices': sub_indices,
                    'count': len(sub_indices),
                    'is_subcluster': is_subcluster,
                    'subcluster_id': sub_id
                }
                cluster_id += 1

    def _find_optimal_k(self, data, constraint_name: str = '', plot: bool = False):
        """
        Find the optimal number of clusters using multiple evaluation metrics with adaptive size constraints.
        Includes a complexity penalty to prevent selecting too many clusters.

        Parameters:
        -----------
        data : numpy.ndarray
            Feature data to cluster
        constraint_name : str, optional
            Name of the constraint for saving results

        Returns:
        --------
        int
            Optimal number of clusters
        """
        data_size = len(data)
        adaptive_min_k = max(2, int(np.log(data_size)))
        adaptive_max_k = max(5, int(np.sqrt(data_size)) / 2)
        max_k = min(self.feature_count, data_size - 1, adaptive_max_k * 2)

        k_range = range(2, int(max_k * 2))
        silhouette_scores = []
        cluster_size_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            if data_size > k:
                sil_score = silhouette_score(data, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)

            if adaptive_min_k <= k <= max_k:
                cluster_size_scores.append(1.0)
            else:
                distance = min(abs(k - adaptive_min_k), abs(k - max_k))
                max_reasonable = int(np.sqrt(data_size) / 2)
                score = max(0, 1.0 - (distance / max(1, max_reasonable)))
                cluster_size_scores.append(score)

        norm_silhouette = normalize(silhouette_scores, higher_is_better=True)
        norm_cluster_size = cluster_size_scores

        complexity_penalty = []
        penalty_weight = 0.1
        max_k_value = max(k_range)
        for k in k_range:
            penalty = penalty_weight * (k / max_k_value)**2
            complexity_penalty.append(penalty)

        weights = {
            'silhouette': 0.65,
            'cluster_size': 0.35 
        }

        final_scores = []
        final_scores_with_penalty = []
        for i in range(len(k_range)):
            score = (
                weights['silhouette'] * norm_silhouette[i] +
                weights['cluster_size'] * norm_cluster_size[i]
            )
            final_scores.append(score)
            penalized_score = score - complexity_penalty[i]
            final_scores_with_penalty.append(penalized_score)

        best_idx = np.argmax(final_scores_with_penalty)
        optimal_k = k_range[best_idx]

        if plot:
            # VISUALIZATION 0: Optimization process
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})

            # Top plot: Detailed progression of each k evaluation
            ax = axes[0]
            x = np.arange(len(k_range))
            width = 0.3
            offsets = [-0.5, 0.5]

            # Plot bars for each metric side by side
            ax.bar(x + offsets[0]*width, norm_silhouette, width, label='Silhouette', color='blue', alpha=0.7)
            ax.bar(x + offsets[1]*width, norm_cluster_size, width, label='Cluster Size', color='orange', alpha=0.7)

            # Add line showing weighted total with and without penalty
            ax.plot(x, final_scores, 'ko-', linewidth=2, label='Original Score')
            ax.plot(x, final_scores_with_penalty, 'ro-', linewidth=2, label='Score with Penalty')

            # Add complexity penalty visualization
            ax.plot(x, complexity_penalty, 'g--', linewidth=1.5, label='Complexity Penalty')

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
                ('Cluster Size', norm_cluster_size, 'orange', 'p-'),
                ('Original Score', final_scores, 'black', '*-'),
                ('Score with Penalty', final_scores_with_penalty, 'red', 'x-'),
                ('Complexity Penalty', complexity_penalty, 'green', '--')
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
            fig.savefig(f'{MODEL_DETAILS_DIR}/optimization_process_{constraint_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # VISUALIZATION 1: Individual metrics (original)
            fig = plt.figure(figsize=(12, 6))
            plt.plot(k_range, norm_silhouette, 'o-', label='Silhouette Score (normalized)', color='blue')
            plt.plot(k_range, norm_cluster_size, 'p-', label='Cluster Size Score', color='orange')
            plt.plot(k_range, complexity_penalty, '--', label='Complexity Penalty', color='green')
            plt.axvline(x=optimal_k, color='black', linestyle='--', label=f'Optimal k = {optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Normalized Score')
            plt.title('Individual Evaluation Metrics for Cluster Optimization')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            # Save to file
            fig.savefig(f'{MODEL_DETAILS_DIR}/individual_metrics_{constraint_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # VISUALIZATION 2: Combined scores with iteration markers
            fig = plt.figure(figsize=(10, 5))
            plt.plot(k_range, final_scores, 'o-', color='blue', label='Original Score')
            plt.plot(k_range, final_scores_with_penalty, 'o-', color='red', label='Score with Penalty')

            # Add arrows showing the progression of testing
            for i in range(len(k_range)-1):
                plt.annotate('', 
                    xy=(k_range[i+1], final_scores_with_penalty[i+1]),
                    xytext=(k_range[i], final_scores_with_penalty[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5))

            # Add iteration numbers
            for i in range(len(k_range)):
                plt.text(k_range[i], final_scores_with_penalty[i] + 0.02, f"{i+1}", 
                        ha='center', va='bottom', fontsize=9, color='darkblue')

            plt.axvline(x=optimal_k, color='black', linestyle='--', label=f'Optimal k = {optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Score')
            plt.title('Combined Cluster Evaluation Scores with Iteration Path')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            # Save to file
            fig.savefig(f'{MODEL_DETAILS_DIR}/combined_scores_{constraint_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Print iterations summary
            print("\nClustering Optimization Iterations:")
            print("-" * 60)
            print(f"{'k':>3} {'Silhouette':>12} {'Size':>12} {'Penalty':>12} {'Original':>12} {'Penalized':>12}")
            print("-" * 60)
            for i, k in enumerate(k_range):
                print(f"{k:3d} {norm_silhouette[i]:12.4f} "
                      f"{norm_cluster_size[i]:12.4f} "
                      f"{complexity_penalty[i]:12.4f} "
                      f"{final_scores[i]:12.4f} "
                      f"{final_scores_with_penalty[i]:12.4f}")
            print("-" * 60)
            print(f"Selected optimal k = {optimal_k} (data size: {data_size})\n")

        return optimal_k
    
    def _add_cluster_labels(self):
        """
        Add cluster labels to the DataFrame.
        """
        cluster_labels = [-1] * self.games_df.height

        if self.name_column and self.name_column in self.games_df.columns:
            game_names = self.games_df.select(pl.col(self.name_column)).to_series().to_list()
            name_to_idx = {name: idx for idx, name in enumerate(game_names)}

            for cluster_id, cluster_info in self.clusters.items():
                for game_name in cluster_info.get('game_names', []):
                    if game_name in name_to_idx:
                        idx = name_to_idx[game_name]
                        cluster_labels[idx] = cluster_id
        else:
            for cluster_id, cluster_info in self.clusters.items():
                for idx in cluster_info.get('indices', []):
                    if 0 <= idx < len(cluster_labels):
                        cluster_labels[idx] = cluster_id

        self.games_df = self.games_df.with_columns([
            pl.Series("cluster", cluster_labels)
        ])
    
    #############################################
    # CACHING AND PERSISTENCE
    #############################################

    def _save(self, model_file):
        """
        Save minimal clustering results to cache file - only cluster numbers and game names.

        Parameters:
        -----------
        cache_file : str
            Path to the cache file
        """
        minimal_clusters = {}

        for cluster_id, cluster_info in self.clusters.items():
            game_names = cluster_info.get('game_names', [])
            minimal_clusters[cluster_id] = game_names

        simplified_cache_data = {
            'cluster_to_games': minimal_clusters,
            'creation_timestamp': time.time()
        }

        with open(model_file, 'wb') as f:
            pickle.dump(simplified_cache_data, f)

        print(f"Saved simplified cluster data to {model_file}")

    def _visualize_clustering_results(self):
        """
        Generate TSNE and PCA visualizations of the clustering results.
        Saves visualizations to the clustering_results directory.
        """
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
            plt.savefig(f'{MODEL_DETAILS_DIR}/pca_clusters_{timestamp}.png', dpi=300, bbox_inches='tight')
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
            plt.savefig(f'{MODEL_DETAILS_DIR}/combined_visualization_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating t-SNE visualization: {e}")