#!/usr/bin/env python3
"""
Anime Clustering Analysis
========================
This script performs cluster analysis on anime data to identify patterns and groupings
based on various features like ratings, popularity, genres, and other characteristics.

Usage:
    python main.py [csv_file_path]

If no file path is provided, it will look for 'anime_cleaned.csv' in the current directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Visualization styling
plt.style.use("default")
sns.set_palette("husl")


class AnimeClusterAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize the analyzer with the CSV file path."""
        self.csv_file_path = csv_file_path
        self.df = None
        self.df_features = None
        self.features_scaled = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.genre_cols = []

    def load_data(self):
        """Load and perform initial examination of the data."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            print(f"\nColumns: {list(self.df.columns)}")
            print(f"\nFirst few rows:")
            print(self.df.head())
            print(f"\nData info:")
            print(self.df.info())
            return True
        except FileNotFoundError:
            print(f"Error: File '{self.csv_file_path}' not found.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def preprocess_genres(self, genre_column="genres", separator=","):
        """
        Preprocess genre data by creating binary columns for each genre.

        Args:
            genre_column: Name of the column containing genre information
            separator: Character used to separate multiple genres
        """
        if genre_column not in self.df.columns:
            print(
                f"Warning: '{genre_column}' column not found. Available columns: {list(self.df.columns)}"
            )
            # Try to find genre-related columns
            genre_candidates = [
                col for col in self.df.columns if "genre" in col.lower()
            ]
            if genre_candidates:
                print(f"Found potential genre columns: {genre_candidates}")
                genre_column = genre_candidates[0]
                print(f"Using '{genre_column}' as genre column.")
            else:
                print("No genre columns found. Please specify the correct column name.")
                return False

        # Handle missing values
        self.df[genre_column] = self.df[genre_column].fillna("Unknown")

        # Extract all unique genres
        all_genres = set()
        for genres in self.df[genre_column]:
            if pd.notna(genres) and genres != "Unknown":
                all_genres.update([g.strip() for g in str(genres).split(separator)])

        print(f"Found {len(all_genres)} unique genres: {sorted(all_genres)}")

        # Create binary columns for each genre
        for genre in all_genres:
            col_name = f'genre_{genre.replace(" ", "_").replace("-", "_")}'
            self.df[col_name] = (
                self.df[genre_column].str.contains(genre, na=False).astype(int)
            )

        # Store genre column names
        self.genre_cols = [col for col in self.df.columns if col.startswith("genre_")]
        print(f"Created {len(self.genre_cols)} genre columns")
        return True

    def prepare_features(self):
        """Prepare features for clustering analysis."""
        # Define potential numeric features for anime data
        potential_features = [
            "score",
            "scored_by", 
            "rank",
            "popularity",
            "members",
            "favorites",
            "duration_min",
            "aired_from_year",
            "episodes"
        ]

        # Find available numeric features
        available_features = []
        for feature in potential_features:
            if feature in self.df.columns:
                try:
                    self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')
                    # Replace infinite values with NaN
                    self.df[feature] = self.df[feature].replace([np.inf, -np.inf], np.nan)
                    available_features.append(feature)
                except:
                    continue

        print(f"Available numeric features: {available_features}")

        # Combine with genre columns
        feature_columns = available_features + self.genre_cols

        # Create feature DataFrame
        self.df_features = self.df[feature_columns].copy()

        # Handle missing values
        for col in available_features:
            self.df_features[col] = pd.to_numeric(self.df_features[col], errors='coerce')
            # Use median for numeric features
            self.df_features[col] = self.df_features[col].fillna(self.df_features[col].median())

        # Standardize features
        self.features_scaled = self.scaler.fit_transform(self.df_features)
        print(f"Features prepared. Shape: {self.features_scaled.shape}")
        return True

    def find_optimal_clusters(self, max_k=15):
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        if self.features_scaled is None:
            print("Error: Features not prepared. Run prepare_features() first.")
            return None

        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []

        print("Finding optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.features_scaled)
            inertias.append(kmeans.inertia_)

            # Calculate silhouette score
            score = silhouette_score(self.features_scaled, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.3f}")

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Elbow plot
        ax1.plot(k_range, inertias, "bo-")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Method for Optimal k")
        ax1.grid(True)

        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, "ro-")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("analisis/cluster_optimization.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Suggest optimal k
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        print(
            f"\nSuggested optimal k based on silhouette score: {optimal_k_silhouette}"
        )

        return optimal_k_silhouette

    def perform_clustering(self, n_clusters=6):
        """Perform K-means clustering with specified number of clusters."""
        if self.features_scaled is None:
            print("Error: Features not prepared. Run prepare_features() first.")
            return False

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df["cluster"] = self.kmeans.fit_predict(self.features_scaled)

        print(f"Clustering completed with {n_clusters} clusters")
        print(f"Cluster distribution:")
        print(self.df["cluster"].value_counts().sort_index())

        return True

    def analyze_genre_importance(self):
        """Analyze genre importance based on various metrics specific to anime."""
        if not self.genre_cols:
            print("Error: No genre columns found. Run preprocess_genres() first.")
            return None

        importance_scores = {}

        for genre_col in self.genre_cols:
            genre_name = genre_col.replace("genre_", "").replace("_", " ")

            # Basic statistics
            total_count = self.df[genre_col].sum()
            frequency = total_count / len(self.df)

            if total_count == 0:
                continue

            # Get anime with this genre
            genre_anime = self.df[self.df[genre_col] == 1]

            # Calculate metrics
            avg_score = genre_anime["score"].mean() if "score" in self.df.columns else np.nan
            avg_members = genre_anime["members"].mean() if "members" in self.df.columns else np.nan
            avg_favorites = genre_anime["favorites"].mean() if "favorites" in self.df.columns else np.nan

            # Uniqueness (how often it appears alone)
            alone_count = self.df[
                (self.df[genre_col] == 1) & (self.df[self.genre_cols].sum(axis=1) == 1)
            ].shape[0]
            uniqueness = alone_count / total_count if total_count > 0 else 0

            # Cluster distribution
            cluster_distribution = genre_anime["cluster"].value_counts().to_dict()

            importance_scores[genre_name] = {
                "frequency": frequency,
                "total_count": total_count,
                "avg_score": avg_score,
                "avg_members": avg_members,
                "avg_favorites": avg_favorites,
                "uniqueness": uniqueness,
                # Convert cluster distribution to a string representation
                "cluster_distribution": str(cluster_distribution)
            }

        importance_df = pd.DataFrame(importance_scores).T

        # Sort by frequency (ascending - lowest first)
        importance_df = importance_df.sort_values("frequency")

        print("\nGenre Importance Analysis:")
        print("=" * 50)
        print(importance_df)

        return importance_df

    def identify_omittable_genres(self, min_frequency=0.05, min_count=10):
        """
        Identify genres that are candidates for omission.

        Args:
            min_frequency: Minimum frequency threshold (default: 5%)
            min_count: Minimum absolute count threshold
        """
        importance_df = self.analyze_genre_importance()
        if importance_df is None:
            return None

        # Find genres below thresholds
        low_frequency = importance_df[importance_df["frequency"] < min_frequency]
        low_count = importance_df[importance_df["total_count"] < min_count]

        omittable = pd.concat([low_frequency, low_count]).drop_duplicates()

        print(
            f"\nGenres candidates for omission (frequency < {min_frequency*100:.1f}% OR count < {min_count}):"
        )
        print("=" * 80)
        for genre, data in omittable.iterrows():
            print(
                f"• {genre}: {data['total_count']} anime ({data['frequency']*100:.1f}%)"
            )

        return omittable.index.tolist()

    def visualize_clusters(self):
        """Create visualizations for anime cluster analysis."""
        if self.df is None or "cluster" not in self.df.columns:
            print("Error: Clustering not performed. Run perform_clustering() first.")
            return

        # 1. PCA Visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features_scaled)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            features_pca[:, 0],
            features_pca[:, 1],
            c=self.df["cluster"],
            cmap="tab10",
            alpha=0.6,
        )
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.title("Anime Clusters (PCA Visualization)")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True, alpha=0.3)
        plt.savefig("analisis/cluster_pca_visualization.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2. Genre Distribution Heatmap
        if self.genre_cols:
            plt.figure(figsize=(15, 10))
            genre_cluster_matrix = self.df.groupby("cluster")[self.genre_cols].mean()

            # Clean genre names for display
            genre_names = [
                col.replace("genre_", "").replace("_", " ") for col in self.genre_cols
            ]
            genre_cluster_matrix.columns = genre_names

            sns.heatmap(
                genre_cluster_matrix.T,
                annot=True,
                cmap="YlOrRd",
                fmt=".2f",
                cbar_kws={"label": "Genre Frequency"},
            )
            plt.title("Genre Distribution Across Clusters")
            plt.xlabel("Cluster")
            plt.ylabel("Genre")
            plt.tight_layout()
            plt.savefig("analisis/genre_cluster_heatmap.png", dpi=300, bbox_inches="tight")
            plt.show()

        # 3. Cluster characteristics - Anime specific metrics
        numeric_cols = ["score", "members", "favorites", "episodes"]
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]

        if numeric_cols:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()

            for i, col in enumerate(numeric_cols[:4]):
                if i < len(axes):
                    self.df.boxplot(column=col, by="cluster", ax=axes[i])
                    axes[i].set_title(f"{col} by Cluster")
                    axes[i].set_xlabel("Cluster")

            plt.tight_layout()
            plt.savefig("analisis/cluster_characteristics.png", dpi=300, bbox_inches="tight")
            plt.show()

    def generate_report(self, output_file="analisis/anime_cluster_analysis_report.txt"):
        """Generate a comprehensive analysis report."""
        with open(output_file, "w") as f:
            f.write("ANIME GENRE CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset: {self.csv_file_path}\n")
            f.write(f"Total anime: {len(self.df)}\n")
            f.write(f"Total genres: {len(self.genre_cols)}\n")
            f.write(f"Number of clusters: {self.df['cluster'].nunique()}\n\n")

            # Cluster sizes
            f.write("Cluster Distribution:\n")
            cluster_counts = self.df["cluster"].value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                f.write(
                    f"  Cluster {cluster}: {count} anime ({count/len(self.df)*100:.1f}%)\n"
                )
            f.write("\n")

            # Genre analysis
            importance_df = self.analyze_genre_importance()
            if importance_df is not None:
                f.write("Genre Frequency Analysis:\n")
                for genre, data in importance_df.iterrows():
                    f.write(
                        f"  {genre}: {data['total_count']} anime ({data['frequency']*100:.1f}%)\n"
                    )
                f.write("\n")

                # Omittable genres
                omittable = self.identify_omittable_genres()
                if omittable:
                    f.write("Recommended genres for omission:\n")
                    for genre in omittable:
                        f.write(f"  • {genre}\n")

        print(f"Report generated: {output_file}")

    def clean_and_export_data(self, output_file="csv/myanimelist/anime_cleaned_filtered.csv"):
        """Clean the anime data by removing low-frequency genres and unwanted columns.
        Updates the genre column instead of using bit arrays.
        
        Args:
            output_file: Path to the output CSV file
        """
        # Get genres to remove
        omittable_genres = self.identify_omittable_genres()
        if not omittable_genres:
            return False

        print(f"\nRemoving {len(omittable_genres)} low-frequency genres...")
        
        # Create a copy of the original dataframe
        cleaned_df = self.df.copy()
        
        # Columns to remove
        columns_to_remove = [
            "opening_theme", "ending_theme", "airing", "aired_string",
            "background", "broadcast", "related", "licensor", "duration_min",
            "image_url", "title_synonyms",
            "cluster"  # Remove cluster column if it exists
        ]
        
        # Drop unwanted columns
        cleaned_df = cleaned_df.drop(columns=[col for col in columns_to_remove if col in cleaned_df.columns])
        
        # Filter anime based on air dates from the aired JSON column
        if 'aired' in cleaned_df.columns:
            
            def get_year_from_aired(aired_json):
                try:
                    aired_dict = json.loads(aired_json.replace("'", '"'))
                    from_date = aired_dict.get('from')
                    to_date = aired_dict.get('to')
                    
                    if from_date:
                        from_year = datetime.fromisoformat(from_date.replace('Z', '+00:00')).year
                        if to_date:
                            to_year = datetime.fromisoformat(to_date.replace('Z', '+00:00')).year
                        else:
                            to_year = from_year
                            
                        # Filter conditions:
                        # 1. Started airing in or after 2000
                        # 2. Not from 2018 (incomplete)
                        if from_year >= 2000 and from_year != 2018:
                            return True
                    return False
                except (json.JSONDecodeError, ValueError, AttributeError):
                    return False
            
            # Apply the filter
            original_count = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df['aired'].apply(get_year_from_aired)]
            removed_count = original_count - len(cleaned_df)
            print(f"\nRemoved {removed_count} anime that aired before 2000 or in 2018")
        
        # Update the genres column by removing omitted genres
        if 'genres' in cleaned_df.columns:
            def filter_genres(genre_string):
                if pd.isna(genre_string):
                    return ""
                genres = genre_string.split(',')
                filtered_genres = [g.strip() for g in genres if g.strip() not in omittable_genres]
                return ','.join(filtered_genres)
            
            cleaned_df['genres'] = cleaned_df['genres'].apply(filter_genres)
            
            # Remove rows with no remaining genres
            cleaned_df = cleaned_df[cleaned_df['genres'] != ""]
        
        # Drop all the genre_* columns
        genre_cols = [col for col in cleaned_df.columns if col.startswith('genre_')]
        cleaned_df = cleaned_df.drop(columns=genre_cols)
        
        # Export to CSV
        cleaned_df.to_csv(output_file, index=False)
        
        print(f"\nCleaned data exported to: {output_file}")
        print(f"Removed genres: {', '.join(omittable_genres)}")
        print(f"Removed columns: {', '.join(columns_to_remove)}")
        print(f"Original rows: {len(self.df)}")
        print(f"Remaining rows: {len(cleaned_df)}")
        print(f"Removed {len(self.df) - len(cleaned_df)} anime that had no remaining genres")
        return True


def main():
    """Main function to run the analysis."""
    # Get CSV file path from command line or use default
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "csv/myanimelist/anime_cleaned.csv"  # Update default path

    print("Anime Clustering Analysis")
    print("=" * 40)

    # Initialize analyzer
    analyzer = AnimeClusterAnalyzer(csv_file)

    # Load data
    if not analyzer.load_data():
        return

    # Ask user for genre column if needed
    print("\nGenre preprocessing...")
    genre_col = input("Enter the name of the genre column (or press Enter for 'genre'): ").strip()
    if not genre_col:
        genre_col = "genre"

    separator = input("Enter the separator for multiple genres (or press Enter for ','): ").strip()
    if not separator:
        separator = ","

    if not analyzer.preprocess_genres(genre_col, separator):
        return

    # Prepare features
    print("\nPreparing features...")
    if not analyzer.prepare_features():
        return

    # Find optimal clusters
    print("\nFinding optimal number of clusters...")
    optimal_k = analyzer.find_optimal_clusters()

    # Ask user for final k value
    user_k = input(f"\nEnter number of clusters to use (suggested: {optimal_k}): ").strip()
    try:
        final_k = int(user_k) if user_k else optimal_k
    except ValueError:
        final_k = optimal_k

    # Perform clustering
    print(f"\nPerforming clustering with k={final_k}...")
    if not analyzer.perform_clustering(final_k):
        return

    # Analyze genres
    print("\nAnalyzing genre importance...")
    analyzer.identify_omittable_genres()

    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_clusters()

    # Generate report
    print("\nGenerating report...")
    analyzer.generate_report()

    # Clean and export data
    print("\nCleaning and exporting filtered data...")
    analyzer.clean_and_export_data()

if __name__ == "__main__":
    main()
