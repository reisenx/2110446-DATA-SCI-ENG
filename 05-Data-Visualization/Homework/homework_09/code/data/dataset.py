import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from data.dataset_utility import DatasetUtility


class Dataset:
    def __init__(self):
        """
        Constructor method of the Dataset class
        """

        # The original dataset
        self.df = Dataset._load_data()

        # The filtered dataset
        self.price_range = [-1, -1]
        self.df_by_price_range = self.df.copy()
        self.df_by_neighborhood = self.df.copy()

        # The clustered dataset
        self.eps = -1
        self.min_samples = -1
        self.clustered_price_range = [-1, -1]
        self.df_clustered = self.df.copy()

        # The visualize clustered dataset
        self.n_top_clusters = -1
        self.top_cluster_ids = None
        self.df_visualize_clustered = self.df.copy()

        # The KDE dataset
        self.bandwidth = -1
        self.kde_price_range = [-1, -1]
        self.df_kde = self.df.copy()

    @staticmethod
    @st.cache_data
    def _load_data():
        """
        Returns a cleaned Pandas DataFrame read from the CSV file.

        Returns:
            pd.DataFrame: cleaned DataFrame
        """

        # Read the CSV file
        FILE_PATH = Path.cwd().joinpath("data", "airbnb_listings.csv")
        df = pd.read_csv(FILE_PATH)

        # Clean the DataFrame
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude", "price"])

        # Return the cleaned DataFrame
        return df

    def set_by_price_range_data(self, price_begin, price_end):
        """
        Set a filtered DataFrame where prices are within the specified range

        Args:
            price_begin (float): minimum price in the range
            price_end (float): maximum price in the range
        """

        # Skip when the filtered price range has not changed
        if [price_begin, price_end] == self.price_range:
            return

        # Update the filtered DataFrame
        COLUMN_NAME = "price"
        self.price_range = [price_begin, price_end]
        self.df_by_price_range = self.df[
            (self.df[COLUMN_NAME] >= price_begin) & (self.df[COLUMN_NAME] <= price_end)
        ]

    def set_by_neighborhood_data(self, price_begin, price_end):
        """
        Set a DataFrame that grouped by the neighborhoods from a filtered DataFrame
        where prices are within the specified range

        Args:
            price_begin (float): minimum price in the range
            price_end (float): maximum price in the range
        """

        self.set_by_price_range_data(price_begin, price_end)
        self.df_by_neighborhood = (
            self.df_by_price_range.groupby("neighbourhood")["price"]
            .agg(["mean", "count"])
            .reset_index()
        )
        self.df_by_neighborhood.columns = [
            "neighbourhood",
            "avg_price",
            "listings_count",
        ]

    def set_clustered_data(self, price_begin, price_end, eps, min_samples):
        """
        Performs DBSCAN clustering to a filtered DataFrame where prices are within
        the specified range

        Args:
            price_begin (float): minimum price in the range
            price_end (float): maximum price in the range
            eps (float): radius from the center data point
            min_samples (int): minimum amount of data points for a cluster
        """

        # Skip when the price range and the clustering parameters has not changed
        if [price_begin, price_end] == self.clustered_price_range and [
            eps,
            min_samples,
        ] == [
            self.eps,
            self.min_samples,
        ]:
            return

        # Update the filtered DataFrame by price range
        self.set_by_price_range_data(price_begin, price_end)

        # Perform clustering by coordinates
        COLUMN_NAMES = ["latitude", "longitude"]
        coordinates = self.df_by_price_range[COLUMN_NAMES]
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

        # Update the clustered DataFrame
        COLUMN_NAME = "cluster"
        self.eps = eps
        self.min_samples = min_samples
        self.clustered_price_range = [price_begin, price_end]
        self.df_clustered = self.df_by_price_range.copy()
        self.df_clustered[COLUMN_NAME] = db.labels_

        # Remove noise data points from a DataFrame
        self.df_clustered = self.df_clustered[self.df_clustered[COLUMN_NAME] != -1]

    def set_visualize_clustered_data(
        self, price_begin, price_end, eps, min_samples, n_top_clusters
    ):
        """
        Performs DBSCAN clustering to a filtered DataFrame where prices are within
        the specified range.
        Then select the cluster assigned with different color for visualization.

        Args:
            price_begin (float): minimum price in the range
            price_end (float): maximum price in the range
            eps (float): radius from the center data point
            min_samples (int): minimum amount of data points for a cluster
            n_top_clusters (int): amount of clusters to visualize
        """

        # Update the clustered DataFrame
        self.set_clustered_data(price_begin, price_end, eps, min_samples)

        # Get cluster IDs to visualize
        self.top_cluster_ids = DatasetUtility.get_top_cluster_ids(self, n_top_clusters)

        # Update the visualize clustered DataFrame
        COLUMN_NAME = "cluster"
        self.n_top_clusters = n_top_clusters
        self.df_visualize_clustered = self.df_clustered[
            self.df_clustered[COLUMN_NAME].isin(self.top_cluster_ids)
        ].copy()

        # Add color column to the visualization DataFrame
        COLUMN_NAME = "cluster"
        NEW_COLUMN_NAME = "color"
        self.df_visualize_clustered[NEW_COLUMN_NAME] = self.df_visualize_clustered[
            COLUMN_NAME
        ].map(DatasetUtility.get_cluster_color_map(self, self.top_cluster_ids))

    def set_kde_data(self, price_begin, price_end, bandwidth):
        """
        Perform Kernel Density Estimation on the dataset to a filtered DataFrame where
        prices are within the specified range.
        Each data point has its own color according to the density value.

        Args:
            price_begin (float): minimum price in the range
            price_end (float): maximum price in the range
            bandwidth (float): bell curve width on individual data point
        """

        # Skip when the price range and the clustering parameters has not changed
        if [
            price_begin,
            price_end,
        ] == self.kde_price_range and bandwidth == self.bandwidth:
            return

        # Update the filtered DataFrame by price range
        self.set_by_price_range_data(price_begin, price_end)

        # Fit KDE model by coordinates
        COLUMN_NAMES = ["latitude", "longitude"]
        coordinates = self.df_by_price_range[COLUMN_NAMES].values
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(coordinates)

        # Calculate density score for each data point
        COLUMN_NAME = "density"
        self.bandwidth = bandwidth
        self.kde_price_range = [price_begin, price_end]
        self.df_kde = self.df_by_price_range.copy()
        self.df_kde[COLUMN_NAME] = np.exp(kde.score_samples(coordinates))

        # Calculate normalized density
        COLUMN_NAME = "density"
        NEW_COLUMN_NAME = "density_normalized"
        self.df_kde[NEW_COLUMN_NAME] = (
            self.df_kde[COLUMN_NAME] - self.df_kde[COLUMN_NAME].min()
        ) / (self.df_kde[COLUMN_NAME].max() - self.df_kde[COLUMN_NAME].min())

        # Get color from density of each data point for visualization
        COLUMN_NAME = "density_normalized"
        NEW_COLUMN_NAME = "density_color"
        self.df_kde[NEW_COLUMN_NAME] = self.df_kde[COLUMN_NAME].apply(
            DatasetUtility.get_color_from_density
        )
