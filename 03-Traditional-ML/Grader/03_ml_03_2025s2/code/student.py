import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self, file_path):
        # Initialization Attributes
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

        # Additional Attributes
        self.scaler = None
        self.model = None
        self.centroids = None
        self.normalized_centroids = None

    def Q1(self):
        """
        Q1: Please do the following operations
            -   Choose edible mushroom only.
            -   Only the variable below have been selected to describe
                the distinctive characteristics of edible mushrooms:
                -   cap-color-rate
                -   stalk-color-above-ring-rate
            -   Provide a proper data preprocessing as follows:
            -   Fill missing value with mean.
            -   Standardize variables with standard scaler.
            After these operations, please answer a shape after standardize variables.
        """
        # Select rows labeled as edible mushroom.
        self.df = self.df[self.df["label"] == "e"]

        # Select columns
        self.df = self.df[["cap-color-rate", "stalk-color-above-ring-rate"]]

        # Fill missing value with mean
        self.df = self.df.fillna(self.df.mean())

        # Initialize StandardScaler object
        self.scaler = StandardScaler()

        # Data normalization
        normalized_arr = self.scaler.fit_transform(self.df)

        # Convert the numpy array back to DataFrame
        self.df = pd.DataFrame(
            normalized_arr, columns=self.df.columns, index=self.df.index
        )

        # Return the shape of the normalized DataFrame
        return self.df.shape

    def Q2(self):
        """
        Q2: Please do the following operations
            -   Do all operations from Q1
            -   Implement K-means clustering with 5 clusters with following:
                -   Set n_cluster to 5
                -   Set random_state to 0
                -   Set n_init to "auto"
            Show the maximum centroid of 2 features in two decimal places.
            -   cap-color-rate
            -   stalk-color-above-ring-rate
        """
        # Ensure that all operations in Q1 are executed.
        self.Q1()

        # Initialize K-means clustering model.
        self.model = KMeans(n_clusters=5, random_state=0, n_init="auto")

        # Begin clustering on the dataset.
        self.model.fit(self.df)

        # Calculate the maximum normalized cluster centroids.
        self.normalized_centroids = self.model.cluster_centers_
        max_centroids = np.max(self.normalized_centroids, axis=0)

        # Rounded to 2 decimal places
        max_centroids = np.round(max_centroids, 2)

        # Return maximum centroids.
        return max_centroids

    def Q3(self):
        """
        Q3: Please do the following operations
        -   Do all operations from `Q1` and `Q2`.
        -   Convert the centrioid value to the original scale.
        Show the minimum centroid of 2 features rounded in 2 decimal places.
        """
        # Ensure that all operations in Q1 to Q2 are executed.
        self.Q2()

        # Scale back to the original scale.
        self.centroids = self.scaler.inverse_transform(self.normalized_centroids)

        # Calculate the minimum cluster centroids.
        min_centroids = np.min(self.centroids, axis=0)

        # Rounded to 2 decimal places
        min_centroids = np.round(min_centroids, 2)

        # Return minimum centroids
        return min_centroids
