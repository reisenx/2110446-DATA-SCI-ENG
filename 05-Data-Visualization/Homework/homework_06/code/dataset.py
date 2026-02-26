import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st


class Dataset:
    def __init__(self):
        # Load Iris Dataset as a Pandas DataFrame
        df, iris_feature_names, iris_target, iris_target_names = Dataset._load_data()
        self.df = df
        self.df["Species"] = pd.Categorical.from_codes(iris_target, iris_target_names)

        # Store features and their names
        self.feature_names = iris_feature_names
        self.X = self.df[self.feature_names].values

        # Store color for each species
        self.species_color = {
            "setosa": "#FF4B4B",
            "versicolor": "#4B4BFF",
            "virginica": "#4BFF4B",
        }

        # Create scaler for normalization
        self.scaler = StandardScaler()

    @staticmethod
    @st.cache_data
    def _load_data():
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        return df, iris.feature_names, iris.target, iris.target_names

    def get_scaled_data(self):
        return self.scaler.fit_transform(self.X)

    def get_correlation(self):
        return np.round(self.df[self.feature_names].corr(), 2)

    def get_elbow_analysis(self, max_clusters):
        inertias = []
        for k in range(1, max_clusters + 1):
            k_means = KMeans(n_clusters=k, random_state=42)
            k_means.fit(self.X)
            inertias.append(k_means.inertia_)
        return inertias

    def perform_clustering(self, n_clusters):
        k_means = KMeans(n_clusters=n_clusters, random_state=42)

        cluster_label = k_means.fit_predict(self.get_scaled_data())
        self.df["Cluster"] = cluster_label.astype(str)

    def get_confusion_matrix(self):
        return pd.crosstab(self.df["Species"], self.df["Cluster"], margins=True)

    def get_species_stats(self):
        return (
            self.df.groupby("Species", observed=True)[self.feature_names]
            .agg(["mean", "std"])
            .round(2)
        )

    def get_cluster_stats(self):
        return (
            self.df.groupby("Cluster", observed=True)[self.feature_names]
            .agg(["mean", "std"])
            .round(2)
        )
