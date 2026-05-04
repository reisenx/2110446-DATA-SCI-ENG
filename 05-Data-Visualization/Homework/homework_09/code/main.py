import streamlit as st
from data.dataset import Dataset
from plot.plot import Plot


class Main:
    @staticmethod
    def main():
        """
        Main Application Logic
        """

        # Setup the streamlit page
        st.set_page_config(page_title="Bangkok Airbnb Analysis", layout="wide")

        # Initialize dataset
        if "dataset" not in st.session_state:
            st.session_state.dataset = Dataset()
        dataset = st.session_state.dataset

        # Display page title
        st.title("Bangkok Airbnb Listings Analysis")

        # Display page sidebar and get user parameters
        user_parameters = Main.display_sidebar()

        # Setup dataset from the user parameters
        try:
            dataset = Main.setup_dataset(dataset, user_parameters)
        except Exception as e:
            st.error(f"[ERROR]: Could not setup the dataset {e}")

        # Display Key metrics
        Plot.display_overall_stats(dataset)

        # Display price distribution histogram
        st.header("Price Distribution")
        Plot.display_price_histogram(dataset)

        # Display price by neighborhood scatter plot
        Plot.display_price_by_neighborhood(dataset)

        # Display Hot spot analysis
        st.header("Accommodation Hotspot Analysis")
        Plot.display_map_cluster_scatter_plot(dataset, user_parameters["map_style"])
        Plot.display_map_heatmap(dataset, user_parameters["map_style"])
        Plot.display_map_hexagon_heatmap(dataset, user_parameters["map_style"])

        # Display cluster statistics
        st.subheader("Cluster Statistics")
        Plot.display_all_clusters_stats(dataset, user_parameters["n_top_clusters"])

        # Display Kernel Density Estimation Analysis
        st.header("Kernel Density Estimation (KDE) Analysis")
        Plot.display_all_kde_stats(dataset, user_parameters["map_style"])

    @staticmethod
    def display_sidebar():
        """
        Display a sidebar of the Streamlit page which is used to
        input the user's parameters.

        Returns:
            dict[str, int|float]: contains all user input parameters
        """

        st.sidebar.header("Filters")
        price_range = Plot.display_price_range_slider()

        st.sidebar.header("DBSCAN Parameters")
        eps = Plot.display_eps_slider()
        min_samples = Plot.display_min_samples_slider()
        n_top_clusters = Plot.display_n_top_cluster_slider()

        map_style = Plot.display_map_style_options()

        st.sidebar.header("KDE Parameters")
        bandwidth = Plot.display_bandwidth_slider()

        return {
            "price_range": price_range,
            "eps": eps,
            "min_samples": min_samples,
            "n_top_clusters": n_top_clusters,
            "map_style": map_style,
            "bandwidth": bandwidth,
        }

    @staticmethod
    def setup_dataset(dataset: Dataset, user_parameters) -> Dataset:
        """
        Setup the dataset using the user's parameters which is configured
        on the page sidebar

        Args:
            dataset (Dataset): a Dataset object of the current instance
            user_parameters (dict[str, int|float]): contains all user input parameters

        Returns:
            Dataset: modified Dataset object
        """

        dataset.set_by_price_range_data(
            price_begin=user_parameters["price_range"][0],
            price_end=user_parameters["price_range"][1],
        )

        dataset.set_by_neighborhood_data(
            price_begin=user_parameters["price_range"][0],
            price_end=user_parameters["price_range"][1],
        )

        dataset.set_clustered_data(
            price_begin=user_parameters["price_range"][0],
            price_end=user_parameters["price_range"][1],
            eps=user_parameters["eps"],
            min_samples=user_parameters["min_samples"],
        )

        dataset.set_visualize_clustered_data(
            price_begin=user_parameters["price_range"][0],
            price_end=user_parameters["price_range"][1],
            eps=user_parameters["eps"],
            min_samples=user_parameters["min_samples"],
            n_top_clusters=user_parameters["n_top_clusters"],
        )

        dataset.set_kde_data(
            price_begin=user_parameters["price_range"][0],
            price_end=user_parameters["price_range"][1],
            bandwidth=user_parameters["bandwidth"],
        )
        return dataset


if __name__ == "__main__":
    Main.main()
