import pydeck as pdk
import plotly.express as px
import streamlit as st
from config.config import Config
from data.dataset import Dataset
from data.dataset_stats import DatasetStats
from data.dataset_utility import DatasetUtility


class Plot:
    @staticmethod
    def display_price_range_slider():
        """
        Display a Streamlit slider for price range parameter on a page for user to adjust.
        This is for dataset filtering.

        Returns:
            tuple[int, int]: contains a price range in baht
        """

        return st.sidebar.slider(
            label="Price Range (THB)",
            min_value=128,
            max_value=40000,
            value=(128, 1000),
            step=1,
        )

    @staticmethod
    def display_eps_slider():
        """
        Display a Streamlit slider for eps parameter on a page for user to adjust.
        eps is a radius from the center data point in DBSCAN clustering

        Returns:
            float: eps value as a DBSCAN parameter
        """

        return st.sidebar.slider(
            label="eps (degree)",
            min_value=0.001,
            max_value=0.005,
            value=0.002,
            step=0.001,
            format="%0.3f",
        )

    @staticmethod
    def display_min_samples_slider():
        """
        Display a Streamlit slider for min_samples parameter on a page for user to adjust
        min_samples is a minimum amount of data points for a cluster in DBSCAN clustering

        Returns:
            int: min_samples as a DBSCAN parameter
        """

        return st.sidebar.slider(
            label="min_samples", min_value=2, max_value=10, value=3, step=1
        )

    @staticmethod
    def display_n_top_cluster_slider():
        """
        Display a Streamlit slider for n_top_cluster parameter on a page for user to adjust.
        n_top_cluster is an amount of DBSCAN cluster to display on a page

        Returns:
            int: n_top_cluster as DBSCAN parameter
        """

        return st.sidebar.slider(
            label="Number of Top Cluster to Show",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )

    @staticmethod
    def display_bandwidth_slider():
        """
        Display a Streamlit slider for bandwidth parameter on a page for user to adjust.
        bandwidth is a width of a bell curve on each data point for KDE.

        Returns:
            float: bandwidth as KDE parameter
        """

        return st.sidebar.slider(
            label="Bandwidth",
            min_value=0.001,
            max_value=0.020,
            value=0.005,
            step=0.001,
            format="%0.3f",
        )

    @staticmethod
    def display_map_style_options():
        """
        Display a Streamlit select box for a map box style on a page for user to adjust

        Returns:
            str: map box style
        """

        return st.sidebar.selectbox(
            "Select Base Map Style",
            options=["Dark", "Light", "Road", "Satellite"],
            index=0,
        )

    @staticmethod
    def display_overall_stats(dataset: Dataset):
        """
        Display key metrics of a filtered dataset on a page.

        Args:
            dataset (Dataset): a Dataset object for the current instance
        """

        N_ROWS = dataset.df_by_price_range.shape[0]
        MEAN_PRICE = DatasetStats.get_by_price_mean(
            dataset=dataset, column_name="price"
        )
        MEAN_REVIEW = DatasetStats.get_by_price_mean(
            dataset=dataset, column_name="number_of_reviews"
        )
        N_NEIGHBORHOODS = DatasetStats.get_by_price_n_unique(
            dataset=dataset, column_name="neighbourhood"
        )

        DISPLAY_INFO = (
            {"text": "Total Listings", "value": f"{N_ROWS:,}"},
            {"text": "Average Price", "value": f"฿{MEAN_PRICE:,.0f}"},
            {"text": "Average Reviews", "value": f"{MEAN_REVIEW:,.1f}"},
            {"text": "Neighborhoods", "value": f"{N_NEIGHBORHOODS}"},
        )

        columns = st.columns(len(DISPLAY_INFO))
        for idx, column in enumerate(columns):
            with column:
                st.metric(
                    DISPLAY_INFO[idx]["text"],
                    DISPLAY_INFO[idx]["value"],
                )

    @staticmethod
    def display_price_histogram(dataset: Dataset):
        """
        Display a histogram of the price distribution of the filtered dataset

        Args:
            dataset (Dataset): a Dataset object for the current instance
        """

        historgram = px.histogram(
            dataset.df_by_price_range,
            x="price",
            nbins=100,
            title="Distribution of Listing Prices",
            labels={"price": "Price (THB)", "count": "Number of Listings"},
        )

        st.plotly_chart(historgram)

    @staticmethod
    def display_price_by_neighborhood(dataset: Dataset):
        """
        Display a scatter plot of the average price of each neighborhood.

        Args:
            dataset (Dataset): a Dataset object for the current instance
        """

        fig_scatter = px.scatter(
            dataset.df_by_neighborhood,
            x="listings_count",
            y="avg_price",
            text="neighbourhood",
            title="Average Price vs Number of Listings by Neighborhood",
            labels={
                "listings_count": "Number of Listings",
                "avg_price": "Average Price (THB)",
            },
        )

        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter)

    @staticmethod
    def display_map_cluster_scatter_plot(dataset: Dataset, map_style):
        """
        Display a scatter of DBSCAN clusters overlay on an actual map

        Args:
            dataset (Dataset): a Dataset object for the current instance
            map_style (str): map box type
        """

        cluster_layer = pdk.Layer(
            "ScatterplotLayer",
            dataset.df_visualize_clustered,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=50,
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                api_keys=Config.API_KEYS,
                layers=[cluster_layer],
                initial_view_state=pdk.ViewState(
                    latitude=DatasetStats.get_by_price_mean(
                        dataset=dataset, column_name="latitude"
                    ),
                    longitude=DatasetStats.get_by_price_mean(
                        dataset=dataset, column_name="longitude"
                    ),
                    zoom=11,
                    pitch=0,
                ),
                map_style=Config.MAP_STYLES[map_style],
                tooltip={
                    "html": "<b>Cluster:</b> {cluster}<br/>"
                    "<b>Price:</b> ฿{price}<br/>"
                    "<b>Name:</b> {name}<br/>"
                    "<b>Neighborhood:</b> {neighbourhood}"
                },
            ),
            height=600,
        )

    @staticmethod
    def display_map_heatmap(dataset: Dataset, map_style):
        """
        Display a heatmap of a DBSCAN cluster overlay on an actual map

        Args:
            dataset (Dataset): a Dataset object for the current instance
            map_style (str): map box type
        """

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            dataset.df_visualize_clustered,
            get_position=["longitude", "latitude"],
            get_weight="price",
            radiusPixels=50,
        )

        st.pydeck_chart(
            pdk.Deck(
                api_keys=Config.API_KEYS,
                layers=[heatmap_layer],
                initial_view_state=pdk.ViewState(
                    latitude=DatasetStats.get_by_price_mean(
                        dataset=dataset, column_name="latitude"
                    ),
                    longitude=DatasetStats.get_by_price_mean(
                        dataset=dataset, column_name="longitude"
                    ),
                    zoom=11,
                    pitch=0,
                ),
                map_style=Config.MAP_STYLES[map_style],
            ),
            height=600,
        )

    @staticmethod
    def display_map_hexagon_heatmap(dataset: Dataset, map_style):
        """
        Display a hexagon heatmap of a DBSCAN cluster overlay on an actual map

        Args:
            dataset (Dataset): a Dataset object for the current instance
            map_style (str): map box type
        """

        MEAN_LATITUDE = DatasetStats.get_by_price_mean(
            dataset=dataset, column_name="latitude"
        )
        MEAN_LONGITUDE = DatasetStats.get_by_price_mean(
            dataset=dataset, column_name="longitude"
        )

        hexagon_layer = pdk.Layer(
            "HexagonLayer",
            dataset.df_visualize_clustered,
            get_position=["longitude", "latitude"],
            radius=500,
            pickable=True,
            extruded=False,
        )

        st.pydeck_chart(
            pdk.Deck(
                api_keys=Config.API_KEYS,
                layers=[hexagon_layer],
                initial_view_state=pdk.ViewState(
                    latitude=MEAN_LATITUDE,
                    longitude=MEAN_LONGITUDE,
                    zoom=11,
                    pitch=0,
                ),
                map_style=Config.MAP_STYLES[map_style],
                tooltip={"html": "<b>Count:</b> {elevationValue}"},
            ),
            height=600,
        )

    @staticmethod
    def display_all_clusters_stats(dataset: Dataset, n_top_clusters):
        """
        Display all DBSCAN cluster stats which contains its overall stats,
        its neighborhood and sample listing of a cluster.

        Args:
            dataset (Dataset): a Dataset object for the current instance
            n_top_clusters (int): amount of cluster to display
        """

        with st.expander("View Cluster Statistics", expanded=False):
            for cluster_id in DatasetUtility.get_top_cluster_ids(
                dataset, n_top_clusters
            ):
                Plot.display_cluster_stats(dataset, cluster_id)
                Plot.display_cluster_neighborhood(dataset, cluster_id)
                Plot.display_cluster_sample_listing(dataset, cluster_id)

    @staticmethod
    def display_cluster_stats(dataset: Dataset, cluster_id):
        """
        Display a single cluster overall stats

        Args:
            dataset (Dataset): a Dataset object for the current instance
            cluster_id (int): an ID of cluster to display
        """

        N_ROWS = DatasetStats.get_cluster_rows(dataset, cluster_id).shape[0]
        MEAN_PRICE = DatasetStats.get_cluster_mean(
            dataset=dataset, cluster_id=cluster_id, column_name="price"
        )
        MEAN_REVIEW = DatasetStats.get_cluster_mean(
            dataset=dataset, cluster_id=cluster_id, column_name="number_of_reviews"
        )
        N_NEIGHBORHOODS = DatasetStats.get_cluster_n_unique(
            dataset=dataset, cluster_id=cluster_id, column_name="neighbourhood"
        )

        DISPLAY_INFO = (
            {"text": "Average Price", "value": f"฿{MEAN_PRICE:,.0f}"},
            {"text": "Average Reviews", "value": f"{MEAN_REVIEW:,.1f}"},
            {"text": "Neighborhoods", "value": f"{N_NEIGHBORHOODS}"},
        )

        st.write(f"**Cluster {cluster_id}** ({N_ROWS} listings)")

        columns = st.columns(len(DISPLAY_INFO))
        for idx, column in enumerate(columns):
            with column:
                st.metric(
                    DISPLAY_INFO[idx]["text"],
                    DISPLAY_INFO[idx]["value"],
                )

    @staticmethod
    def display_cluster_neighborhood(dataset: Dataset, cluster_id):
        """
        Display a list of neighborhood of a single cluster

        Args:
            dataset (Dataset): a Dataset object for the current instance
            cluster_id (int): an ID of cluster to display
        """

        N_ROWS = 10
        COLUMN_NAMES = ["neighbourhood"]

        df_neighborhood = DatasetStats.get_cluster_sample_unique_rows(
            dataset=dataset,
            cluster_id=cluster_id,
            column_names=COLUMN_NAMES,
            n_rows=N_ROWS,
        )
        n_neighborhood = df_neighborhood.shape[0]

        COLUMN_NAME = "neighbourhood"
        neighborhood_list = df_neighborhood[COLUMN_NAME].tolist()
        st.write("**Neighborhoods:** ", ", ".join(neighborhood_list))
        if n_neighborhood > N_ROWS:
            st.write(f"... and {n_neighborhood - N_ROWS} more.")

    @staticmethod
    def display_cluster_sample_listing(dataset: Dataset, cluster_id):
        """
        Display a sample listing of a single cluster

        Args:
            dataset (Dataset): a Dataset object for the current instance
            cluster_id (int): an ID of cluster to display
        """

        N_ROWS = 5
        COLUMN_NAMES = ["name", "neighbourhood", "price", "number_of_reviews"]

        st.write("**Sample listings:**")
        df_sample_listings = DatasetStats.get_cluster_sample_rows(
            dataset=dataset,
            cluster_id=cluster_id,
            column_names=COLUMN_NAMES,
            n_rows=N_ROWS,
        )
        st.dataframe(df_sample_listings, width="stretch")

    @staticmethod
    def display_all_kde_stats(dataset: Dataset, map_style):
        """
        Display all stats related to KDE

        Args:
            dataset (Dataset): a Dataset object for the current instance
            map_style (str): map box style
        """

        Plot.display_kde_map_scatter_plot(dataset, map_style)
        Plot.display_kde_density_stats(dataset)
        Plot.display_kde_high_density_locations(dataset)

    @staticmethod
    def display_kde_map_scatter_plot(dataset: Dataset, map_style):
        """
        display KDE scatter plot overlay on an actual map

        Args:
            dataset (Dataset): a Dataset object for the current instance
            map_style (str): map box style
        """

        MEAN_LATITUDE = DatasetStats.get_by_price_mean(
            dataset=dataset, column_name="latitude"
        )
        MEAN_LONGITUDE = DatasetStats.get_by_price_mean(
            dataset=dataset, column_name="longitude"
        )

        kde_layer = pdk.Layer(
            "ScatterplotLayer",
            dataset.df_kde,
            get_position=["longitude", "latitude"],
            get_color="density_color",
            get_radius=50,
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                api_keys=Config.API_KEYS,
                layers=[kde_layer],
                initial_view_state=pdk.ViewState(
                    latitude=MEAN_LATITUDE,
                    longitude=MEAN_LONGITUDE,
                    zoom=11,
                    pitch=0,
                ),
                map_style=Config.MAP_STYLES[map_style],
                tooltip={
                    "html": "<b>Density:</b> {density_formatted}<br/>"
                    "<b>Price:</b> ฿{price}<br/>"
                    "<b>Name:</b> {name}<br/>"
                    "<b>Neighborhood:</b> {neighbourhood}"
                },
            ),
            height=600,
        )

    @staticmethod
    def display_kde_density_stats(dataset: Dataset):
        """
        Display overall KDE density stats

        Args:
            dataset (Dataset): a Dataset object for the current instance
        """

        MIN_DENSITY = DatasetStats.get_kde_min_density(dataset)
        MEAN_DENSITY = DatasetStats.get_kde_mean_density(dataset)
        MAX_DENSITY = DatasetStats.get_kde_max_density(dataset)

        DISPLAY_INFO = (
            {"text": "Min Density", "value": f"{MIN_DENSITY:.4f}"},
            {"text": "Mean Density", "value": f"{MEAN_DENSITY:.4f}"},
            {"text": "Max Density", "value": f"{MAX_DENSITY:.4f}"},
        )

        columns = st.columns(len(DISPLAY_INFO))
        for idx, column in enumerate(columns):
            with column:
                st.metric(
                    DISPLAY_INFO[idx]["text"],
                    DISPLAY_INFO[idx]["value"],
                )

    @staticmethod
    def display_kde_high_density_locations(dataset: Dataset):
        """
        Display a list of top 10 of high density area

        Args:
            dataset (Dataset): a Dataset object for the current instance
        """

        N_ROWS = 10
        COLUMN_NAMES = [
            "name",
            "neighbourhood",
            "price",
            "density",
            "number_of_reviews",
        ]

        df_top_density = DatasetStats.get_kde_top_n_density(
            dataset=dataset, column_names=COLUMN_NAMES, n_rows=N_ROWS
        )

        st.dataframe(df_top_density, width="stretch")
