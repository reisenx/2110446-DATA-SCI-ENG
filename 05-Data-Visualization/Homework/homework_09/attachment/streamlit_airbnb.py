import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

st.set_page_config(page_title="Bangkok Airbnb Analysis", layout="wide")
st.title("Bangkok Airbnb Listings Analysis")


# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv("airbnb_listings.csv")

    # Clean and prepare data
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["latitude", "longitude", "price"])
    return data


# Load data
data = load_data()

## Sidebar code

# Sidebar filters
st.sidebar.header("Filters")

# Price range filter
max_price = 500
min_price = 30000
price_range = st.sidebar.slider(
    "Price Range (THB)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=500,
)


# DBSCAN clustering parameters
st.sidebar.header("DBSCAN Parameters")

st.sidebar.write("eps (degree)")
st.sidebar.write("#Create a slider for eps here")
eps_degrees = 0.002

st.sidebar.write("min_samples")
st.sidebar.write("#Create a slider for min_sample here")
min_samples = 3

st.sidebar.write("min_samples")
st.sidebar.write("#Create a slider for num_top_clusters")
num_top_clusters = 5

# Map style selection
map_style = st.sidebar.selectbox(
    "Select Base Map Style", options=["Dark", "Light", "Road", "Satellite"], index=0
)

# KDE parameters
st.sidebar.header("KDE Parameters")

st.sidebar.write("bandwidth")
st.sidebar.write("#Create a slider for bandwidth here")
bandwidth = 0.005


## Main panel code

# Define map style dictionary
MAP_STYLES = {
    "Dark": "mapbox://styles/mapbox/dark-v8",
    "Light": "mapbox://styles/mapbox/light-v8",
    "Road": "mapbox://styles/mapbox/streets-v8",
    "Satellite": "mapbox://styles/mapbox/satellite-v8",
}

# Filter data based on selections
filtered_data = data.copy()
filtered_data = filtered_data[
    (filtered_data["price"] >= price_range[0])
    & (filtered_data["price"] <= price_range[1])
]


# Main content - Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Listings", len(filtered_data))
with col2:
    st.metric("Average Price", f"฿{filtered_data['price'].mean():.0f}")
with col3:
    st.metric("Average Reviews", f"{filtered_data['number_of_reviews'].mean():.1f}")
with col4:
    st.metric("Neighborhoods", filtered_data["neighbourhood"].nunique())

# Price Distribution
st.header("Price Distribution")

fig_hist = px.histogram(
    filtered_data,
    x="price",
    nbins=100,  # You can adjust the number of bins
    title="Distribution of Listing Prices",
    labels={"price": "Price (THB)", "count": "Number of Listings"},
)

st.plotly_chart(fig_hist)

# Price by neighborhood
price_by_neighborhood = (
    filtered_data.groupby("neighbourhood")["price"].agg(["mean", "count"]).reset_index()
)
price_by_neighborhood.columns = ["neighbourhood", "avg_price", "listings_count"]

fig_scatter = px.scatter(
    price_by_neighborhood,
    x="listings_count",
    y="avg_price",
    text="neighbourhood",
    title="Average Price vs Number of Listings by Neighborhood",
    labels={"listings_count": "Number of Listings", "avg_price": "Average Price (THB)"},
)
fig_scatter.update_traces(textposition="top center")
st.plotly_chart(fig_scatter)

# Hotspot Analysis
st.header("Accommodation Hotspot Analysis")

try:
    # Perform DBSCAN clustering
    coords = filtered_data[["latitude", "longitude"]]
    db = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)

    # Add cluster labels to dataframe
    filtered_data["cluster"] = db.labels_

    # Analyze clusters
    clusters_count = filtered_data["cluster"].value_counts()
    clusters_count = clusters_count[clusters_count.index != -1]  # Exclude noise points
    top_clusters = clusters_count.head(num_top_clusters)

    # Generate colors for clusters
    unique_clusters = filtered_data[filtered_data["cluster"].isin(top_clusters.index)][
        "cluster"
    ].unique()
    colormap = plt.get_cmap("hsv")
    cluster_colors = {
        cluster: [int(x * 255) for x in colormap(i / len(unique_clusters))[:3]] + [160]
        for i, cluster in enumerate(unique_clusters)
    }

    # Create visualization dataframe
    viz_data = filtered_data[filtered_data["cluster"].isin(top_clusters.index)].copy()
    viz_data["color"] = viz_data["cluster"].map(cluster_colors)

    # Create scatterplot layer
    cluster_layer = pdk.Layer(
        "ScatterplotLayer",
        viz_data,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius=50,
        pickable=True,
    )

    # Create and display the map for scatterplot layer
    st.pydeck_chart(
        pdk.Deck(
            layers=[cluster_layer],
            initial_view_state=pdk.ViewState(
                latitude=filtered_data["latitude"].mean(),
                longitude=filtered_data["longitude"].mean(),
                zoom=11,
                pitch=0,
            ),
            map_style=MAP_STYLES[map_style],
            tooltip={
                "html": "<b>Cluster:</b> {cluster}<br/>"
                "<b>Price:</b> ฿{price}<br/>"
                "<b>Name:</b> {name}<br/>"
                "<b>Neighborhood:</b> {neighbourhood}"
            },
        ),
        height=600,
    )

    st.write("#Draw a heatmap for clusters here")
    # Create heatmap layer

    # Create and display the map for heatmap layer

    st.write("#Draw a hexagon map for clusters here")
    # Create hexagon layer

    # Create and display the map for hexagon layer

    # View Cluster Statistics
    st.subheader("Cluster Statistics")
    with st.expander("View Cluster Statistics", expanded=False):
        for cluster_id in top_clusters.index:
            cluster_data = viz_data[viz_data["cluster"] == cluster_id]
            avg_price = cluster_data["price"].mean()
            avg_reviews = cluster_data["number_of_reviews"].mean()

            st.write(f"**Cluster {cluster_id}** ({len(cluster_data)} listings)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Price", f"฿{avg_price:.0f}")
            with col2:
                st.metric("Average Reviews", f"{avg_reviews:.1f}")
            with col3:
                st.metric("Neighborhoods", cluster_data["neighbourhood"].nunique())

            st.write(
                "**Neighborhoods:**",
                ", ".join(cluster_data["neighbourhood"].unique()[:10]),
            )
            if cluster_data["neighbourhood"].nunique() > 10:
                st.write(f"... and {cluster_data['neighbourhood'].nunique() - 10} more")

            st.write("**Sample listings:**")
            sample_listings = cluster_data[
                ["name", "neighbourhood", "price", "number_of_reviews"]
            ].head(5)
            st.dataframe(sample_listings, use_container_width=True)

            st.divider()

except Exception as e:
    st.error(f"Error in clustering analysis: {e}")


# Kernel Density Estimation Analysis
st.header("Kernel Density Estimation (KDE) Analysis")

try:
    # Prepare coordinates
    coords = filtered_data[["latitude", "longitude"]].values

    # Fit KDE model
    st.write("#Call KernelDensity() with correct parameters here")
    kde = KernelDensity()  # Add correct parameters
    kde.fit(coords)

    # Calculate density score for each point
    log_density = kde.score_samples(coords)
    density = np.exp(log_density)

    # Add density scores to dataframe
    filtered_data["density"] = density
    filtered_data["density_normalized"] = (density - density.min()) / (
        density.max() - density.min()
    )

    # Format density for tooltip
    filtered_data["density_formatted"] = filtered_data["density"].apply(
        lambda x: f"{x:.4f}"
    )

    # Create color mapping based on density (blue = low, red = high)
    def density_to_color(density_normalized):
        """Convert normalized density to RGB color (blue to red gradient)"""
        r = int(density_normalized * 255)
        g = int((1 - abs(2 * density_normalized - 1)) * 255)
        b = int((1 - density_normalized) * 255)
        return [r, g, b, 160]  # Added alpha channel

    # Apply color mapping
    filtered_data["density_color"] = filtered_data["density_normalized"].apply(
        density_to_color
    )

    st.write("#Draw a scatter map for KDE here")
    # Create scatter layer for KDE

    # Create and display the map for scattermap layer

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Density", f"{filtered_data['density'].min():.4f}")
    with col2:
        st.metric("Mean Density", f"{filtered_data['density'].mean():.4f}")
    with col3:
        st.metric("Max Density", f"{filtered_data['density'].max():.4f}")

    # View top high-density locations
    with st.expander("View Highest Density Locations", expanded=False):
        st.subheader("Top 10 Highest Density Locations")
        top_density = filtered_data.nlargest(10, "density")[
            ["name", "neighbourhood", "price", "density", "number_of_reviews"]
        ].reset_index(drop=True)
        top_density["density"] = top_density["density"].apply(lambda x: f"{x:.4f}")
        st.dataframe(top_density, use_container_width=True)

except Exception as e:
    st.error(f"Error in KDE analysis: {e}")
