import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
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

# TODO: Adjust the min-max filter range to match the solution PDF file
# Price range filter
min_price = 128
max_price = 40000
price_range = st.sidebar.slider(
    label="Price Range (THB)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=500,
)


# DBSCAN clustering parameters
st.sidebar.header("DBSCAN Parameters")

# TODO: Implement the eps_degrees slider bar
eps_degrees = st.sidebar.slider(
    label="eps (degree)",
    min_value=0.001,
    max_value=0.005,
    value=0.002,
    step=0.001,
    format="%0.3f",
)

# TODO: Implement the min_samples slider bar
min_samples = st.sidebar.slider(
    label="min_samples", min_value=2, max_value=10, value=3, step=1
)

# TODO: Implement the num_top_clusters slider bar
num_top_clusters = st.sidebar.slider(
    label="Number of Top Cluster to Show", min_value=1, max_value=10, value=5, step=1
)

# Map style selection
map_style = st.sidebar.selectbox(
    "Select Base Map Style", options=["Dark", "Light", "Road", "Satellite"], index=0
)

# KDE parameters
st.sidebar.header("KDE Parameters")

# TODO: Implement the bandwidth slider bar
bandwidth = st.sidebar.slider(
    label="Bandwidth",
    min_value=0.001,
    max_value=0.020,
    value=0.005,
    step=0.001,
    format="%0.3f",
)


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
            api_keys={"mapbox": st.secrets.get("MAP_BOX_API_KEY")},
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

    # TODO: Implement a heatmap layer for clusters
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        viz_data,
        get_position=["longitude", "latitude"],
        get_weight="price",
        radiusPixels=50,
    )

    # TODO: Implement the map layer for heatmap layer
    st.pydeck_chart(
        pdk.Deck(
            api_keys={"mapbox": st.secrets.get("MAP_BOX_API_KEY")},
            layers=[heatmap_layer],
            initial_view_state=pdk.ViewState(
                latitude=filtered_data["latitude"].mean(),
                longitude=filtered_data["longitude"].mean(),
                zoom=11,
                pitch=0,
            ),
            map_style=MAP_STYLES[map_style],
        ),
        height=600,
    )

    # TODO: Implement the hexagon map layer for clusters
    # Create hexagon layer
    hexagon_layer = pdk.Layer(
        "HexagonLayer",
        viz_data,
        get_position=["longitude", "latitude"],
        radius=500,
        pickable=True,
        extruded=False,
    )

    # Create and display the map for hexagon layer
    st.pydeck_chart(
        pdk.Deck(
            api_keys={"mapbox": st.secrets.get("MAP_BOX_API_KEY")},
            layers=[hexagon_layer],
            initial_view_state=pdk.ViewState(
                latitude=filtered_data["latitude"].mean(),
                longitude=filtered_data["longitude"].mean(),
                zoom=11,
                pitch=0,
            ),
            map_style=MAP_STYLES[map_style],
            tooltip={"html": "<b>Count:</b> {elevationValue}"},
        ),
        height=600,
    )

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

    # TODO: Adds KDE parameters
    # Fit KDE model
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
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

    # TODO: Implement a scatter map for KDE
    # Create scatter layer for KDE
    kde_layer = pdk.Layer(
        "ScatterplotLayer",
        filtered_data,
        get_position=["longitude", "latitude"],
        get_color="density_color",
        get_radius=50,
        pickable=True,
    )

    # Create and display the map for scatter map layer
    st.pydeck_chart(
        pdk.Deck(
            api_keys={"mapbox": st.secrets.get("MAP_BOX_API_KEY")},
            layers=[kde_layer],
            initial_view_state=pdk.ViewState(
                latitude=filtered_data["latitude"].mean(),
                longitude=filtered_data["longitude"].mean(),
                zoom=11,
                pitch=0,
            ),
            map_style=MAP_STYLES[map_style],
            tooltip={
                "html": "<b>Density:</b> {density_formatted}<br/>"
                "<b>Price:</b> ฿{price}<br/>"
                "<b>Name:</b> {name}<br/>"
                "<b>Neighborhood:</b> {neighbourhood}"
            },
        ),
        height=600,
    )

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
