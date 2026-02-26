import dataset

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


class DataPlot:
    @staticmethod
    def display_slider(min_value, max_value):
        value = st.sidebar.slider(
            label="Select Number of Clusters", min_value=min_value, max_value=max_value
        )
        return value

    @staticmethod
    def display_selection(dataset: dataset.Dataset):
        _, col, _ = st.columns([1, 3, 1])
        with col:
            selection = st.selectbox(
                label="Select Feature for Box Plot:", options=dataset.feature_names
            )
        return selection

    @staticmethod
    def display_box_plot(dataset: dataset.Dataset, selection):
        fig_box = px.box(
            dataset.df,
            x="Species",
            y=selection,
            color="Species",
            color_discrete_map=dataset.species_color,
            title=f"Distribution of {selection} by Species",
            labels={selection: selection, "Species": "Species"},
            category_orders={"Species": sorted(dataset.df["Species"].unique())},
        )

        fig_box.update_layout(
            title=f"Distribution of {selection} by Species",
            yaxis_title=selection,
            showlegend=True,
        )

        _, col, _ = st.columns([1, 3, 1])
        with col:
            st.plotly_chart(fig_box)

    @staticmethod
    def display_scatter_matrix(dataset: dataset.Dataset):
        fig_matrix = px.scatter_matrix(
            dataset.df,
            dimensions=dataset.feature_names,
            color="Species",
            color_discrete_map=dataset.species_color,
        )

        fig_matrix.update_layout(
            height=800, title="Features Relationships by Species", dragmode="select"
        )

        st.plotly_chart(fig_matrix, use_container_width=True)

    @staticmethod
    def display_correlations(dataset: dataset.Dataset):
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=dataset.get_correlation(),
                x=dataset.feature_names,
                y=dataset.feature_names,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=dataset.get_correlation(),
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False,
            )
        )

        fig_corr.update_layout(title="Feature Correlation Matrix")

        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.plotly_chart(fig_corr)

    @staticmethod
    def display_elbow_analysis(dataset: dataset.Dataset, max_clusters):
        fig_elbow = px.line(
            x=range(1, max_clusters + 1),
            y=dataset.get_elbow_analysis(max_clusters),
            markers=True,
            title="Elbow Method Analysis",
            labels={"x": "Number of Clusters", "y": "Inertia"},
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    @staticmethod
    def display_clustering_result(dataset: dataset.Dataset):
        fig_cluster = px.scatter(
            dataset.df,
            x="petal length (cm)",
            y="petal width (cm)",
            color="Cluster",
            title="KMeans Clustering Result",
            category_orders={"Cluster": sorted(dataset.df["Cluster"].unique())},
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    @staticmethod
    def display_actual_species(dataset: dataset.Dataset):
        fig_actual = px.scatter(
            dataset.df,
            x="petal length (cm)",
            y="petal width (cm)",
            color="Species",
            color_discrete_map=dataset.species_color,
            title="Actual Species Distribution",
        )
        st.plotly_chart(fig_actual, use_container_width=True)

    @staticmethod
    def display_confusion_matrix(dataset: dataset.Dataset):
        st.write("Confusion Matrix (Species vs Clusters):")
        st.write(dataset.get_confusion_matrix())

    @staticmethod
    def display_species_stats(dataset: dataset.Dataset):
        st.write(dataset.get_species_stats())

    @staticmethod
    def display_cluster_stats(dataset: dataset.Dataset):
        st.write(dataset.get_cluster_stats())
