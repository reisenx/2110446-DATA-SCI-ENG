import dataset
import plot

import streamlit as st


class Main:
    @staticmethod
    def main():
        Main.setup_page_config()
        n_clusters = Main.setup_page_sidebar(min_value=1, max_value=6)
        data = dataset.Dataset()

        st.header("1. Feature Distributions by Species")
        selection = plot.DataPlot.display_selection(data)
        plot.DataPlot.display_box_plot(data, selection)

        st.header("2. Feature Relationships")
        plot.DataPlot.display_scatter_matrix(data)

        st.header("3. Feature Correlations")
        plot.DataPlot.display_correlations(data)

        st.header("4. Elbow Analysis")
        plot.DataPlot.display_elbow_analysis(dataset=data, max_clusters=6)

        st.header("5. Clustering Analysis")
        data.perform_clustering(n_clusters)
        col_1, col_2 = st.columns(2)
        with col_1:
            st.subheader("Clustering Result")
            plot.DataPlot.display_clustering_result(data)
        with col_2:
            st.subheader("Actual Species")
            plot.DataPlot.display_actual_species(data)

        st.header("6. Clustering Performance")
        plot.DataPlot.display_confusion_matrix(data)

        st.header("7. Feature Statistics")
        col_3, col_4 = st.columns(2)
        with col_3:
            st.subheader("Statistics by Species")
            plot.DataPlot.display_species_stats(data)
        with col_4:
            st.subheader("Statistics by Cluster")
            plot.DataPlot.display_cluster_stats(data)

    @staticmethod
    def setup_page_config():
        st.set_page_config(layout="wide")
        st.title("Iris Dataset Analysis")

    @staticmethod
    def setup_page_sidebar(min_value, max_value):
        st.sidebar.header("Analysis Controls")
        value = plot.DataPlot.display_slider(min_value, max_value)
        return value


if __name__ == "__main__":
    Main.main()
