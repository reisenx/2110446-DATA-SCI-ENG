import matplotlib.pyplot as plt


class DatasetUtility:
    @staticmethod
    def get_top_cluster_ids(dataset, n_top_clusters):
        """
        Returns cluster IDs which are ranked at the top N among all clusters

        Args:
            dataset (Dataset): the dataset object
            n_top_clusters (int): amount of clusters to visualize

        Returns:
            pd.Index: cluster IDs which are ranked at the top N
        """

        COLUMN_NAME = "cluster"

        return (
            dataset.df_clustered[COLUMN_NAME].value_counts().head(n_top_clusters).index
        )

    @staticmethod
    def get_cluster_color_map(dataset, top_clusters_ids):
        """
        Returns a mapping dictionary that maps each unique cluster id
        to the corresponding HSV color

        Args:
            dataset (Dataset): the dataset object
            top_clusters_ids (pd.Index): IDs of the top selected unique clusters

        Returns:
            dict[int, list[int]]: cluster color mapping dict for visualization
        """

        cluster_color_map = {}
        for idx, cluster_id in enumerate(top_clusters_ids):
            # Calculate the scale in the HSV color system
            hsv_scale = idx / len(top_clusters_ids)

            # Get the base HSV color in range 0.0 to 1.0
            base_red, base_green, base_blue, _ = plt.get_cmap("hsv")(hsv_scale)

            # Convert the HSV color into the range 0 to 255
            red = int(255 * base_red)
            green = int(255 * base_green)
            blue = int(255 * base_blue)
            alpha = 160

            # Update the mapping dictionary
            cluster_color_map[cluster_id] = [red, green, blue, alpha]

        return cluster_color_map

    @staticmethod
    def get_color_from_density(density_normalized):
        """
        Returns an RGB color value converted from the normalized density value

        Args:
            density_normalized (float): the normalized KDE density value

        Returns:
            list[int]: the converted RGB color value
        """

        red = int(density_normalized * 255)
        green = int((1 - abs(2 * density_normalized - 1)) * 255)
        blue = int((1 - density_normalized) * 255)
        alpha = 160

        return [red, green, blue, alpha]
