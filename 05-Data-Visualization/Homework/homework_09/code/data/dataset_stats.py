class DatasetStats:
    @staticmethod
    def get_by_price_mean(dataset, column_name):
        """
        Returns a mean value of the specified column of the filtered DataFrame
        where prices are within the specified range

        Args:
            dataset (Dataset): the dataset object
            column_name (str): the specified column name

        Returns:
            float: mean value of the specified column
        """

        return dataset.df_by_price_range[column_name].mean()

    @staticmethod
    def get_by_price_n_unique(dataset, column_name):
        """
        Returns amount of unique item in the specified column of the filtered DataFrame
        where prices are within the specified range

        Args:
            dataset (Dataset): the dataset object
            column_name (str): the specified column name

        Returns:
            int: amount of unique item in the specified column
        """

        return dataset.df_by_price_range[column_name].nunique()

    @staticmethod
    def get_cluster_mean(dataset, cluster_id, column_name):
        """
        Returns a mean value of specified column of a single cluster

        Args:
            dataset (Dataset): the dataset object
            cluster_id (int): a single cluster ID
            column_name (str): the specified column name

        Returns:
            float: mean value of the specified column of a single cluster
        """

        CLUSTER_COLUMN_NAME = "cluster"

        return dataset.df_clustered[
            dataset.df_clustered[CLUSTER_COLUMN_NAME] == cluster_id
        ][column_name].mean()

    @staticmethod
    def get_cluster_n_unique(dataset, cluster_id, column_name):
        """
        Returns amount of unique item in the specified column of a single cluster

        Args:
            dataset (Dataset): the dataset object
            cluster_id (int): a single cluster ID
            column_name (str): the specified column name

        Returns:
            int: amount of unique item in the specified column of a single column
        """

        CLUSTER_COLUMN_NAME = "cluster"

        return dataset.df_clustered[
            dataset.df_clustered[CLUSTER_COLUMN_NAME] == cluster_id
        ][column_name].nunique()

    @staticmethod
    def get_cluster_rows(dataset, cluster_id):
        """
        Return a DataFrame of a single cluster

        Args:
            dataset (Dataset): the dataset object
            cluster_id (int): a single cluster ID

        Returns:
            pd.DataFrame: DataFrame of a single cluster
        """

        CLUSTER_COLUMN_NAME = "cluster"

        return dataset.df_clustered[
            dataset.df_clustered[CLUSTER_COLUMN_NAME] == cluster_id
        ]

    @staticmethod
    def get_cluster_sample_rows(dataset, cluster_id, column_names, n_rows):
        """
        Returns a DataFrame of the selected columns of a single column name

        Args:
            dataset (Dataset): the dataset object
            cluster_id (int): a single cluster ID
            column_name (str): the specified column name
            n_rows (int): amount of rows to display

        Returns:
            pd.Dataframe: DataFrame of the selected columns of a single column name
        """

        df_single_cluster = DatasetStats.get_cluster_rows(dataset, cluster_id)[
            column_names
        ]
        n_cluster_rows = df_single_cluster.shape[0]

        return df_single_cluster.head(n=min(n_rows, n_cluster_rows))

    @staticmethod
    def get_cluster_sample_unique_rows(dataset, cluster_id, column_names, n_rows):
        """
        Returns a DataFrame of unique rows the selected columns of a single column name

        Args:
            dataset (Dataset): the dataset object
            cluster_id (int): a single cluster ID
            column_name (str): the specified column name
            n_rows (int): amount of rows to display

        Returns:
            pd.Dataframe: DataFrame of unique rows of the selected columns of a single column name
        """

        df_single_cluster_unique = DatasetStats.get_cluster_rows(dataset, cluster_id)[
            column_names
        ].drop_duplicates()
        n_cluster_unique_rows = df_single_cluster_unique.shape[0]

        return df_single_cluster_unique.head(min(n_rows, n_cluster_unique_rows))

    @staticmethod
    def get_kde_min_density(dataset):
        """
        Returns minimum density value of KDE dataset

        Args:
            dataset (Dataset): the dataset object

        Returns:
            float: minimum density value of KDE dataset
        """

        COLUMN_NAME = "density"

        return dataset.df_kde[COLUMN_NAME].min()

    @staticmethod
    def get_kde_max_density(dataset):
        """
        Returns maximum density value of KDE dataset

        Args:
            dataset (Dataset): the dataset object

        Returns:
            float: maximum density value of KDE dataset
        """

        COLUMN_NAME = "density"

        return dataset.df_kde[COLUMN_NAME].max()

    @staticmethod
    def get_kde_mean_density(dataset):
        """
        Returns mean density value of KDE dataset

        Args:
            dataset (Dataset): the dataset object

        Returns:
            float: mean density value of KDE dataset
        """

        COLUMN_NAME = "density"

        return dataset.df_kde[COLUMN_NAME].mean()

    @staticmethod
    def get_kde_top_n_density(dataset, column_names, n_rows):
        """
        Returns a DataFrame of the selected columns sorted by density

        Args:
            dataset (Dataset): the dataset object
            column_name (str): the specified column name
            n_rows (int): amount of rows to display

        Returns:
            pd.Dataframe: DataFrame of the selected columns sorted by density
        """

        DENSITY_COLUMN_NAME = "density"

        return dataset.df_kde.nlargest(n_rows, DENSITY_COLUMN_NAME)[
            column_names
        ].reset_index(drop=True)
