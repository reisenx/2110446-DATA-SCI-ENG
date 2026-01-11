import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from (videos.csv and category_id.json) and answer the questions.
"""


def Q1():
    """
    1. How many rows are there in the videos.csv after removing duplications?
    - To access 'videos.csv', use the path '/data/videos.csv'.
    """
    # Load data from videos.csv into a DataFrame.
    vdo_df = pd.read_csv("/data/videos.csv")

    # Remove duplicate rows.
    vdo_df = vdo_df.drop_duplicates()

    # Count amount of the remaining videos.
    return vdo_df.shape[0]


def Q2(vdo_df):
    """
    2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    # Condition to filter the videos with dislikes more than likes.
    FILTER_CONDITIONS = vdo_df["dislikes"] > vdo_df["likes"]

    # Count amount of the unique remaining videos.
    return vdo_df[FILTER_CONDITIONS].drop_duplicates(subset=["title"]).shape[0]


def Q3(vdo_df):
    """
    3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    """
    # Condition to filter the videos that are trending on 22 Jan 2018 with comments more than 10,000 comments.
    FILTER_CONDITIONS = (vdo_df["trending_date"] == "18.22.01") & (
        vdo_df["comment_count"] > 10000
    )

    # Count amount of the remaining videos.
    return vdo_df[FILTER_CONDITIONS].shape[0]


def Q4(vdo_df):
    """
    4. Which trending date that has the minimum average number of comments per VDO?
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    # Calculate average number of comments (group by trending date) then create a new column called "avg_comment_count".
    grouped_vdo_df = (
        vdo_df.groupby("trending_date")["comment_count"]
        .mean()
        .reset_index(name="avg_comment_count")
    )

    # Condition to filter the trending date that have minimum average comment count.
    FILTER_CONDITIONS = (
        grouped_vdo_df["avg_comment_count"] == grouped_vdo_df["avg_comment_count"].min()
    )

    # Return the trending date that have minimum average comment count.
    return grouped_vdo_df[FILTER_CONDITIONS]["trending_date"].iloc[0]


def Q5(vdo_df):
    """
    5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - You must load the additional data from 'category_id.json' into memory before executing any operations.
        - To access 'category_id.json', use the path '/data/category_id.json'.
    """
    # Load categories information from the JSON file.
    categories_df = pd.read_json("/data/category_id.json")

    # Create video categories mappings
    CATEGORIES_MAPPINGS = {}
    for data in categories_df["items"]:
        # Extract the category ID and name from raw value.
        category_id = int(data["id"])
        category_name = data["snippet"]["title"]

        # Store the mappings to a dict.
        CATEGORIES_MAPPINGS[category_id] = category_name

    # Use deep copy, so it won't affect the original DataFrame.
    modified_vdo_df = vdo_df.copy()

    # Create the category_name column by mapping values.
    modified_vdo_df["category_name"] = modified_vdo_df["category_id"].map(
        CATEGORIES_MAPPINGS
    )

    FILTER_CONDITIONS = modified_vdo_df["category_name"].isin(["Sports", "Comedy"])

    modified_vdo_df = modified_vdo_df[FILTER_CONDITIONS]

    # Create a pivot table to find total daily views of each category.
    vdo_pivot_table = modified_vdo_df.pivot_table(
        index=["trending_date"],
        columns=["category_name"],
        values=["views"],
        aggfunc="sum",
    )

    # Filter trending days with sports video is more than comedy video total daily views
    FILTER_CONDITIONS = (
        vdo_pivot_table[("views", "Sports")] > vdo_pivot_table[("views", "Comedy")]
    )
    vdo_pivot_table = vdo_pivot_table[FILTER_CONDITIONS]

    # Return number of videos.
    return vdo_pivot_table.shape[0]
