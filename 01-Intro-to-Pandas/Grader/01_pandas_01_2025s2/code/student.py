import pandas as pd


def Q1(df):
    """
    For Q1, please return the shape of the data
    """
    return df.shape


def Q2(df):
    """
    For Q2, please return the max score of the data
    """
    return df["score"].max()


def Q3(df):
    """
    For Q3, please return the total student that have score equal or more than 80 points
    """
    return df[df["score"] >= 80].shape[0]


def Q4(df):
    """
    Otherwise, just return “No Output”
    """
    return "No Output"
