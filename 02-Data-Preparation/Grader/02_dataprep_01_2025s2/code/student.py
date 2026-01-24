import pandas as pd
from sklearn.model_selection import train_test_split

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic_to_student.csv) and answer the questions.
    (Note that the following functions already take the Titanic dataset as a DataFrame, so you don’t need to use read_csv.)
"""


def Q1(df):
    """
    Problem 1:
        How many rows are there in the "titanic_to_student.csv"?
    """
    return df.shape[0]


def Q2(df):
    """
    Problem 2:
        2.1 Drop variables with missing > 50%
        2.2 Check all columns except 'Age' and 'Fare' for flat values, drop the columns where flat value > 70%
        From 2.1 and 2.2, how many columns do we have left?
        Note:
        - Ensure missing values are considered in your calculation. If you use normalize in .value_counts(), please include dropna=False.
    """
    # Create a deep copy of the DataFrame
    cleaned_df = df.copy()

    # Calculate amount of rows
    dataset_rows = cleaned_df.shape[0]

    # Calculate drop threshold
    drop_threshold = 0.5 * dataset_rows

    # Drop columns with missing value more than 50%
    cleaned_df.dropna(axis=1, thresh=drop_threshold, inplace=True)

    # Ignore these columns
    IGNORE_COLUMNS = ("Age", "Fare")

    # Initialize a list to store names to drop.
    drop_columns = []

    # Calculate drop threshold
    drop_threshold = 0.7 * dataset_rows

    # Iterate each columns
    for column_name in cleaned_df.columns:
        # Skip the ignored columns.
        if column_name in IGNORE_COLUMNS:
            continue

        # Calculate the highest count of the current columns.
        highest_count = cleaned_df[column_name].value_counts(dropna=False).iloc[0]

        # If the count exceeds the threshold, append the column name to a list.
        if highest_count > drop_threshold:
            drop_columns.append(column_name)

    # Drop columns from the list.
    cleaned_df.drop(columns=drop_columns, inplace=True)

    # Return remaining columns amount.
    return cleaned_df.shape[1]


def Q3(df):
    """
    Problem 3:
        Remove all rows with missing targets (the variable "Survived")
        How many rows do we have left?
    """
    cleaned_df = df.dropna(subset=["Survived"])
    return cleaned_df.shape[0]


def Q4(df):
    """
    Problem 4:
        Handle outliers
        For the variable “Fare”, replace outlier values with the boundary values
        If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
        If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
        What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
        Hint: Use function round(_, 2)
    """
    # Calculate quantile
    q1 = df["Fare"].quantile(0.25)
    q3 = df["Fare"].quantile(0.75)

    # Calculate IQR
    iqr = q3 - q1

    # Calculate lower bound and upper bound.
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # Create a deep copy of a DataFrame
    cleaned_df = df.copy()

    # Handle low outlier.
    low_outlier_rows = cleaned_df["Fare"] < lower_bound
    cleaned_df.loc[low_outlier_rows, "Fare"] = lower_bound

    # Handle high outlier
    high_outlier_rows = cleaned_df["Fare"] > upper_bound
    cleaned_df.loc[high_outlier_rows, "Fare"] = upper_bound

    # Return new mean value of Fare column
    mean_fare = cleaned_df["Fare"].mean()
    return round(mean_fare, 2)


def Q5(df):
    """
    Problem 5:
        Impute missing value
        For number type column, impute missing values with mean
        What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
        Hint: Use function round(_, 2)
    """
    # Calculate average age.
    average_age = df["Age"].mean()

    # Create a deep copy of a DataFrame
    cleaned_df = df.copy()

    # Impute missing value by average age.
    cleaned_df["Age"].fillna(average_age)

    # Return new average age.
    average_age = cleaned_df["Age"].mean()
    return round(average_age, 2)


def Q6(df):
    """
    Problem 6:
        Convert categorical to numeric values
        For the variable “Embarked”, perform the dummy coding.
        What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
        Hint: Use function round(_, 2)
    """
    # Perform one-hot encoding
    one_hot_df = pd.get_dummies(df["Embarked"], prefix="Embarked")

    # Return the mean of Embarked_Q column
    mean_embarked_q = one_hot_df["Embarked_Q"].mean()
    return round(mean_embarked_q, 2)


def Q7(df):
    """
    Problem 7:
        Split train/test split with stratification using 70%:30% and random seed with 123
        Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
        What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
        Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection,
        Don't forget to impute missing values with mean.
    """
    # Separate features and targets from a dataset.
    features = df.drop(columns=["Survived"])
    target = df["Survived"]

    # Separate train dataset and test dataset by ratio 7:3 by using stratification.
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=123, stratify=target
    )

    # Calculate the proportion of survivors in the train dataset.
    train_dataset_rows = target_train.shape[0]
    train_dataset_survivors = (target_train == 1).sum()
    proportions = train_dataset_survivors / train_dataset_rows

    # Return proportion of survivors in the train dataset.
    return round(proportions, 2)
