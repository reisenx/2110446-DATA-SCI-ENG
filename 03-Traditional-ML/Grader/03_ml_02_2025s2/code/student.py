import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning


class BankLogistic:
    def __init__(self, data_path):
        """
        Class constructor method.

        Args:
            data_path (string): CSV dataset path
        """
        # Initialization attributes
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

        # Additional attributes
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.model = None
        self.y_pred = None

    def Q1(self):
        """
        Q1: How many rows of data are there in total?
        """
        return self.df.shape[0]

    def Q2(self):
        """
        Q2: Return the tuple of numeric variables and categorical variables
            are presented in the dataset.
        """
        # Select columns by its data type.
        df_numeric = self.df.select_dtypes(include=["number"])
        df_categorical = self.df.select_dtypes(exclude=["number"])

        # Get column amount of numeric columns and categorical columns.
        n_col_numeric = df_numeric.shape[1]
        n_col_categorical = df_categorical.shape[1]

        # Return amount of numeric columns and categorical columns.
        return (n_col_numeric, n_col_categorical)

    def Q3(self):
        """
        Q3: Return the tuple of ratio the Class 0 (no)
            followed by Class 1 (yes) in 3 digits.
        """
        # Get ratio of both class in the dataset.
        class_ratio = self.df["y"].value_counts(normalize=True)

        # Rounded ratio into 3 decimal places
        negative_ratio = float(round(class_ratio["no"], 3))
        positive_ratio = float(round(class_ratio["yes"], 3))

        # Return as a tuple
        return (negative_ratio, positive_ratio)

    def Q4(self):
        """
        Q4: Remove duplicate records from the data.
            What are the shape of the dataset afterward?
        """
        self.df = self.df.drop_duplicates()
        return self.df.shape

    def Q5(self):
        """
        Q5: Do the following operations
            -   Replace "unknown" value with null.
            -   Remove features with more than 99% flat values.
            -   Split the dataset into training and testing sets with a 70:30 ratio.
                by stratification and use seed 0
            After these operations, return the tuple of X_train shape and X_test shape.
        """
        # Ensure that all processes in Q4 has already executed.
        self.Q4()

        # Replace all "unknown" in the entire DataFrame to null value.
        self.df = self.df.replace("unknown", np.nan)

        # Initialize list to store column name with 99% flat values.
        REMOVE_THRESHOLD = 0.99
        drop_cols = []

        # Iterate each column and find the one with 99% flat value.
        for col_name in self.df.columns:
            max_percentage = self.df[col_name].value_counts(normalize=True).max()
            if max_percentage >= REMOVE_THRESHOLD:
                drop_cols.append(col_name)

        # Drop the column with 99% flat value
        self.df = self.df.drop(columns=drop_cols)

        # Split features and target column.
        X = self.df.drop(columns=["y"])
        y = self.df["y"]

        # Split train dataset and test dataset by stratification.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )

        # Return shape of X_train and X_test
        return (self.X_train.shape, self.X_test.shape)

    def Q6(self):
        """
        Q6: Do the following operations
            -	For numeric variables, impute missing values using the mean.
            -	For categorical variables, impute missing values using the mode.
            -   Map the nominal data for the education variable using the following order:
                -   Map "illiterate" to 1
                -   Map "basic.4y" to 2
                -   Map "basic.6y" to 3
                -   Map "basic.9y" to 4
                -   Map "high.school" to 5
                -   Map "professional.course" to 6
                -   Map "university.degree" to 7
            After these operations, return the shape of X_train.
        """
        # Ensure that all processes in Q4 to Q5 has already executed.
        self.Q5()

        # Selecting numeric columns and categorical columns
        NUMERIC_COLS = self.X_test.select_dtypes(include=["number"]).columns
        CATEGORICAL_COLS = self.X_test.select_dtypes(exclude=["number"]).columns

        # For numerical columns, fill missing value with mean value.
        mean_values = self.X_train[NUMERIC_COLS].mean()
        self.X_train[NUMERIC_COLS] = self.X_train[NUMERIC_COLS].fillna(mean_values)

        mean_values = self.X_test[NUMERIC_COLS].mean()
        self.X_test[NUMERIC_COLS] = self.X_test[NUMERIC_COLS].fillna(mean_values)

        # For categorical columns, fill missing value with mode value.
        mode_values = self.X_train[CATEGORICAL_COLS].mode().iloc[0]
        self.X_train[CATEGORICAL_COLS] = self.X_train[CATEGORICAL_COLS].fillna(
            mode_values
        )

        mode_values = self.X_test[CATEGORICAL_COLS].mode().iloc[0]
        self.X_test[CATEGORICAL_COLS] = self.X_test[CATEGORICAL_COLS].fillna(
            mode_values
        )

        # Create mappings for ordinal category columns (education).
        ORDINAL_COLS = "education"
        EDUCATION_ORDER = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "professional.course": 6,
            "university.degree": 7,
        }

        # Apply mappings to the education column.
        self.X_train[ORDINAL_COLS] = self.X_train[ORDINAL_COLS].map(EDUCATION_ORDER)
        self.X_test[ORDINAL_COLS] = self.X_test[ORDINAL_COLS].map(EDUCATION_ORDER)

        # Apply one hot encoding for nominal categorical columns.
        NOMINAL_COLS = self.X_train.select_dtypes(exclude=["number"]).columns
        self.X_train = pd.get_dummies(self.X_train, columns=NOMINAL_COLS, dtype=int)
        self.X_test = pd.get_dummies(self.X_test, columns=NOMINAL_COLS, dtype=int)

        # Return shape of X_train
        return self.X_train.shape

    def Q7(self):
        """
        Q7: Train a Logistic Regression model with the following parameters
            -   Set "random_state" to 2025
            -   Set "class_weight" to "balanced"
            -   Set "max_iter" to 500.
            What is the macro F1 score of the model on the test data rounded
            in 2 decimal places.
        """
        # Ensure that all processes in Q4 to Q6 has already executed.
        self.Q6()

        # Ignore the convergence warning.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Create a Logistic Regression model
        model = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=2025
        )

        # Train a model
        model.fit(self.X_train, self.y_train)

        # Model prediction
        self.y_pred = model.predict(self.X_test)

        # Calculate and return macro F1 score
        macro_f1 = f1_score(self.y_test, self.y_pred, average="macro")
        return float(round(macro_f1, 2))
