import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score


class MushroomClassifier:
    def __init__(self, data_path):
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
        Q1: Before doing the data preparation,
            how many missing value are there in "gill-size" variables?
        """
        missing_values = self.df["gill-size"].isna().sum()
        return missing_values

    def Q2(self):
        """
        Q2: How many rows of data and variables after doing these operations?
            - Drop rows where the target ("label") variable is missing.
            - Drop the following variables:
                - id
                - gill-attachment
                - gill-spacing
                - gill-size
                - gill-color-rate
                - stalk-root
                - stalk-surface-above-ring
                - stalk-surface-below-ring
                - stalk-color-above-ring-rate
                - stalk-color-below-ring-rate
                - veil-color-rate
                - veil-type
        """
        # Drop rows where the target ("label") variable is missing.
        self.df = self.df.dropna(subset=["label"])

        # Drop the following columns.
        TARGET_COLS = [
            "id",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color-rate",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring-rate",
            "stalk-color-below-ring-rate",
            "veil-color-rate",
            "veil-type",
        ]
        self.df = self.df.drop(columns=TARGET_COLS, errors="ignore")

        # Return shape of the new DataFrame
        return self.df.shape

    def Q3(self):
        """
        Q3: Answer the quantity of "class_0" and "class1"  after doing operations
            in Q2 and these following operations.
            -   Fill missing values by adding the mean for numeric variables
                and the mode for nominal variables.
            -   Convert the label variable `e` (edible) to `1` and `p` (poisonous) to `0`
                and check the quantity of `class0` and `class1`.
        """
        # Ensure that all processes in Q2 has already executed.
        self.Q2()

        # Fill missing value on numerical columns with mean.
        NUMERIC_COLS = ["cap-color-rate"]
        mean_value = self.df[NUMERIC_COLS].mean()
        self.df[NUMERIC_COLS] = self.df[NUMERIC_COLS].fillna(mean_value)

        # Fill missing value on categorical columns with mode.
        CATEGORICAL_COLS = [
            "cap-shape",
            "cap-surface",
            "bruises",
            "odor",
            "stalk-shape",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ]
        mode_values = self.df[CATEGORICAL_COLS].mode().iloc[0]
        self.df[CATEGORICAL_COLS] = self.df[CATEGORICAL_COLS].fillna(mode_values)

        # Convert targets (label) into numerical values.
        MAPPINGS = {"e": 1, "p": 0}
        self.df["label"] = self.df["label"].map(MAPPINGS)

        # Count positive and negative class
        n_positive = (self.df["label"] == 1).sum()
        n_negative = (self.df["label"] == 0).sum()

        # Return counts as a tuple
        return (n_negative, n_positive)

    def Q4(self):
        """
        Q4: What are training dataset's shape ("X_train")
            and testing dataset's shape ("X_test")
            after doing all operations in Q2 to Q3
            and all of these following operations.
            -   Convert the nominal variable to numeric using
                a dummy code with `drop_first = True`.
            -   Split train/test with 20% test, stratify,
                and seed = 2020.
        """
        # Ensure that all processes in Q2 to Q3 has already executed.
        self.Q3()

        # Initialize a list of categorical columns
        CATEGORICAL_COLS = [
            "cap-shape",
            "cap-surface",
            "bruises",
            "odor",
            "stalk-shape",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ]

        # Apply one-hot encoding
        self.df = pd.get_dummies(
            self.df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int
        )

        # Features and target splitting
        X = self.df.drop(columns=["label"])
        y = self.df["label"]

        # Train dataset and test dataset splitting using stratification.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=2020, stratify=y
        )

        # Return a shape of train dataset and test dataset.
        return (self.X_train.shape, self.X_test.shape)

    def Q5(self):
        """
        Q5: Find the best parameter for Random Forest model using Grid Search
            on training data with 5 CV after complete all process in Q2 to Q4.
            Set GridSearchCV parameters as follows:
            - "criterion": ["gini", "entropy"]
            - "max_depth": [2, 3]
            - "min_samples_leaf": [2, 5]
            - "n_estimators": [100]
            - "random_state": [2020]
        """
        # Ensure that all processes in Q2 to Q4 has already executed.
        self.Q4()

        # Initialize parameters for GridSearchCV
        PARAM_GRID = {
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 3],
            "min_samples_leaf": [2, 5],
            "n_estimators": [100],
            "random_state": [2020],
        }

        # Initialize GridSearchCV object
        self.model = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=PARAM_GRID,
            cv=5,
            n_jobs=-1,
            scoring="f1_weighted",
        )

        # Begin grid search algorithm to search for the best model
        self.model.fit(self.X_train, self.y_train)

        # Returns tuple of all parameters values
        best_params = tuple(self.model.best_params_.values())
        return best_params

    def Q6(self):
        """
        Q6: After doing all process in Q2 to Q5, what is the value of
            macro F1 score rounded to 2 decimal places?
        """
        # Ensure that all processes in Q2 to Q5 has already executed.
        self.Q5()

        # Predict the test value using the model
        self.y_pred = self.model.predict(self.X_test)

        # Get a list of per-class F1 score.
        f1_scores = f1_score(self.y_test, self.y_pred, average=None)

        # Rounded to 2 decimal places and store it in the tuple.
        f1_scores = tuple(round(float(score), 2) for score in f1_scores)

        # Returns a tuple of pre-class F1 score.
        return f1_scores
