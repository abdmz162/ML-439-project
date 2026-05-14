import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        self.age_median = None
        self.embarked_mode = None
        self.fare_mean = None
        self.means = None
        self.stds = None

    def fit(self, df):
        # Calculate statistics for imputation
        self.age_median = df['age'].median()
        self.embarked_mode = df['embarked'].mode()[0]
        self.fare_mean = df['fare'].mean()

        # For scaling, but we'll scale after imputation
        pass

    def transform(self, df):
        # Drop irrelevant columns
        df = df.drop(['passengerid', 'name', 'ticket', 'cabin'], axis=1)

        # Handle missing values
        df['age'] = df['age'].fillna(self.age_median)
        df['embarked'] = df['embarked'].fillna(self.embarked_mode)
        df['fare'] = df['fare'].fillna(self.fare_mean)

        # Encode categorical variables
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
        df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

        # Convert to numpy arrays
        X = df.drop('survived', axis=1).values
        y = df['survived'].values

        return X, y

    def fit_transform(self, df):
        self.fit(df)
        X, y = self.transform(df)

        # Standardize features
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        X_scaled = (X - self.means) / self.stds

        return X_scaled, y

    def transform_test(self, df):
        X, y = self.transform(df)
        X_scaled = (X - self.means) / self.stds
        return X_scaled, y