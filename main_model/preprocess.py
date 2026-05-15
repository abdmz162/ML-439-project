import pandas as pd
import numpy as np


class TitanicPreprocessor:
    def __init__(self):
        self.numeric_columns = []
        self.categorical_columns = []
        self.fill_values = {}
        self.category_maps = {}

    def fit(self, df):
        df = df.copy()

        drop_cols = ["name", "ticket", "cabin"]
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        if "survived" in df.columns:
            feature_df = df.drop(columns=["survived"])
        else:
            feature_df = df

        self.numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in self.numeric_columns:
            self.fill_values[col] = feature_df[col].median()

        for col in self.categorical_columns:
            mode_value = feature_df[col].mode()[0]
            self.fill_values[col] = mode_value

            categories = sorted(feature_df[col].fillna(mode_value).unique())
            self.category_maps[col] = {
                category: idx for idx, category in enumerate(categories)
            }

    def transform(self, df):
        df = df.copy()

        drop_cols = ["name", "ticket", "cabin"]
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        for col, value in self.fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        for col in self.categorical_columns:
            if col in df.columns:
                mapping = self.category_maps[col]
                df[col] = df[col].map(lambda x: mapping.get(x, -1))

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
