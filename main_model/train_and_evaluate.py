
import pandas as pd
import numpy as np

from preprocess import TitanicPreprocessor
from random_forest import RandomForest
from metrics import Metrics


np.random.seed(42)


def main():
    train_df = pd.read_csv("dev_df.csv")
    test_df = pd.read_csv("test_df.csv")

    preprocessor = TitanicPreprocessor()

    train_df = preprocessor.fit_transform(train_df)
    test_df = preprocessor.transform(test_df)

    X_train = train_df.drop(columns=["survived"]).values
    y_train = train_df["survived"].values

    X_test = test_df.drop(columns=["survived"]).values
    y_test = test_df["survived"].values

    model = RandomForest(
        n_estimators=20,
        max_depth=8,
        min_samples_split=4
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = Metrics.accuracy(y_test, predictions)
    matrix = Metrics.confusion_matrix(y_test, predictions)

    print("===== RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)

    output = pd.DataFrame({
        "Prediction": predictions
    })

    output.to_csv("predictions.csv", index=False)

    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()