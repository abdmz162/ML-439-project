import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from logistic_regression import LogisticRegression

def calculate_metrics(y_true, y_pred):
    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Precision, Recall, F1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def main():
    # Load train and test data
    train_df = pd.read_csv('dev_df.csv')
    test_df = pd.read_csv('test_df.csv')

    # Preprocess training data and fit transformer
    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)

    # Train a new model for evaluation
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)

    # Preprocess test data
    X_test, y_test = preprocessor.transform_test(test_df)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()